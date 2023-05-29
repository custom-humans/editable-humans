import os
import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

from .neural_fields import NeuralField
from .tracer import SDFTracer
from .losses import GANLoss
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
from kaolin.ops.mesh import subdivide_trianglemesh

import wandb

class Trainer(nn.Module):

    def __init__(self, config, smpl_V, smpl_F, log_dir):

        super().__init__()

        # Set device to use
        self.device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.cfg = config
        self.use_2d = self.cfg.use_2d_from_epoch >= 0
        self.use_2d_nrm = self.cfg.use_nrm_dis

        self.log_dir = log_dir
        self.log_dict = {}

        self.smpl_F = smpl_F.to(self.device).detach()
        self.smpl_V = smpl_V.to(self.device).detach()

        self.epoch = 0
        self.global_step = 0

        self.init_model()
        self.init_optimizer()
        self.init_log_dict()


    def init_model(self):
        """Initialize model.
        """

        log.info("Initializing geometry neural field...")

        self.sdf_field = NeuralField(self.cfg,
                                     self.smpl_V, 
                                     self.smpl_F,
                                     self.cfg.shape_dim,
                                     1,
                                     self.cfg.shape_freq,
                                     self.cfg.shape_pca_dim).to(self.device)

        log.info("Initializing texture neural field...")

        self.rgb_field = NeuralField(self.cfg,
                                     self.smpl_V, 
                                     self.smpl_F,
                                     self.cfg.color_dim,
                                     3,
                                     self.cfg.color_freq,
                                     self.cfg.color_pca_dim,
                                     sigmoid=True).to(self.device)
        
        self.tracer = SDFTracer(self.cfg)


        if self.use_2d:
            log.info("Initializing RGB discriminators...")
            self.gan_loss_rgb = GANLoss(self.cfg, self.cfg.gan_loss_type, auxillary=True).to(self.device)
            if self.use_2d_nrm:
                log.info("Initializing normal discriminators...")
                self.gan_loss_nrm = GANLoss(self.cfg, self.cfg.gan_loss_type).to(self.device)



    def init_optimizer(self):
        """Initialize optimizer.
        """
    
        decoder_params = []
        decoder_params.extend(list(self.sdf_field.decoder.parameters()))
        decoder_params.extend(list(self.rgb_field.decoder.parameters()))
        dictionary_params = []
        dictionary_params.extend(list(self.sdf_field.dictionary.parameters()))
        dictionary_params.extend(list(self.rgb_field.dictionary.parameters()))

        params = []
        params.append({'params': decoder_params,
                          'lr': self.cfg.lr_decoder,
                          "weight_decay": self.cfg.weight_decay})
        params.append({'params': dictionary_params,
                            'lr': self.cfg.lr_codebook})
        
        
        self.optimizer = torch.optim.Adam(params,
                                    betas=(self.cfg.beta1, self.cfg.beta2))
        
        if self.use_2d:
            dis_params = list(self.gan_loss_rgb.discriminator.parameters())
            if self.use_2d_nrm:
                dis_params += list(self.gan_loss_nrm.discriminator.parameters())
            
            self.optimizer_d = torch.optim.Adam(dis_params,
                                    lr=self.cfg.lr_dis,
                                    betas=(0.0, self.cfg.beta2))
    def init_log_dict(self):
        """Custom logging dictionary.
        """
        self.log_dict['total_iter_count'] = 0
        # 3D Loss
        self.log_dict['Loss_3D/rgb_loss'] = 0
        self.log_dict['Loss_3D/nrm_loss'] = 0
        self.log_dict['Loss_3D/reco_loss'] = 0
        self.log_dict['Loss_3D/reg_loss'] = 0
        self.log_dict['Loss_3D/total_loss'] = 0
        
        # RGB Discriminator
        self.log_dict['total_2D_count'] = 0

        self.log_dict['RGB_dis/D_loss'] = 0
        self.log_dict['RGB_dis/penalty_loss']= 0
        self.log_dict['RGB_dis/logits_real']= 0
        self.log_dict['RGB_dis/logits_fake']= 0
        
        # Nrm Discriminator
        self.log_dict['Nrm_dis/penalty_loss'] = 0
        self.log_dict['Nrm_dis/loss_D'] = 0
        self.log_dict['Nrm_dis/logits_real'] = 0
        self.log_dict['Nrm_dis/logits_fake'] = 0
        
        # 2D Loss
        self.log_dict['Loss_2D/RGB_G_loss'] = 0 
        self.log_dict['Loss_2D/Nrm_G_loss'] = 0 


    def step(self, epoch, n_iter, data):
        """Training step.
            1. 3D forward
            2. 3D backward
            3. 2D forward
            4. 2D backward
        """
        # record stats
        self.epoch = epoch
        self.global_step = n_iter

        # Set inputs to device
        self.set_inputs(data)

        # Train
        self.optimizer.zero_grad()
        self.forward_3D()
        self.backward_3D()

        if self.use_2d and \
           epoch >= self.cfg.use_2d_from_epoch and \
           n_iter % self.cfg.train_2d_every_iter == 0:
            self.forward_2D_rgb()
            self.backward_2D_rgb()
            if self.use_2d_nrm:
                self.forward_2D_nrm()
                self.backward_2D_nrm()
            self.log_dict['total_2D_count'] += 1

        self.optimizer.step()
        self.log_dict['total_iter_count'] += 1

    def set_inputs(self, data):
        """Set inputs for training.
        """
        self.b_szie, self.n_vertice, _ = data['pts'].shape
        self.idx = data['idx'].to(self.device)

        self.pts = data['pts'].to(self.device)
        self.gts = data['sdf'].to(self.device)
        self.rgb = data['rgb'].to(self.device) 

        # Downsample normal for faster training
        self.nrm_pts = self.pts[:, :self.n_vertice//10].to(self.device)
        self.nrm = data['nrm'][:, :self.n_vertice//10].to(self.device)

        if self.use_2d:
            self.width =  data['rgb_image'].shape[2]
            
            self.label = data['label'].view(self.b_szie).to(self.device)
            self.ray_dir = data['ray_dir_image'].view(self.b_szie,-1,3).to(self.device)
            self.ray_ori = data['ray_ori_image'].view(self.b_szie,-1,3).to(self.device)
            self.gt_xyz = data['xyz_image'].view(self.b_szie,-1,3).to(self.device)
            self.gt_nrm = data['nrm_image'].view(self.b_szie,-1,3).to(self.device)
            self.gt_rgb = data['rgb_image'].view(self.b_szie,-1,3).to(self.device)
            self.gt_mask = data['mask_image'].view(self.b_szie,-1,1).to(self.device)

    def forward_3D(self):
        """Forward pass for 3D.
            predict sdf, rgb, nrm
        """
        self.pred_sdf, geo_h = self.sdf_field(self.pts, self.idx, return_h=True)
        self.pred_rgb = self.rgb_field(self.pts, self.idx)
        self.pred_nrm = self.sdf_field.finitediff_gradient(self.nrm_pts, self.idx)
        self.pred_nrm = F.normalize(self.pred_nrm, p=2, dim=-1, eps=1e-5)

    def backward_3D(self):
        """Backward pass for 3D.
            Compute 3D loss
        """
        total_loss = 0.0
        reco_loss = 0.0
        rgb_loss = 0.0
        reg_loss = 0.0

        reco_loss += torch.abs(self.pred_sdf - self.gts).mean()

        rgb_loss += torch.abs(self.pred_rgb - self.rgb).mean()

        #nrm_loss = torch.abs(1 - F.cosine_similarity(self.pred_nrm, self.nrm, dim=-1)).mean()
        nrm_loss = torch.abs(self.pred_nrm - self.nrm).mean()

        reg_loss += self.sdf_field.regularization_loss()
        reg_loss += self.rgb_field.regularization_loss()

        total_loss += reco_loss * self.cfg.lambda_sdf + \
                      rgb_loss * self.cfg.lambda_rgb + \
                      nrm_loss * self.cfg.lambda_nrm + \
                      reg_loss * self.cfg.lambda_reg

        total_loss.backward()

        # Update logs
        self.log_dict['Loss_3D/reco_loss'] += reco_loss.item()
        self.log_dict['Loss_3D/rgb_loss'] += rgb_loss.item()
        self.log_dict['Loss_3D/nrm_loss'] += nrm_loss.item()
        self.log_dict['Loss_3D/reg_loss'] += reg_loss.item()

        self.log_dict['Loss_3D/total_loss'] += total_loss.item()

    def forward_2D_rgb(self):
        """Forward pass for 2D rgb images.
           Fix geroemtry (3D coordinates) and random sample texture
        """
        x = self.gt_xyz
        hit = self.gt_mask

        self.rgb_2d = self.rgb_field.sample(x.detach(), self.idx) * hit

    def forward_2D_nrm(self):
        """Forward pass for 2D nrm images. Random sample geometry and output normal.
            This requires online ray tracing and is slow.
            Cached points can be used as an approximation.
        """
        if self.cfg.use_cached_pts:
            x = self.gt_xyz
            hit = self.gt_mask
        else:
            x, hit = self.tracer(self.sdf_field.sample, self.idx, self.ray_ori, self.ray_dir)

        _normal = self.sdf_field.finitediff_gradient(x, self.idx, sample=True)
        _normal = F.normalize(_normal, p=2, dim=-1, eps=1e-5)
        self.nrm_2d = _normal * hit   

    def backward_2D_rgb(self):
        """Backward pass for 2D rgb images.
            Compute 2D adversarial loss for the discriminator and generator.
        """
   
        total_2D_loss = 0.0

        # RGB GAN loss
        disc_in_fake = self.rgb_2d.view(self.b_szie, self.width, self.width, 3).permute(0,3,1,2)
        disc_in_real = (self.gt_rgb * self.gt_mask).view(self.b_szie, self.width, self.width, 3).permute(0,3,1,2)
        disc_in_real.requires_grad = True  # for R1 gradient penalty

        self.optimizer_d.zero_grad()
        d_loss, log = self.gan_loss_rgb(disc_in_real, disc_in_fake, mode='d', gt_label=self.label)
        d_loss.backward()
        self.optimizer_d.step()

        self.log_dict['RGB_dis/D_loss'] += log['loss_train/disc_loss']
        self.log_dict['RGB_dis/penalty_loss'] += log['loss_train/r1_loss']
        self.log_dict['RGB_dis/logits_real'] += log['loss_train/logits_real']
        self.log_dict['RGB_dis/logits_fake'] += log['loss_train/logits_fake']

        g_loss, log = self.gan_loss_rgb(None, disc_in_fake, mode='g')
        total_2D_loss += g_loss
        total_2D_loss.backward()

        self.log_dict['Loss_2D/RGB_G_loss'] += log['loss_train/g_loss']
    
    def backward_2D_nrm(self):
        """Backward pass for 2D normal images.
            Compute 2D adversarial loss for the discriminator and generator.
        """

        # Nrm GAN loss
        total_2D_loss = 0.0

        disc_in_fake = self.nrm_2d.view(self.b_szie, self.width, self.width, 3).permute(0,3,1,2)
        disc_in_real = (self.gt_nrm * self.gt_mask).view(self.b_szie, self.width, self.width, 3).permute(0,3,1,2)
        disc_in_real.requires_grad = True  # for R1 gradient penalty

        self.optimizer_d.zero_grad()
        d_loss, log = self.gan_loss_nrm(disc_in_real, disc_in_fake, mode='d')
        d_loss.backward()
        self.optimizer_d.step()

        self.log_dict['Nrm_dis/loss_D'] += log['loss_train/disc_loss']
        self.log_dict['Nrm_dis/penalty_loss'] += log['loss_train/r1_loss']
        self.log_dict['Nrm_dis/logits_real'] += log['loss_train/logits_real']
        self.log_dict['Nrm_dis/logits_fake'] += log['loss_train/logits_fake']
        
        g_loss, log = self.gan_loss_rgb(None, disc_in_fake, mode='g')
        total_2D_loss += g_loss
        total_2D_loss.backward()

        self.log_dict['Loss_2D/Nrm_G_loss'] += log['loss_train/g_loss']

    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['Loss_3D/total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['Loss_3D/total_loss'])
        self.log_dict['Loss_3D/reco_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | Reco loss: {:>.3E}'.format(self.log_dict['Loss_3D/reco_loss'])
        self.log_dict['Loss_3D/rgb_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['Loss_3D/rgb_loss'])
        self.log_dict['Loss_3D/nrm_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | nrm loss: {:>.3E}'.format(self.log_dict['Loss_3D/nrm_loss'])
        self.log_dict['Loss_3D/reg_loss'] /= self.log_dict['total_iter_count'] + 1e-6

        log.info(log_text)

        for key, value in self.log_dict.items():
            if ['RGB_dis', 'Nrm_dis', 'Loss_2D'].count(key.split('/')[0]) > 0:
                value /= self.log_dict['total_2D_count'] + 1e-6
            wandb.log({key: value}, step=step)
        self.init_log_dict()

    def write_images(self, i):
        """Write images to wandb.
        """    
        gen_img = self.rgb_2d.view(self.b_szie, self.width , self.width , 3).clone().detach().cpu().numpy()
        gt_img = (self.gt_rgb * self.gt_mask).view(self.b_szie, self.width , self.width , 3).clone().detach().cpu().numpy()
        wandb.log({"Generated Images": [wandb.Image(gen_img[i]) for i in range(self.b_szie)]}, step=i)
        wandb.log({"Ground Truth Images": [wandb.Image(gt_img[i]) for i in range(self.b_szie)]}, step=i)

        if self.use_2d_nrm:
            gen_nrm = self.nrm_2d.view(self.b_szie, self.width , self.width , 3).clone().detach().cpu().numpy() * 0.5 + 0.5
            gt_nrm = (self.gt_nrm * self.gt_mask).view(self.b_szie, self.width , self.width , 3).clone().detach().cpu().numpy() * 0.5 + 0.5
            gen_nrm = np.clip(gen_nrm, 0, 1)
            gt_nrm = np.clip(gt_nrm, 0, 1)
            wandb.log({"Generated Normals": [wandb.Image(gen_nrm[i]) for i in range(self.b_szie)]}, step=i)
            wandb.log({"Ground Truth Normals": [wandb.Image(gt_nrm[i]) for i in range(self.b_szie)]}, step=i)

    def save_checkpoint(self, full=True, replace=False):
        """Save the model checkpoint.
        """

        if replace:
            model_fname = os.path.join(self.log_dir, f'model-.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'model-{self.epoch:04d}.pth')

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'log_dir': self.log_dir
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            if self.use_2d:
                state['optimizer_d'] = self.optimizer_d.state_dict()
    

        state['sdf'] = self.sdf_field.state_dict()
        state['rgb'] = self.rgb_field.state_dict()
        if self.use_2d:
            state['D_rgb'] = self.gan_loss_rgb.state_dict()
            if self.use_2d_nrm:
                state['D_nrm'] = self.gan_loss_nrm.state_dict()

        log.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(state, model_fname)


    def load_checkpoint(self, fname):
        """Load checkpoint.
        """
        try:
            checkpoint = torch.load(fname, map_location=self.device)
            log.info(f'Loading model checkpoint from: {fname}')
        except FileNotFoundError:
            log.warning(f'No checkpoint found at: {fname}, model randomly initialized.')
            return

        # update meta info
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.log_dir = checkpoint['log_dir']
        
        self.sdf_field.load_state_dict(checkpoint['sdf'])
        self.rgb_field.load_state_dict(checkpoint['rgb'])
        if self.use_2d:
            if 'D_rgb' in checkpoint:
                self.gan_loss_rgb.load_state_dict(checkpoint['D_rgb'])
            if self.use_2d_nrm and 'D_nrm' in checkpoint:
                self.gan_loss_nrm.load_state_dict(checkpoint['D_nrm'])

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.use_2d:
                self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])

        log.info(f'Loaded checkpoint at epoch {self.epoch} with global step {self.global_step}.')

'''
#######################################################################################################################################
    
    def reconstruction(self, epoch, i, subdivide, res=300):
        
        torch.cuda.empty_cache()

        with torch.no_grad():
            h = self._marching_cubes (i, subdivide=subdivide, res=res)
        h.export(os.path.join(self.log_dir, '%03d_reco_src-%03d.obj' % (epoch, i)) )
        
        torch.cuda.empty_cache()


    def _marching_cubes (self, i, subdivide=True, res=300):

        width = res
        window_x = torch.linspace(-1., 1., steps=width, device='cuda')
        window_y = torch.linspace(-1., 1., steps=width, device='cuda')
        window_z = torch.linspace(-1., 1., steps=width, device='cuda')

        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z)).permute(1, 2, 3, 0).reshape(1, -1, 3).contiguous()

        
        # Debug smpl grid
        smpl_vertice = self.smpl_V[i]
        d = trimesh.Trimesh(vertices=smpl_vertice.cpu().detach().numpy(), 
                    faces=self.smpl_F.cpu().detach().numpy())
        d.export(os.path.join(self.log_dir, 'smpl_sub_%03d.obj' % (i)) )
        

        idx = torch.tensor([i], dtype=torch.long, device = torch.device('cuda')).view(1).detach()
        _points = torch.split(coord, int(2*1e6), dim=1)
        voxels = []
        for _p in _points:
            pred_sdf = self.sdf_field(_p, idx)
            voxels.append(pred_sdf)

        voxels = torch.cat(voxels, dim=1)
        voxels = voxels.reshape(1, width, width, width)
        
        vertices, faces = voxelgrids_to_trianglemeshes(voxels, iso_value=0.)
        vertices = ((vertices[0].reshape(1, -1, 3) - 0.5) / (width/2)) - 1.0
        faces = faces[0]

        if subdivide:
            vertices, faces = subdivide_trianglemesh(vertices, faces, iterations=1)

        pred_rgb = self.rgb_field(vertices, idx+1, pose_idx=idx)            
        
        h = trimesh.Trimesh(vertices=vertices[0].cpu().detach().numpy(), 
                faces=faces.cpu().detach().numpy(), 
                vertex_colors=pred_rgb[0].cpu().detach().numpy())

        # remove disconnect par of mesh
        connected_comp = h.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        h = max_comp
    
        trimesh.repair.fix_inversion(h)

        return h
'''