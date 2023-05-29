import os
import time
import copy
import pickle
import torch
import trimesh

import numpy as np
import logging as log
from tqdm import tqdm
from PIL import Image

from .tracer import SDFTracer
from ..ops.mesh import load_obj, point_sample, closest_tex
from ..utils.camera import *

from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
from kaolin.ops.mesh import subdivide_trianglemesh

class Evaluator(object):

    def __init__(self, config, log_dir, mode='valid'):
        super().__init__()

        self.cfg = config
        self.log_dir = log_dir
        self.mesh_dir = os.path.join(log_dir, mode, 'meshes')
        os.makedirs(self.mesh_dir, exist_ok=True)
        self.image_dir = os.path.join(log_dir, mode, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        self.sdf_field = None
        self.rgb_field = None

        self.tracer = SDFTracer(self.cfg)
        self.subdivide = self.cfg.subdivide
        self.res = self.cfg.grid_size


    def init_models(self, trainer):
        '''Initialize the models for evaluation.
        Args:
            sdf_field (SDFNet): the sdf field model from trainer.
            rgb_field (RGBNet): the rgb field model from trainer.
        '''

        self.sdf_field = copy.deepcopy(trainer.sdf_field)
        self.rgb_field = copy.deepcopy(trainer.rgb_field)
        self.smpl_F = trainer.smpl_F.clone().detach().cpu()

    def _marching_cubes (self, geo_idx=0, tex_idx=None, subdivide=True, res=300):
        '''Marching cubes to generate mesh.
        Args:
            geo_idx (int): the index of geometry to be generated.
            tex_idx (int): the index of texture to be generated.
            subdivide (bool): whether to subdivide the mesh.
            res (int): the resolution of the marching cubes.
        Returns:
            mesh (trimesh): the generated mesh.
        '''

        width = res
        window_x = torch.linspace(-1., 1., steps=width, device='cuda')
        window_y = torch.linspace(-1., 1., steps=width, device='cuda')
        window_z = torch.linspace(-1., 1., steps=width, device='cuda')

        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z, indexing='ij')).permute(1, 2, 3, 0).reshape(1, -1, 3).contiguous()

        
        # Debug smpl grid
        #smpl_vertice = self.sdf_field.get_smpl_vertices_by_idx(geo_idx)
        #d = trimesh.Trimesh(vertices=smpl_vertice.cpu().detach().numpy(), 
        #            faces=self.smpl_F.cpu().detach().numpy())
        #d.export(os.path.join(self.log_dir, 'smpl_sub_%03d.obj' % (geo_idx)) )
        
        if tex_idx is None:
            tex_idx = geo_idx
        geo_idx = torch.tensor([geo_idx], dtype=torch.long, device = torch.device('cuda')).view(1).detach()
        tex_idx = torch.tensor([tex_idx], dtype=torch.long, device = torch.device('cuda')).view(1).detach()

        _points = torch.split(coord, int(2*1e6), dim=1)
        voxels = []
        for _p in _points:
            pred_sdf = self.sdf_field(_p, geo_idx)
            voxels.append(pred_sdf)

        voxels = torch.cat(voxels, dim=1)
        voxels = voxels.reshape(1, width, width, width)
        
        vertices, faces = voxelgrids_to_trianglemeshes(voxels, iso_value=0.)
        vertices = ((vertices[0].reshape(1, -1, 3) - 0.5) / (width/2)) - 1.0
        faces = faces[0]

        if subdivide:
            vertices, faces = subdivide_trianglemesh(vertices, faces, iterations=1)

        pred_rgb = self.rgb_field(vertices, tex_idx, pose_idx=geo_idx)            
        
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
    
    def _get_camera_rays(self, n_views=4, fov=20, width=1024):
        '''Get camera rays for rendering.
        Args:
            n_views (int): the number of views.
            fov (float): the field of view.
            width (int): the width of the image.
        Returns:
            ray_o_images : the origin of the rays of n_views*height*width*3
            ray_d_images : the direction of the rays of n_views*height*width*3
        '''
            
        look_at = torch.zeros( (n_views, 3), dtype=torch.float32, device=torch.device('cuda'))
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=torch.device('cuda')).repeat(n_views, 1,)
        angle = torch.linspace(0, 2*np.pi, n_views+1)[:-1]
        camera_position = torch.stack( (2*torch.sin(angle), torch.zeros_like(angle), 2*torch.cos(angle)), dim=1).cuda()

        ray_o_images = []
        ray_d_images = []
        for i in range(n_views):
            camera = Camera.from_args(eye=camera_position[i],
                                      at=look_at[i],
                                      up=camera_up_direction[i],
                                      fov=fov,
                                      width=width,
                                      height=width,
                                      dtype=torch.float32)

            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height, device=torch.device('cuda'))

            ray_orig, ray_dir = \
                generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid)

            ray_o_images.append(ray_orig.reshape(camera.height, camera.width, -1))
            ray_d_images.append(ray_dir.reshape(camera.height, camera.width, -1))

        return torch.stack(ray_o_images, dim=0), torch.stack(ray_d_images, dim=0)

    def reconstruction(self, idx, epoch=None):
        '''
        Reconstruct the mesh the idx-th subject.
        '''
        if epoch is None:
            epoch = 0
        log.info(f"Reconstructing {idx}th mesh at epoch {epoch}...")
        start = time.time()
        
        with torch.no_grad():
            h = self._marching_cubes (geo_idx=idx, subdivide=self.subdivide, res=self.res)
        
        h.export(os.path.join(self.mesh_dir, '%03d_reco_src-%03d.obj' % (epoch, idx)) )
        end = time.time()
        log.info(f"Reconstruction finished in {end-start} seconds.")

    def render_2D(self, idx, epoch=None):
        '''
        Render the 2D images of the idx-th subject.
        '''
        torch.cuda.empty_cache()

        log.info(f"Rendering {idx}th subject at epoch {epoch}...")
        start = time.time()

        with torch.no_grad():

            ray_o_images, ray_d_images = self._get_camera_rays(n_views=self.cfg.n_views, fov=self.cfg.fov, width=self.cfg.width)
            _idx = torch.tensor([idx], dtype=torch.long, device = torch.device('cuda')).repeat(self.cfg.n_views).detach()
            x, hit = self.tracer(self.sdf_field.forward, _idx,
                              ray_o_images.view(self.cfg.n_views, -1, 3),
                              ray_d_images.view(self.cfg.n_views, -1, 3))
            log.info(f"Rat tracing finished in {time.time()-start} seconds.")
            start = time.time()
            rgb_2d = self.rgb_field.forward(x.detach(), _idx) * hit

        rgb_img = rgb_2d.reshape(self.cfg.n_views, self.cfg.width, self.cfg.width, 3).cpu().detach().numpy() * 255

        for i in range(self.cfg.n_views):
            Image.fromarray(rgb_img[i].astype(np.uint8)).save(
                os.path.join(self.image_dir, '%03d_render_src-%03d_view-%03d.png' % (epoch, idx, i)) )

        log.info(f"Rendering finished in {time.time()-start} seconds.")
        render_dict = {'coord': x.cpu().detach(), 'rgb': rgb_2d.cpu().detach(), 'mask': hit.cpu().detach()}
        with open(os.path.join(self.image_dir, 'render_dict.pkl'), 'wb') as f:
            pickle.dump(render_dict, f)

        return render_dict


    def reposing(self, idx, target_smpl_obj, epoch=None):
        '''
        Reconstruct the mesh the idx-th subject. given the target smpl obj.
        '''
        
        if epoch is None:
            epoch = 0
        smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
        log.info(f"Reposing {idx}th mesh at epoch {epoch}...")
        start = time.time()

        with torch.no_grad():

            tmp_smpl_V = self.sdf_field.get_smpl_vertices_by_idx(idx)

            self.sdf_field.replace_smpl_vertices_by_idx(idx, smpl_V)
            self.rgb_field.replace_smpl_vertices_by_idx(idx, smpl_V)

            h = self._marching_cubes (geo_idx=idx, subdivide=self.subdivide, res=self.res)

            self.sdf_field.replace_smpl_vertices_by_idx(idx, tmp_smpl_V)
            self.rgb_field.replace_smpl_vertices_by_idx(idx, tmp_smpl_V)

        h.export(os.path.join(self.mesh_dir, '%03d_repose_src-%03d.obj' % (epoch, idx)) )
        end = time.time()
        log.info(f"Reposing finished in {end-start} seconds.")

    def transfer_features(self, src_idx, tar_idx, vert_idx=None):
        '''
        Copy the features from src_idx to tar_idx at vert_idx.
        '''
        with torch.no_grad():
            src_geo = self.sdf_field.get_feature_by_idx(src_idx, vert_idx=vert_idx).clone()
            src_tex = self.rgb_field.get_feature_by_idx(src_idx, vert_idx=vert_idx).clone()
            self.sdf_field.replace_feature_by_idx(tar_idx, src_geo, vert_idx=vert_idx)
            self.rgb_field.replace_feature_by_idx(tar_idx, src_tex, vert_idx=vert_idx)


    def fitting_3D(self, code_idx, target_mesh, target_smpl_obj, num_steps=300, fit_nrm=False, fit_rgb=False):
        """Fitting the latent code to the target mesh.
           Store the optimzed code in the code_idx-th entry of the codebook.
        """

        torch.cuda.empty_cache()

        geo_code = self.sdf_field.get_mean_feature().clone().unsqueeze(0).detach().data
        tex_code = self.rgb_field.get_mean_feature().clone().unsqueeze(0).detach().data

        geo_code.requires_grad = True
        tex_code.requires_grad = True

        V, F, texv, texf, mats = load_obj(target_mesh, load_materials=True)
        smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
        smpl_V = smpl_V.cuda()

        params = []
        params.append({'params': geo_code, 'lr': 0.005})
        params.append({'params': tex_code, 'lr': 0.01})

        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        loop = tqdm(range(num_steps))
        log.info(f"Start fitting latent code to the target mesh...")
        for i in loop:
            coord_1 = point_sample(V.cuda(), F.cuda(), ['near', 'trace', 'rand'], 20000, 0.01)
            coord_2 = point_sample(smpl_V, self.smpl_F.cuda(), ['near', 'trace'], 50000, 0.2)
            coord = torch.cat((coord_1, coord_2), dim=0)
            rgb, nrm, sdf = closest_tex(V.cuda(), F.cuda(), texv.cuda(), texf.cuda(), mats, coord.cuda())
            coord = coord.unsqueeze(0)
            sdf = sdf.unsqueeze(0)
            rgb = rgb.unsqueeze(0)
            nrm = nrm.unsqueeze(0)

            sdf_loss = torch.tensor(0.0).cuda()
            nrm_loss = torch.tensor(0.0).cuda()
            rgb_loss = torch.tensor(0.0).cuda()

            optimizer.zero_grad()

            pred_sdf = self.sdf_field.forward_fitting(coord, geo_code, smpl_V.unsqueeze(0))
            sdf_loss += torch.abs(pred_sdf - sdf).mean()

            if fit_rgb:
                pred_rgb = self.rgb_field.forward_fitting(coord, tex_code, smpl_V.unsqueeze(0))
                rgb_loss += torch.abs(pred_rgb - rgb).mean()

            if fit_nrm:
                pred_nrm = self.sdf_field.normal_fitting(coord, tex_code, smpl_V.unsqueeze(0))
                nrm_loss += torch.abs(pred_nrm - nrm).mean()
            

            loss = 10*sdf_loss + rgb_loss + nrm_loss
            loss.backward()
            optimizer.step()
            loop.set_description('Step [{}/{}] Total Loss: {:.4f} - L1:{:.4f} - RGB:{:.4f} - NRM:{:.4f}'
                           .format(i, num_steps, loss.item(), sdf_loss.item(), rgb_loss.item(), nrm_loss.item()))

        log.info(f"Fitting finished. Store the optimized code and the new SMPL pose in the codebook.")

        with torch.no_grad():
            self.sdf_field.replace_feature_by_idx(code_idx, geo_code)
            self.rgb_field.replace_feature_by_idx(code_idx, tex_code)
            self.sdf_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)
            self.rgb_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)


    def fitting_2D(self, code_idx, target_dict, target_smpl_obj=None, num_steps=500):
        """Fitting the color latent code to the rendered images
           Store the optimzed code in the code_idx-th entry of the codebook.
        """

        torch.cuda.empty_cache()

        tex_code = self.rgb_field.get_feature_by_idx(code_idx).clone().unsqueeze(0).detach().data
        tex_code.requires_grad = True

        rgb = target_dict['rgb'].cuda()
        coord = target_dict['coord'].cuda()
        mask = target_dict['mask'].cuda()

        b_size = rgb.shape[0] # b_size = n_views


        inputs = []
        targets = []
        for i in range(b_size):
            _xyz = coord[i]
            _rgb = rgb[i]
            _mask = mask[i, :, 0]
            inputs.append(_xyz[_mask].view(1,-1,3))
            targets.append(_rgb[_mask].view(1,-1,3))

        inputs = torch.cat(inputs, dim=1)
        targets = torch.cat(targets, dim=1)

        if target_smpl_obj is not None:
            smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
            smpl_V = smpl_V.cuda()
        else:
            smpl_V = self.rgb_field.get_smpl_vertices_by_idx(code_idx)

        params = []
        params.append({'params': tex_code, 'lr': 0.005})

        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        loop = tqdm(range(num_steps))


        for i in loop:

            rgb_loss = torch.tensor(0.0).cuda()

            optimizer.zero_grad()

            pred_rgb = self.rgb_field.forward_fitting(inputs, tex_code, smpl_V.unsqueeze(0))
            rgb_loss += torch.abs(pred_rgb - targets).mean()

            rgb_loss.backward()
            optimizer.step()
            loop.set_description('Step [{}/{}] Total Loss: {:.4f}'.format(i, num_steps, rgb_loss.item()))

        with torch.no_grad():
            self.rgb_field.replace_feature_by_idx(code_idx, tex_code)
            #self.rgb_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)
