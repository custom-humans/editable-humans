import os
import h5py
import numpy as np
import json
from tqdm import tqdm
import argparse
import torch
import pickle
import kaolin as kal
from kaolin.render.camera import *

from lib.utils.camera import *
from lib.ops.mesh import *
from smplx import SMPLX

SMPL_PATH = 'smplx/'

NUM_SAMPLES = 3000000

N_VIEWS = 4
FOV = 20
HEIGHT = 1024
WIDTH = 1024
RATIO = 1.0

N_JOINTS = 25
HALF_PATCH_SIZE = 64

def _get_smpl_vertices(smpl_data):
    device = torch.device('cuda')
    param_betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_poses = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            
    param_expression = torch.tensor(smpl_data['expression'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_jaw_pose = torch.tensor(smpl_data['jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_leye_pose = torch.tensor(smpl_data['leye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_reye_pose = torch.tensor(smpl_data['reye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()


    body_model = SMPLX(model_path=SMPL_PATH, gender='male', use_pca=True, num_pca_comps=12, flat_hand_mean=True).to(device)
                
    J_0 = body_model(body_pose = param_poses, betas=param_betas).joints.contiguous().detach()


    output = body_model(betas=param_betas,
                                   body_pose=param_poses,
                                   transl=-J_0[:,0,:],
                                   left_hand_pose=param_left_hand_pose,
                                   right_hand_pose=param_right_hand_pose,
                                   expression=param_expression,
                                   jaw_pose=param_jaw_pose,
                                   leye_pose=param_leye_pose,
                                   reye_pose=param_reye_pose,
                                   )
    return output.vertices.contiguous()[0].detach(), \
           output.joints.contiguous()[0].detach()[:25]


#########################################################################################################################

def main(args):
    device = torch.device('cuda')

    outfile = h5py.File(os.path.join(args.output_path), 'w')

    subject_list  = [x for x in sorted(os.listdir(args.input_path)) if os.path.isdir(os.path.join(args.input_path, x))]
    num_subjects = len(subject_list)

    outfile.create_dataset( 'num_subjects', data=num_subjects, dtype=np.int32)



    dataset_pts = outfile.create_dataset( 'pts', shape=(num_subjects, NUM_SAMPLES*6, 3),
                                 chunks=True, dtype=np.float32)
    dataset_rgb = outfile.create_dataset( 'rgb',shape=(num_subjects, NUM_SAMPLES*6, 3),
                                 chunks=True, dtype=np.float32)
    dataset_nrm = outfile.create_dataset( 'nrm', shape=(num_subjects, NUM_SAMPLES*6, 3),
                                 chunks=True, dtype=np.float32)
    dataset_d = outfile.create_dataset( 'd', shape=(num_subjects, NUM_SAMPLES*6, 1),
                                 chunks=True, dtype=np.float32)
    

    dataset_smpl_v = outfile.create_dataset( 'smpl_v', shape=(num_subjects, 10475, 3),
                                 chunks=True, dtype=np.float32)

    dataset_ray_ori_image = outfile.create_dataset( 'ray_ori_image', shape=(num_subjects, N_JOINTS*4,
                                 HALF_PATCH_SIZE*2, HALF_PATCH_SIZE*2, 3),
                                 chunks=True, dtype=np.float32)
    
    dataset_ray_dir_image = outfile.create_dataset( 'ray_dir_image', shape=(num_subjects, N_JOINTS*4,
                                 HALF_PATCH_SIZE*2, HALF_PATCH_SIZE*2, 3),
                                 chunks=True, dtype=np.float32)


    dataset_xyz_image = outfile.create_dataset( 'xyz_image', shape=(num_subjects, N_JOINTS*4,
                                 HALF_PATCH_SIZE*2, HALF_PATCH_SIZE*2, 3),
                                 chunks=True, dtype=np.float32)
    dataset_nrm_image = outfile.create_dataset( 'nrm_image', shape=(num_subjects, N_JOINTS*4,
                                 HALF_PATCH_SIZE*2, HALF_PATCH_SIZE*2, 3),
                                 chunks=True, dtype=np.float32)
    dataset_rgb_image = outfile.create_dataset( 'rgb_image', shape=(num_subjects, N_JOINTS*4,
                                 HALF_PATCH_SIZE*2, HALF_PATCH_SIZE*2, 3),
                                 chunks=True, dtype=np.float32)
    dataset_mask_image = outfile.create_dataset( 'mask_image', shape=(num_subjects, N_JOINTS*4,\
                                 HALF_PATCH_SIZE*2, HALF_PATCH_SIZE*2, 1),
                                    chunks=True, dtype=np.bool)
    
    for s, subject in enumerate(tqdm(subject_list)):
        subject_path = os.path.join(args.input_path, subject)
        json_file  = [x for x in sorted(os.listdir(subject_path)) if x.endswith('.json')][0]
        filename = json_file.split('.')[0]

        smpl_data = json.load(open(os.path.join(subject_path, filename+'.json')))
        smpl_V, smpl_J = _get_smpl_vertices(smpl_data)
        with open('data/smpl_mesh.pkl', 'rb') as f:
            smpl_mesh = pickle.load(f)

        smpl_F = smpl_mesh['smpl_F'].cuda().detach()


        mesh_data = os.path.join(subject_path, filename+'.obj')
        out = load_obj(mesh_data, load_materials=True)
        V, F, texv, texf, mats = out
        FN = per_face_normals(V, F).cuda()
        

        pts1 = point_sample( V.cuda(), F.cuda(), ['near', 'near', 'trace'], NUM_SAMPLES, 0.01)
        pts2 = point_sample(smpl_V, smpl_F, ['rand', 'near', 'trace'], NUM_SAMPLES, 0.1)

        rgb1, nrm1, d1 = closest_tex(V.cuda(), F.cuda(), 
                                           texv.cuda(), texf.cuda(), mats, pts1.cuda())
        rgb2, nrm2, d2 = closest_tex(V.cuda(), F.cuda(), 
                                               texv.cuda(), texf.cuda(), mats, pts2.cuda())

            
        look_at = torch.zeros( (N_VIEWS, 3), dtype=torch.float32, device=device)


        camera_position = torch.tensor( [ [0, 0, 2],
                                             [2, 0, 0],
                                             [0, 0, -2],
                                             [-2, 0, 0]  ]  , dtype=torch.float32, device=device)

        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device).repeat(N_VIEWS, 1,)

        cam_transform = generate_transformation_matrix(camera_position, look_at, camera_up_direction)
        cam_proj = generate_perspective_projection(FOV, RATIO)

        face_vertices_camera, face_vertices_image, face_normals = \
                kal.render.mesh.prepare_vertices(
                V.unsqueeze(0).repeat(N_VIEWS, 1, 1).cuda(),
                F.cuda(), cam_proj.cuda(), camera_transform=cam_transform
            )
        face_uvs = texv[texf[...,:3]].unsqueeze(0).cuda()

         ### Perform Rasterization ###
            # Construct attributes that DIB-R rasterizer will interpolate.
            # the first is the UVS associated to each face
            # the second will make a hard segmentation mask
        face_attributes = [
                V[F].unsqueeze(0).cuda().repeat(N_VIEWS, 1, 1, 1),
                face_uvs.repeat(N_VIEWS, 1, 1, 1),
                FN.unsqueeze(0).unsqueeze(2).repeat(N_VIEWS, 1, 3, 1),
        ]            

        padded_joints = torch.nn.functional.pad(
        smpl_J.unsqueeze(0).repeat(N_VIEWS, 1, 1), (0, 1), mode='constant', value=1.)

        joints_camera = (padded_joints @ cam_transform)
        # Project the vertices on the camera image plan
        jonts_image = perspective_camera(joints_camera, cam_proj.cuda())
        jonts_image = ((jonts_image) * torch.tensor([1, -1], device=device)  + 1 ) * \
                           torch.tensor([WIDTH//2, HEIGHT//2], device=device)
        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd
        image_features, face_idx = kal.render.mesh.rasterize(
        HEIGHT, WIDTH, face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, backend='cuda', multiplier=1000)

        coords, uv, normal= image_features

        TM = torch.zeros((N_VIEWS, HEIGHT, WIDTH, 1), dtype=torch.long, device=device)

        rgb = sample_tex(uv.view(-1, 2), TM.view(-1), mats).view(N_VIEWS, HEIGHT, WIDTH, 3)
        mask = (face_idx != -1).unsqueeze(-1)


        ray_dir_patches = []
        ray_ori_patches = []
        xyz_patches = []
        rgb_patches = []
        nrm_patches = []
        mask_patches = []

        for i in range(N_VIEWS):

            camera = Camera.from_args(eye=camera_position[i],
                                      at=look_at[i],
                                      up=camera_up_direction[i],
                                      fov=FOV,
                                      width=WIDTH,
                                      height=HEIGHT,
                                      dtype=torch.float32)

            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height, device=device)

            ray_orig, ray_dir = \
                generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid)

            ray_orig = ray_orig.reshape(camera.height, camera.width, -1)
            ray_dir = ray_dir.reshape(camera.height, camera.width, -1)

            for j in range(N_JOINTS):
                x = min (max( int(jonts_image[i, j, 0]), HALF_PATCH_SIZE), WIDTH - HALF_PATCH_SIZE)
                y = min (max( int(jonts_image[i, j, 1]), HALF_PATCH_SIZE), HEIGHT - HALF_PATCH_SIZE)

                ray_ori_patches.append( ray_orig[y-HALF_PATCH_SIZE:y+HALF_PATCH_SIZE, x-HALF_PATCH_SIZE:x+HALF_PATCH_SIZE] )
                ray_dir_patches.append( ray_dir[y-HALF_PATCH_SIZE:y+HALF_PATCH_SIZE, x-HALF_PATCH_SIZE:x+HALF_PATCH_SIZE] )
                xyz_patches.append( coords[i, y-HALF_PATCH_SIZE:y+HALF_PATCH_SIZE, x-HALF_PATCH_SIZE:x+HALF_PATCH_SIZE] )
                rgb_patches.append( rgb[i, y-HALF_PATCH_SIZE:y+HALF_PATCH_SIZE, x-HALF_PATCH_SIZE:x+HALF_PATCH_SIZE] )
                nrm_patches.append( normal[i, y-HALF_PATCH_SIZE:y+HALF_PATCH_SIZE, x-HALF_PATCH_SIZE:x+HALF_PATCH_SIZE] )
                mask_patches.append( mask[i, y-HALF_PATCH_SIZE:y+HALF_PATCH_SIZE, x-HALF_PATCH_SIZE:x+HALF_PATCH_SIZE] )

            
        dataset_pts[s] = torch.cat([pts1, pts2], dim=0).detach().cpu().numpy()
        dataset_rgb[s] = torch.cat([rgb1, rgb2], dim=0).detach().cpu().numpy()
        dataset_nrm[s] = torch.cat([nrm1, nrm2], dim=0).detach().cpu().numpy()
        dataset_d[s] = torch.cat([d1, d2], dim=0).detach().cpu().numpy()
        dataset_smpl_v[s] = smpl_V.detach().cpu().numpy()
        dataset_xyz_image[s] = torch.stack(xyz_patches).detach().cpu().numpy()
        dataset_rgb_image[s] = torch.stack(rgb_patches).detach().cpu().numpy()
        dataset_nrm_image[s] = torch.stack(nrm_patches).detach().cpu().numpy()
        dataset_mask_image[s] = torch.stack(mask_patches).detach().cpu().numpy()
        dataset_ray_ori_image[s] = torch.stack(ray_ori_patches).detach().cpu().numpy()
        dataset_ray_dir_image[s] = torch.stack(ray_dir_patches).detach().cpu().numpy()


    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset to H5 file')

    parser.add_argument("-i", "--input_path", default='./CustomHumans', type=str, help="Path of the input mesh folder")
    parser.add_argument("-o", "--output_path", default='./CustomHumans.h5', type=str, help="Path of the output h5 file")

    main(parser.parse_args())
