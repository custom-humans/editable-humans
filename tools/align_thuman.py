import torch
import numpy as np
import os
import trimesh
from PIL import Image
import pickle
import cv2
from tqdm import tqdm
from smplx import SMPLX
import json
device = torch.device('cuda')

SMPLX_PATH = 'smplx'
OUT_PATH = 'new_thuman'

body_model = SMPLX(model_path=SMPLX_PATH, num_pca_comps=12,gender='male')

for id in tqdm(range(526)):
    name_id = "%04d" % id
    input_file = os.path.join('THuman2.0', name_id, name_id + '.obj')
    tex_file = os.path.join('THuman2.0', name_id, 'material0.jpeg')
    smpl_file = os.path.join('THuman2.0_smplx', name_id, 'smplx_param.pkl')

    smpl_data = pickle.load(open(smpl_file,'rb'))
    out_file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_aligned_path = os.path.join(OUT_PATH, out_file_name)
    os.makedirs(output_aligned_path, exist_ok=True)

    
    textured_mesh = trimesh.load(input_file)


    output = body_model(body_pose = torch.tensor(smpl_data['body_pose']),
                                betas = torch.tensor(smpl_data['betas']),
                                left_hand_pose = torch.tensor(smpl_data['left_hand_pose']),
                                right_hand_pose = torch.tensor(smpl_data['right_hand_pose']),
                               )
    J_0 = output.joints.detach().cpu().numpy()[0,0,:]

    d = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy()[0] -J_0 ,
                                    faces=body_model.faces)


    R = np.asarray(smpl_data['global_orient'][0])
    rot_mat = np.zeros(shape=(3,3))
    rot_mat, _ = cv2.Rodrigues(R)
    scale = smpl_data['scale']

    T = -np.asarray(smpl_data['translation'])
    S = np.eye(4)
    S[:3, 3] = T
    textured_mesh.apply_transform(S)
    
    S = np.eye(4)
    S[:3, :3] *= 1./scale
    textured_mesh.apply_transform(S)

    T = -J_0
    S = np.eye(4)
    S[:3, 3] = T
    textured_mesh.apply_transform(S)

    S = np.eye(4)
    S[:3, :3] = np.linalg.inv(rot_mat)
    textured_mesh.apply_transform(S)



    visual = trimesh.visual.texture.TextureVisuals(uv=textured_mesh.visual.uv, image=Image.open(tex_file))
    
    t = trimesh.Trimesh(vertices=textured_mesh.vertices,
                                 faces=textured_mesh.faces,
                                 vertex_normals=textured_mesh.vertex_normals,
                                 visual=visual)

    #t = t.simplify_quadratic_decimation(50000)
    #t.visual.material.name = out_file_name


    d.export(os.path.join(output_aligned_path, out_file_name + '_smplx.obj')  )
    t.export(os.path.join(output_aligned_path, out_file_name + '.obj')  )
    with open(os.path.join(output_aligned_path, out_file_name + '.mtl'), 'w') as f:
            f.write('newmtl {}\n'.format(out_file_name))
            f.write('map_Kd {}.jpeg\n'.format(out_file_name))
    
    result = {}
    result ['transl'] = [0.,0.,0.]
    for key, val in smpl_data.items():
        if key not in ['scale', 'translation']:
            result[key] = val[0].tolist()

    json_file = os.path.join(output_aligned_path, out_file_name + '_smplx.json')
    json.dump(result, open(json_file, 'w'), indent=4)
