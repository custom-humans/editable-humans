import os
from smplx import SMPLX
import torch
import json
import trimesh
import argparse

SMPL_PATH = 'body_model/smplx/'
'''
We use the following minimal code snippet to generate the SMPL-X model across all our scans
'''
def main(args):

    smpl_data = json.load(open(os.path.join(args.input_file)))


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    
    d = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy()[0], faces=body_model.faces)
    d.export('smplx.obj')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Minimal code snippet to generate SMPL-X mesh from json file')

    parser.add_argument("-i", "--input-file", default='./mesh-f00021.json', type=str, help="Input json file")

    main(parser.parse_args())
