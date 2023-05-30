import os
import json
import shutil

DATASET_PATH = 'CustomHumans'
OUTPUT_PATH = 'CustomHumans/training_dataset'
os.makedirs(OUTPUT_PATH, exist_ok=True)

mesh_path = { x.split('_')[0]:x for x in sorted(os.listdir(os.path.join(DATASET_PATH, 'mesh'))) }
subject_idx = json.load(open('data/Custom_train.json'))

for idx in subject_idx:
    folder_name = mesh_path[idx]
    shutil.copytree(os.path.join(DATASET_PATH, 'mesh', folder_name), os.path.join(OUTPUT_PATH, idx), dirs_exist_ok = True)
    shutil.copytree(os.path.join(DATASET_PATH, 'smplx', folder_name), os.path.join(OUTPUT_PATH, idx), dirs_exist_ok = True)


