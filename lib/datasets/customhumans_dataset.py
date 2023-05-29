
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging as log
import time

class CustomHumanDataset(Dataset):
    """Base class for single mesh datasets with points sampled only at a given octree sampling region.
    """

    def __init__(self, 
        num_samples        : int = 20480,
        repeat_times      : int = 8,
    ):
        """Construct dataset. This dataset also needs to be initialized.
        """
        self.repeat_times = repeat_times    # epeate how many times each epoch
        self.num_samples = num_samples      # number of points per subject

        self.initialization_mode = None
        self.label_map = {
            '0': 1, '1': 2, '2': 2, '3': 1, '4': 2,
            '5': 2, '6': 1, '7': 2, '8': 2, '9': 1,
            '10': 2, '11': 2, '12': 1, '13': 1, '14': 1,
            '15': 0, '16': 1, '17': 1, '18': 1, '19': 1,
            '20': 1, '21': 1, '22': 0, '23': 0, '24': 0,
            }

    def init_from_h5(self, dataset_path):
        """Initializes the dataset from a h5 file.
           copy smpl_v from h5 file.
        """

        self.h5_path = dataset_path
        with h5py.File(dataset_path, "r") as f:
            try:
                self.num_subjects = f['num_subjects'][()]
                self.num_pts = f['d'].shape[1]
                self.smpl_V = torch.tensor(np.array(f['smpl_v']))
            except:
                raise ValueError("[Error] Can't load from h5 dataset")
        self.resample()
        self.initialization_mode = "h5"

    def resample(self):
        """Resamples a new working set of indices.
        """
        
        start = time.time()
        log.info(f"Resampling...")

        self.id = np.random.randint(0, self.num_subjects, self.num_subjects * self.repeat_times)

        log.info(f"Time: {time.time() - start}")

    def _get_h5_data(self, subject_id, pts_id, img_id):
        with h5py.File(self.h5_path, "r") as f:
            try:
                pts = np.array(f['pts'][subject_id,pts_id])
                d = np.array(f['d'][subject_id,pts_id])
                nrm = np.array(f['nrm'][subject_id,pts_id])
                rgb = np.array(f['rgb'][subject_id,pts_id])
                image_label = self.label_map[str(img_id[0] % 25)]

                xyz_image = np.array(f['xyz_image'][subject_id,img_id])
                rgb_image = np.array(f['rgb_image'][subject_id,img_id])
                nrm_image = np.array(f['nrm_image'][subject_id,img_id])
                mask_image = np.array(f['mask_image'][subject_id,img_id])
                ray_ori_image = np.array(f['ray_ori_image'][subject_id,img_id])
                ray_dir_image = np.array(f['ray_dir_image'][subject_id,img_id])

            except:
                raise ValueError("[Error] Can't read key (%s, %s, %s) from h5 dataset" % (subject_id, pts_id, img_id))

        return {
                'pts' : pts, 'sdf' : d, 'nrm' : nrm, 'rgb' : rgb, 'idx' : subject_id, 'label' : image_label,
                'xyz_image' : xyz_image, 'rgb_image' : rgb_image,  'nrm_image' : nrm_image,
                'mask_image' : mask_image, 'ray_ori_image' : ray_ori_image, 'ray_dir_image' : ray_dir_image
        }

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        subject_id = self.id[idx]
        # points id need to be in accending order
        pts_id = np.random.randint(self.num_pts - self.num_samples, size=1)
        img_id = np.random.randint(100, size=1)

        return self._get_h5_data(subject_id, np.arange(pts_id, pts_id + self.num_samples), img_id)
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")

        return self.num_subjects * self.repeat_times
