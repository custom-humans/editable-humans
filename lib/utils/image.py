import os
import pickle
import cv2

import torchvision.transforms as transforms

def update_edited_images(image_path, pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    img_list = [ os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith('.png') ]
    transform = transforms.Compose([
                transforms.ToTensor()
                ])
    for i, img in enumerate(img_list):
        rgb_img = cv2.imread(img)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb = transform(rgb_img).permute(1,2,0).view(-1, 3)
        data['rgb'][i] = rgb

    return data