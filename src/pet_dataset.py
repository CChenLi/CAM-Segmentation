import torch
import numpy as np
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import cv2
from torch.utils.data import Dataset,DataLoader
import os
import io



class PetDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, resizer=None):
        self.label_data = pd.read_csv(csv_file, sep='\s+')
        self.root_dir = root_dir
        self.cam_dir = "/content/cam_score_compressed"
        self.mask_dir = "/content/trimaps_2compressed"
        self.transform = transform
        self.resizer = resizer

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.label_data.iloc[idx, 0] + ".jpg")
        raw_image = io.imread(img_name)
        image = io.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        cam_path = os.path.join(self.cam_dir, self.label_data.iloc[idx, 0] + ".png")
        cam = io.imread(cam_path)

        mask_path = os.path.join(self.mask_dir, self.label_data.iloc[idx, 0] + ".png")
        mask = io.imread(mask_path)

        '''
        if self.transform:
            image = self.transform(image)
            # cam = self.resizer(cam)
            mask = self.resizer(mask)[0]
        '''

        if self.transform is not None:
            masks = [mask, cam]
            augmentations = self.transform(image=image, masks=masks)
            image = augmentations['image'] / 255.0
            mask = augmentations['masks'][0]
            cam = np.expand_dims(augmentations['masks'][1] / 255.0, 0)

        classid = self.label_data.iloc[idx, 1] - 1 # 0:36 class ids
        species = self.label_data.iloc[idx, 2] - 1 # 0:Cat 1:Dog
        sample = {'image': image,
                  'mask': mask,
                  'cam': cam,
                  'classid': classid,
                  'species': species,
                  'img_name':img_name
                  }

        return sample 