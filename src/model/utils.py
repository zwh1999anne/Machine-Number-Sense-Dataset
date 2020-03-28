# -*- coding: utf-8 -*-


import glob
import os

import numpy as np
import torch
from scipy import misc
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, dataset_path, img_size, transform=None, names=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(self.dataset_path, "*.npz"))]
        if names != None:
            def judge(file_name):
                for name in names:
                    if name not in file_name:
                        return False
                return True
            self.file_names = filter(judge, self.file_names)    
        self.img_size = img_size
        for _  in range(10):
            np.random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        
        data = np.load(data_path)
        image = data["image"]
        target = data["target"] - 1

        if self.img_size != 160:
            resize_image = []
            for idx in range(3):
                resize_image.append(misc.imresize(image[idx, :, :], (self.img_size, self.img_size)))
            image = np.stack(resize_image)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.long)

        return image, target
