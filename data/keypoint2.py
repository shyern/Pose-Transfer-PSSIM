import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
from skimage.io import imread, imsave


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_M = os.path.join(opt.dataroot, opt.phase + 'M')  # limbs mask
        # self.dir_M = os.path.join(opt.dataroot, opt.phase + 'MSM')  # limbs mask

        # dir_KC = os.path.join(opt.dataroot, opt.phase+'KC.npy')   # keypoints coor
        # self.keypoint_coor = np.load(dir_KC, allow_pickle = True).item()

        self.init_categories(opt.images_dir)
        self.transform = get_transform(opt)

    def init_categories(self, image_dir):
        self.images_dir = []
        for img_name in os.listdir(image_dir):
            img_dir = os.path.join(image_dir, img_name)
            self.images_dir.append(img_dir)
        self.size = len(self.images_dir)

        print('Loading data images finished ...')

    def __getitem__(self, index):
        img_name = self.images_dir[index]
        img = imread(img_name)
        w = int(img.shape[1] / 5)  # h, w ,c
        input_image = img[:, :w]
        target_image = img[:, 2 * w:3 * w]
        generated_image = target_image
        # generated_image = img[:, 4 * w:5 * w]

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'

        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        mask_path = os.path.join(self.dir_M, to + '.npy')
        mask_img = np.load(mask_path)
        mask = torch.from_numpy(mask_img).float()
        mask = mask.transpose(-1, -3)  # s,c,w,h
        mask = mask.transpose(-1, -2)  # s,c,h,w

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)
        generated_image = self.transform(generated_image)

        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)_vis.png', img_name)
        # fr = m.groups()[0]
        # to = m.groups()[1]
        # names = [fr, to]

        return {'input': input_image, 'target': target_image, 'generated': generated_image, 'mask': mask}


        # return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'BP2_mask': BP2_mask,
        #         'P1_path': P1_name, 'P2_path': P2_name}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'
