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


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.dataset, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.dataset, opt.phase + 'K') #keypoints
        self.dir_M = os.path.join(opt.dataroot, opt.dataset, opt.phase + 'M')  # limbs mask
        # self.dir_M = os.path.join(opt.dataroot, opt.phase + 'MSM')  # limbs mask

        # dir_KC = os.path.join(opt.dataroot, opt.phase+'KC.npy')   # keypoints coor
        # self.keypoint_coor = np.load(dir_KC, allow_pickle = True).item()

        pairLst = os.path.join(opt.dataroot, opt.dataset, opt.pairLst)
        self.init_categories(pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2
        BP2_mask_path = os.path.join(self.dir_M, P2_name + '.npy')

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path)
        BP2_mask_img = np.load(BP2_mask_path)
        # BP2_keypoint_coor = self.keypoint_coor[P2_name]  # (h,w) (128, 64)
        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip
                BP2_mask_img = np.array(BP2_mask_img[:, ::-1, :]) # flip
                # BP2_keypoint_coor = np.array(P1_img.shape[1]-BP2_keypoint_coor[0,:])  ???

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w

            BP2_mask = torch.from_numpy(BP2_mask_img).float()
            BP2_mask = BP2_mask.transpose(-1, -3) #c,w,h
            BP2_mask = BP2_mask.transpose(-1, -2) #c,h,w

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            BP2_mask = torch.from_numpy(BP2_mask_img).float()
            BP2_mask = BP2_mask.transpose(-1, -3) #s,c,w,h
            BP2_mask = BP2_mask.transpose(-1, -2) #s,c,h,w

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'BP2_mask': BP2_mask,
                'P1_path': P1_name, 'P2_path': P2_name}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'
