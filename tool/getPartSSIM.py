import os
import os.path
import numpy as np
from skimage.io import imread, imsave
import pandas as pd
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append('./losses/')
from pose_utils import load_pose_cords_from_strings, make_gaussain_limb_masks
from pytorch_msssim import FPart_BSSIM


class KeyDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        self.init_categories(opt.pairLst, opt.annoLst, opt.results_dir)
        self.transform = self.get_transform()

    def init_categories(self, pairLst, annoLst, results_dirs):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.images_dir = []
        print('Loading images from' + results_dirs)
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            save_name = pair[0] + '___' + pair[1] + '_vis.jpg'
            img_dir = os.path.join(results_dirs, save_name)
            self.images_dir.append(img_dir)
        print('Loading images finished ...')

        print('Loading data annos ...')
        annotations_file = pd.read_csv(annoLst, sep=':')
        self.annos = annotations_file.set_index('name')
        print('Loading data annos finished ...')

    def get_transform(self):
        transform_list = []
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def get_gaussian_mask(self, P2_name, img_size):
        to = self.annos.loc[P2_name]

        kp_array2 = load_pose_cords_from_strings(to['keypoints_y'],
                                                 to['keypoints_x'])

        BP2_mask = make_gaussain_limb_masks(kp_array2, img_size)  # BP2 mask
        return BP2_mask

    def __getitem__(self, index):
        img_name = self.images_dir[index]
        img = imread(img_name)

        # for PATN
        w = int(img.shape[1] / 5)  # h, w ,c
        input_image = img[:, :w]
        target_image = img[:, 2 * w:3 * w]
        generated_image = img[:, 4 * w:5 * w]

        assert img_name.endswith('_vis.png') or img_name.endswith(
            '_vis.jpg'), 'unexpected img name: should end with _vis.png'

        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        # fr = img_name[0]
        to = img_name[1]

        # print(generated_image.shape)
        img_size = [generated_image.shape[0], generated_image.shape[1]]
        mask = self.get_gaussian_mask(to, img_size)

        # # # for Def-GAN
        # w = int(img.shape[1] / 3)  # h, w ,c
        # input_image = img[:, :w]
        # target_image = img[:, w:2 * w]
        # # generated_image = target_image
        # generated_image = img[:, 2 * w:3 * w]
        #
        # assert img_name.endswith('jpg.png') or img_name.endswith('jpg.jpg'), 'unexpected img name: should end with _vis.png'
        # img_name = img_name[:-4]
        # img_name = img_name.split('g_')
        # assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        # fr = img_name[0] + 'g'
        # to = img_name[1]

        # # for GFLA market
        # from PIL import Image
        # img = Image.open(img_name).convert("RGB")
        # generated_image = np.array(img).astype(np.uint8)
        # # generated_image = img
        # img_name_s = img_name.split('_2_')
        # fr = img_name_s[0].split('/')[-1] + '.jpg'
        # to = img_name_s[1].split('_vis')[0] + '.jpg'
        # # target_image = imread(os.path.join(self.dir_P, to))
        # input_image = imread(os.path.join(self.dir_P, fr))
        #
        # target_image = Image.open(os.path.join(self.dir_P, to)).convert("RGB").resize((256, 256))
        # target_image = np.array(target_image).astype(np.uint8)

        input_image = self.transform(input_image).cuda()
        target_image = self.transform(target_image).cuda()
        generated_image = self.transform(generated_image).cuda()

        mask = torch.from_numpy(mask).float().cuda()

        input_image = input_image.unsqueeze(0)
        target_image = target_image.unsqueeze(0)
        generated_image = generated_image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {'input': input_image, 'target': target_image, 'generated': generated_image, 'mask': mask}

    def __len__(self):
        return self.size

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Computing ssd_score")

    parser.add_argument("--results_dir", default='./results/results_market/results_pssim/market_pssim_xing/test_latest/images', help='Folder with images')
    parser.add_argument('--annoLst', type=str, default='./datasets/market_data/market-annotation-test.csv', help='market annos')
    parser.add_argument('--pairLst', type=str, default='./datasets/market_data/market-pairs-test.csv', help='market pairs')

    args = parser.parse_args()

    dataset = KeyDataset(args)

    criterionSSIM = FPart_BSSIM(data_range=1.0, size_average=True, win_size=7, win_sigma=0.8)

    num_test_image = len(dataset)
    print(num_test_image)

    # test
    ppssim_list = []
    for i, data in tqdm(enumerate(dataset)):
        # print(' process %d/%d img ..' % (i, num_test_image))
        ppssim = criterionSSIM(data['target'], data['generated'], data['mask'])
        ppssim = ppssim.cpu().numpy()
        ppssim_list.append(ppssim)
        # print(ppssim)

    print(args.results_dir)
    print('ppssim', np.mean(np.array(ppssim_list)))
