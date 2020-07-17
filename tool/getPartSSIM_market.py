import torch
import os.path
import numpy as np
from skimage.io import imread, imsave
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from losses.pytorch_msssim import SSIM, MS_SSIM, Part_SSIM, P_plus_P_SSIM, MS_Part_SSIM

from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time


def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []

    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / 5) #h, w ,c
        input_images.append(img[:, :w])
        target_images.append(img[:, 2*w:3*w])
        generated_images.append(img[:, 4*w:5*w])

        # assert img_name.endswith('_vis.png'), 'unexpected img name: should end with _vis.png'
        assert img_name.endswith('_vis.png') or img_name.endswith('_vis.jpg'), 'unexpected img name: should end with _vis.png'

        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)', img_name)
        # m = re.match(r'([A-Za-z0-9_]*.jpg)_([A-Za-z0-9_]*.jpg)_vis.png', img_name)
        # fr = m.groups()[0]
        # to = m.groups()[1]
        names.append([fr, to])

    return input_images, target_images, generated_images, names


def bp_ssim(X, Y, bp_mask, data_range=255, K=(0.01,0.03)):
    H, W, img_channel = X.shape[0], X.shape[1], X.shape[2]
    mask_channel = bp_mask.shape[2]
    new_shape = (H, W, img_channel*mask_channel)

    X_repeat = np.expand_dims(X, -1).repeat(mask_channel, -1).reshape(new_shape)
    Y_repeat = np.expand_dims(Y, -1).repeat(mask_channel, -1).reshape(new_shape)
    bp_mask_repeat = np.expand_dims(bp_mask, -2).repeat(img_channel, -2).reshape(new_shape)

    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # calculate mean of X and Y (method 1)
    bp_mask_sum = np.expand_dims(np.expand_dims(bp_mask_repeat.sum((0,1)),0).repeat(W,0),0).repeat(H,0)
    bp_mask_avg = bp_mask_repeat/(bp_mask_sum + 1e-6) # sum(bp_mask[0,0,:,:]) = 1
    mu1 = np.multiply(X_repeat, bp_mask_avg).sum((0,1))
    mu2 = np.multiply(Y_repeat, bp_mask_avg).sum((0,1))

    # # calculate mean of X and Y (method 2)
    # no_zero_count = torch.sum(bp_mask>mask_t)
    # mu1 = torch.mul(X, bp_mask).sum()/no_zero_count
    # mu2 = torch.mul(Y, bp_mask).sum()/no_zero_count

    mu1_sq = np.power(mu1,2)
    mu2_sq = np.power(mu2,2)
    mu1_mu2 = mu1 * mu2

    mu11 = np.multiply(X_repeat*X_repeat, bp_mask_avg).sum((0,1))
    mu22 = np.multiply(Y_repeat*Y_repeat, bp_mask_avg).sum((0,1))
    mu12 = np.multiply(X_repeat*Y_repeat, bp_mask_avg).sum((0,1))
    sigma1_sq = compensation * (mu11 - mu1_sq)
    sigma2_sq = compensation * (mu22 - mu2_sq)
    sigma12   = compensation * (mu12 - mu1_mu2)

    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # set alpha=beta=gamma=1
    ssim = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs

    ssim_avg = ssim.mean()

    return ssim_avg

def test(generated_images_dir, annotations_file_test, bodypart_mask_dir):
    input_images, target_images, generated_images, names = load_generated_images(generated_images_dir)

    ssim_score_list = []
    for reference_image, generated_image, P2_name in zip(generated_images, target_images, names):
        BP2_mask_path = os.path.join(bodypart_mask_dir, P2_name[1] + '.npy')
        BP2_mask_img = np.load(BP2_mask_path)
        BP2_mask = torch.from_numpy(BP2_mask_img).float()
        # BP2_mask = BP2_mask.transpose(2, 0) #c,w,h
        # BP2_mask = BP2_mask.transpose(2, 1) #c,h,w

        ssim = bp_ssim(reference_image, generated_image, BP2_mask[:,:,1:])
        ssim_score_list.append(ssim)
        print(P2_name, ssim)
    return np.mean(ssim_score_list)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Computing ssd_score")

    parser.add_argument("--images_dir", default='/data0/haoyue/codes/results/market_PATN_ssim/test_latest/images', help='Folder with images')
    parser.add_argument("--annotations_file", default='/data0/haoyue/codes/datasets/market_data/market-annotation-test.csv',  help='')
    parser.add_argument("--bodypart_mask_dir", default='/data0/haoyue/codes/datasets/market_data/testM/', help='')
    parser.add_argument('--dataroot', required=True,
                             help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--dataset_mode', type=str, default='unaligned',
                             help='chooses how datasets are loaded. [unaligned | aligned | single]')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                             help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    # parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                             help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    # parser.add_argument('--pairLst', type=str, default='./keypoint_data/market-pairs-train.csv',
    #                          help='market pairs')

    args = parser.parse_args()
    # generated_images_dir = args.generated_images_dir
    # # generated_images_dir = '/home/haoyue/codes/results/market_PATN/test_latest_SSIM/images'
    # annotations_file_test = args.annotations_file_test
    # bodypart_mask_dir = args.bodypart_mask_dir

    # opt = TestOptions().parse()
    # opt.nThreads = 1  # test code only supports nThreads = 1
    # opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    # opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()

    criterionSSIM = P_plus_P_SSIM(data_range=1.0, size_average=True, win_size=7,
                                       win_sigma=0.8)

    # print(args.how_many)
    num_test_image = len(dataset)
    print(num_test_image)

    # args.how_many = 999999
    # test
    ppssim_list = []
    for i, data in enumerate(dataset):
        print(' process %d/%d img ..' % (i, num_test_image))
        ppssim = criterionSSIM(data['target'], data['generated'], data['mask'])
        ppssim_list.append(ppssim)
        print(ppssim)

    ppssim_list = torch.from_numpy(np.array(ppssim_list)).float()
    print(args.images_dir)
    print('ppssim', torch.sum(ppssim_list)/num_test_image)



    # bpssim = test(generated_images_dir, annotations_file_test, bodypart_mask_dir)
    # print(generated_images_dir, bpssim)