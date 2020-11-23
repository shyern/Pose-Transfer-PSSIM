import torch
import numpy as np
import sys

sys.path.append("/home/haoyue/codes/Pose-Transfer-master/data")
sys.path.append("/home/haoyue/codes/Pose-Transfer-master/losses")

from data_loader import CreateDataLoader
from pytorch_msssim import APart_SSIM, FPart_BSSIM


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Computing ssd_score")

    parser.add_argument("--images_dir", default='./results/market_PATN_ssim/test_latest/images', help='Folder with images')
    parser.add_argument("--annotations_file", default='./datasets/market_data/market-annotation-test.csv',  help='')
    parser.add_argument("--bodypart_mask_dir", default='./datasets/market_data/testM/', help='')
    parser.add_argument('--dataroot', required=True,
                             help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--dataset_mode', type=str, default='unaligned',
                             help='chooses how datasets are loaded. [unaligned | aligned | single]')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                             help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    # parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
    parser.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                             help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    args = parser.parse_args()

    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()

    criterionSSIM = FPart_BSSIM(data_range=1.0, size_average=True, win_size=7,
                                       win_sigma=0.8)

    num_test_image = len(dataset)
    print(num_test_image)

    # test
    ppssim_list = []
    for i, data in enumerate(dataset):
        print(' process %d/%d img ..' % (i, num_test_image/args.batchSize))
        ppssim = criterionSSIM(data['target'], data['generated'], data['mask'])
        ppssim_list.append(ppssim)
        print(ppssim)

    ppssim_list = torch.from_numpy(np.array(ppssim_list)).float()
    len_data = ppssim_list.shape[0]
    print(args.images_dir)
    print('ppssim', torch.sum(ppssim_list)/len_data)
