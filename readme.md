# Loss Functions for Pose-guided Person Image Generation

Code for the paper **Loss Function for Person Image Generation** in **BMVC2020**.

<img src="/home/haoyue/codes/Pose-Transfer-master/imgs/compare1.png" style="zoom:70%;" />

## Get Start

### Requirements

* pytorch(0.3.1)
* torchvision(0.2.0)
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate

### Installation

Clone this repo:

```bash
git clone https://github.com/shyern/Pose-Transfer-pSSIM.git
cd Pose-Transfer-pSSIM
```

### Dataset

We build the market-1501 dataset and DeepFashion dataset following PATN. The details for building these two datasets are shown here.

#### Market-1501

- Download the Market-1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html). Rename **bounding_box_train** and **bounding_box_test** as **train** and **test**, and put them under the `./datasets/market_data` directory
- Download train/test key points annotations from [Google Drive](https://drive.google.com/drive/folders/1fZCN_FZUAuCqC533qd9L3OWyf4w_vnI8?usp=sharing) including **market-pairs-train.csv**, **market-pairs-test.csv**, **market-annotation-train.csv**, **market-annotation-train.csv**. Put these files under the  `./datasets/market_data` directory.
- Generate the pose heatmaps. Launch

```bash
python tool/generate_pose_map_market.py
```

- Generate the limbs mask. Launch

```bash
python tool/generate_limbs_mask_market.py
```

#### DeepFashion

- Download the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 
- Unzip `img.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then put the obtained folder **img** under the `./datasets/fashion_data` directory. 
- Download train/test key points annotations and the dataset list from [Google Drive](https://drive.google.com/drive/folders/17RrcoCJWsBWnfHGmj6DJ-DLxHhKeWY7c?usp=sharing) including **fashion-pairs-train.csv**, **fashion-pairs-test.csv**, **fashion-annotation-train.csv**, **fashion-annotation-train.csv,** **train.lst**, **test.lst**. Put these files under the  `./datasets/fashion_data` directory.
- Run the following code to split the train/test dataset.

```bash
python tool/generate_fashion_datasets.py
```

- Generate the pose heatmaps. Launch

```bash
python tool/generate_pose_map_fashion.py
```

- Generate the limbs mask. Launch

```bash
python tool/generate_limbs_mask_fashion.py
```

### Training

Training with **Perceptual loss**, **Adversarial loss**, and **part-based SSIM loss**.

- Market-1501

```bash
python train.py --dataroot ./datasets/market_data/ --name market_PATN_ganppssimperl1 --model PATN --lambda_GAN 5 --lambda_A 0 --lambda_B 10 --lambda_SSIM 10 --dataset_mode keypoint --no_lsgan --norm batch --batchSize 32 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --niter 500 --niter_decay 200 --checkpoints_dir ./checkpoints_market --pairLst ./datasets/market_data/market-pairs-train.csv --L1_type FPart_BSSIM_plus_perL1_L1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1 --display_id 0 --win_size 7 --win_sigma 0.8
```

- DeepFashion

```bash
python train.py --dataroot ./datasets/fashion_data/ --name fashion_PATN_ganppssimperl1 --model PATN --lambda_GAN 5 --lambda_A 0 --lambda_B 1 --lambda_SSIM 1 --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 7 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --niter 500 --niter_decay 200 --checkpoints_dir ./checkpoints_fashion --pairLst ./datasets/fashion_data/fashion-pairs-train.csv --L1_type FPart_BSSIM_plus_perL1_L1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 0 --win_size 7 --win_sigma 0.8
```

### Testing

**Download the trained weights from [Fashion](https://drive.google.com/drive/folders/1BPkEx8ERVJlEZhgn_ps1nYiEmiC43jS6?usp=sharing),  [Market](https://drive.google.com/drive/folders/10KEG6-DNFPpw-OHdoTeCYTcOFm1oFwWY?usp=sharing)**. Put the obtained checkpoints under `./checkpoints_fashion` and `./checkpoints_market` respectively.

- Market-1501

```bash
python test.py --dataroot ./datasets/market_data/ --name market-1501 --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints_market --pairLst ./datasets/market_data/market-pairs-test.csv --with_D_PP 0 --with_D_PB 0 --which_epoch 680 --results_dir ./results_market --display_id 0
```

- DeepFashion

```bash
python test.py --dataroot ./datasets/fashion_data/ --name DeepFashion --model PATN --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints_fashion --pairLst ./datasets/fashion_data/fashion-pairs-test.csv --which_epoch 660 --results_dir ./results_fashion --display_id 0
```

### Evaluation

We adopt SSIM, IS, mask-SSIM, mask-IS, DS, and pSSIM for evaluation of Market-1501. SSIM, IS, DS, pSSIM for DeepFashion.

  SSIM, IS, mask-SSIM, mask-IS, DS

  Please follow [PATN]() to acquire SSIM, IS, mask-SSIM, amask-IS, and DS.

- pSSIM

for Market-1501

```bash
python getPartSSIM.py --dataroot ./datasets/market_data/ --dataset_mode keypoint2 --resize_or_crop no --annotations_file ./datasets/market_data/market-annotation-test.csv --bodypart_mask_dir .datasets/market_data/testM --images_dir ./results_market/images
```

for DeepFashion

```bash
python getPartSSIM.py --dataroot ./datasets/fashion_data/ --dataset_mode keypoint2 --resize_or_crop no --annotations_file ./datasets/fashion_data/fashion-annotation-test.csv --bodypart_mask_dir .datasets/fashion_data/testM --images_dir ./results_fashion/images
```

## Citation

```tex


```

## Acknowledgement

We build our project base on  [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer).

