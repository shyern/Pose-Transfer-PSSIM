# Loss Functions for Pose-guided Person Image Generation

Code for the conference paper **Loss Function for Person Image Generation** in **BMVC2020** and the submitted journal paper **A Comprehensive Study of Loss Functions for Pose Guided Person Image Generation**.

## Get Start

### Requirements

* pytorch
* torchvision
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


### Training

Training with **Perceptual loss**, **Adversarial loss**, and **part-based SSIM loss**.

- Market-1501

```bash
bash ./script/train_market.sh
```

- DeepFashion

```bash
bash ./script/train_fashion.sh
```

### Testing

**Download the trained weights from [Fashion](https://drive.google.com/drive/folders/1BPkEx8ERVJlEZhgn_ps1nYiEmiC43jS6?usp=sharing),  [Market](https://drive.google.com/drive/folders/10KEG6-DNFPpw-OHdoTeCYTcOFm1oFwWY?usp=sharing)**. Put the obtained checkpoints under `./checkpoints_fashion` and `./checkpoints_market` respectively.

- Market-1501

```bash
bash ./script/test_market.sh
```

- DeepFashion

```bash
bash ./script/test_fashion.sh
```

### Evaluation

We adopt SSIM, IS, mask-SSIM, mask-IS, DS, and pSSIM for evaluation of Market-1501. SSIM, IS, DS, pSSIM for DeepFashion.

  SSIM, IS, mask-SSIM, mask-IS, DS

  Please follow [PATN]() to acquire SSIM, IS, mask-SSIM, amask-IS, and DS.


## Citation

```tex


```

## Acknowledgement

We build our project base on  [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer).

