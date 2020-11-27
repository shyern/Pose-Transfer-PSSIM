#!/usr/bin/env bash

pro_dir="/home/haoyue/remote/Person-Image-Gen-pssim/"

cd ${pro_dir}
crtdir=$(pwd)
echo ${crtdir}

######################################################################################
################################get metrics###############################################
gpu_id=0
name="market_att_sty_gan"
which_epoch=latest
results_dir=${pro_dir}"results/results_market/results4"

##HOST=58.206.101.18
#HOST=202.200.142.249
#src="haoyue@"${HOST}":"${results_dir}"/"${name}
#tar=${results_dir}
#echo ${src} ${tar}
#rsync -vaz ${src} ${tar}

anno_file_test=${pro_dir}"datasets/market_data/market-annotation-test.csv"
gt_path=${pro_dir}"datasets/market_data/test"
fid_real_path=${pro_dir}"datasets/market_data/train"
gen_imgs_dir=${results_dir}"/"${name}"/test_"${which_epoch}"/images"


echo "SSIM,IS,Mask-SSIM,Mask-IS..."
CUDA_VISIBLE_DEVICES=${gpu_id} /home/haoyue/anaconda3/envs/p37tf114/bin/python ${pro_dir}/tool/metrics_ssim_market.py --generated_images_dir ${gen_imgs_dir} --annotations_file_test ${anno_file_test}

echo "FID,LPIPS..."
CUDA_VISIBLE_DEVICES=${gpu_id} /home/haoyue/anaconda3/envs/p37pytorch/bin/python ${pro_dir}/tool/metrics_fid_market.py  --gt_path=${gt_path} --distorated_path=${gen_imgs_dir} --fid_real_path=${fid_real_path} --name=${name} --calculate_mask
