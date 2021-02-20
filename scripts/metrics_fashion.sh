#!/usr/bin/env bash

pro_dir="./Pose-Transfer-PSSIM"
cd ${pro_dir}

gpu_id=0
name="fashion_pssim"
which_epoch=latest
results_dir=${pro_dir}"results/results_fashion"

anno_file_test=${pro_dir}"datasets/fashion_data/fasion-resize-annotation-test.csv"
pair_file_test=${pro_dir}"datasets/fashion_data/fasion-resize-pairs-test.csv"
gt_path=${pro_dir}"datasets/fashion_data/test"
fid_real_path=${pro_dir}"datasets/fashion_data/train"
gen_imgs_dir=${results_dir}"/"${name}"/test_"${which_epoch}"/images"
image_size=(256,176)


echo "SSIM,IS,Mask-SSIM,Mask-IS..."
CUDA_VISIBLE_DEVICES=${gpu_id} ~/anaconda3/envs/p37tf114/bin/python ${pro_dir}/tool/metrics_ssim_fashion.py --generated_images_dir ${gen_imgs_dir}\
| tee -a ${pro_dir}/records/fashion/r_ssim.txt

echo "DS..."
~/anaconda3/envs/p37tf114/bin/python ${pro_dir}/ssd_score/compute_ssd_score_fashion.py --input_dir=${gen_imgs_dir} --image_size=${image_size} --gpu_id=${gpu_id}\
| tee -a ${pro_dir}/records/fashion/r_ds.txt

echo  "pSSIM ..."
CUDA_VISIBLE_DEVICES=${gpu_id} ~/anaconda3/envs/p37pytorch/bin/python ${pro_dir}/tool/getPartSSIM.py --results_dir=${gen_imgs_dir} --annoLst=${anno_file_test} --pairLst=${pair_file_test}\
| tee -a ${pro_dir}/records/fashion/r_pssim.txt
