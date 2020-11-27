#!/usr/bin/env bash

pro_dir="/home/haoyue/remote/Person-Image-Gen-pssim/"
#pro_dir="./Person-Image-Gen-pssim/"

######################################################################################
################################Train###############################################
L1_type="FPart_BSSIM_plus_perL1_L1"
#L1_type="l1_plus_perL1"
batchSize=32
dataset="market_data_s"
display_id=1
display_port=8098
gpu_ids=2
lambda_GAN=5
lambda_A=0
lambda_B=10
lambda_SSIM=10
lr=0.0002
name=market_pssim
win_sigma=0.8
win_size=7

dataroot=${pro_dir}"datasets/"
checkpoints_dir=${pro_dir}"checkpoints/checkpoints_market/checkpoints_pssim"
model=PATN
phase=train
dataset_mode=keypoint
norm=batch
resize_or_crop=no
BP_input_nc=18
which_model_netG=PATN
with_D_PP=1
with_D_PB=1
pairLst="market-pairs-train.csv"
annoLst="market-annotation-train.csv"
n_layers=3
niter=500
niter_decay=200
n_layers_D=3

/home/haoyue/anaconda3/envs/p37pytorch/bin/python ${pro_dir}train.py \
--dataroot=${dataroot} --dataset=${dataset} --name=${name} --model=${model} --lambda_GAN=${lambda_GAN} --lambda_A=${lambda_A} --lambda_B=${lambda_B} \
--lambda_SSIM=${lambda_SSIM} --dataset_mode=${dataset_mode} --n_layers=${n_layers} --norm=${norm} --batchSize=${batchSize} \
--resize_or_crop=${resize_or_crop} --gpu_ids=${gpu_ids} --lr=${lr} --BP_input_nc=${BP_input_nc} --phase=${phase} --which_model_netG=${which_model_netG} \
--niter=${niter} --niter_decay=${niter_decay} --checkpoints_dir=${checkpoints_dir} --pairLst=${pairLst} --annoLst=${annoLst} \
--L1_type=${L1_type} --n_layers_D=${n_layers_D} --with_D_PP=${with_D_PP} --with_D_PB=${with_D_PB} --display_id=${display_id} --display_port=${display_port} \
--win_sigma=${win_sigma} --win_size=${win_size} --no_lsgan --no_flip
#--continue_train --which_epoch 660 --epoch_count 661
