#!/usr/bin/env bash

pro_dir="./Pose-Transfer-pSSIM"

######################################################################################
################################Train###############################################
L1_type="l1_plus_perL1"
batchSize=8
dataset="fashion_data"
display_id=1
display_port=8096
gpu_ids=0,6
lambda_GAN=5
lambda_A=10
lambda_B=10
lambda_SSIM=0
lr=0.0001
name=fashion_sty_r


dataroot=${pro_dir}"datasets/"
checkpoints_dir=${pro_dir}"checkpoints/checkpoints_fashion/checkpoints4"
model=PATN
phase=train
dataset_mode=keypoint
norm=instance
resize_or_crop=no
BP_input_nc=18
which_model_netG=PATN
with_D_PP=1
with_D_PB=1
pairLst="fasion-resize-pairs-train.csv"
annoLst="fasion-resize-annotation-train.csv"
n_layers=3
niter=200
niter_decay=100
n_layers_D=3
pool_size=0


/home/haoyue/anaconda3/envs/p37pytorch/bin/python ${pro_dir}train.py \
--dataroot=${dataroot} --dataset=${dataset} --name=${name} --model=${model} --lambda_GAN=${lambda_GAN} --lambda_A=${lambda_A} --lambda_B=${lambda_B} \
--lambda_SSIM=${lambda_SSIM} --dataset_mode=${dataset_mode} --n_layers=${n_layers} --norm=${norm} --batchSize=${batchSize} --pool_size=${pool_size} \
--resize_or_crop=${resize_or_crop} --gpu_ids=${gpu_ids} --lr=${lr} --BP_input_nc=${BP_input_nc} --phase=${phase} --which_model_netG=${which_model_netG} \
--niter=${niter} --niter_decay=${niter_decay} --checkpoints_dir=${checkpoints_dir} --pairLst=${pairLst} --annoLst=${annoLst} \
--L1_type=${L1_type} --n_layers_D=${n_layers_D} --with_D_PP=${with_D_PP} --with_D_PB=${with_D_PB} --display_id=${display_id} --display_port=${display_port} --no_flip \
#--continue_train --which_epoch 300 --epoch_count 301
