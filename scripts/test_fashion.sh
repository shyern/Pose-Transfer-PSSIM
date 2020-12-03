#!/usr/bin/env bash

pro_dir="./Pose-Transfer-pSSIM"

######################################################################################
################################Testing###############################################
batchSize=1
dataset="fashion_data"
display_id=1
display_port=8097
gpu_ids=3,4
name="fashion_att_sty_l5"
which_epoch=300

dataroot=${pro_dir}"datasets/"
checkpoints_dir=${pro_dir}"checkpoints/checkpoints_fashion/checkpoints4"
results_dir=${pro_dir}"results/results_fashion/results4"
model=PATN
phase=test
dataset_mode=keypoint
norm=instance
resize_or_crop=no
BP_input_nc=18
which_model_netG=PATN
with_D_PP=0
with_D_PB=0
pairLst="fasion-resize-pairs-test.csv"
annoLst="fasion-resize-annotation-test.csv"
n_layers=3


/home/haoyue/anaconda3/envs/p37pytorch/bin/python ${pro_dir}test.py \
--dataroot=${dataroot} --dataset=${dataset} --name=${name} --which_epoch=${which_epoch} --model=${model} --dataset_mode=${dataset_mode} --n_layers=${n_layers} \
--norm=${norm} --batchSize=${batchSize} --resize_or_crop=${resize_or_crop} --gpu_ids=${gpu_ids} --BP_input_nc=${BP_input_nc} \
--phase=${phase} --which_model_netG=${which_model_netG} --checkpoints_dir=${checkpoints_dir} --results_dir=${results_dir} --pairLst=${pairLst} --annoLst=${annoLst} \
--with_D_PP=${with_D_PP} --with_D_PB=${with_D_PB} --display_id=${display_id} --display_port=${display_port} --no_flip

