#!/bin/bash

set -x

python ./code/scripts/cluster/cluster_sobel_twohead.py \
--model_ind 640  \
--arch ClusterNet5gTwoHead \
--mode IID \
--dataset CIFAR10 \
--dataset_root /h/yanxi/Disk/datasets \
--gt_k 10 \
--output_k_A 70 \
--output_k_B 10 \
--lamb 1.0 \
--lr 0.0001  \
--num_epochs 2000 \
--batch_sz 660 \
--num_dataloaders 3 \
--num_sub_heads 5 \
--crop_orig \
--rand_crop_sz 20 \
--input_sz 32 \
--head_A_first \
--head_B_epochs 2
