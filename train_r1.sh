#!/bin/bash

set -x

# single sub-head, no sample repeats, should be fast

python ./code/scripts/cluster/cluster_sobel_twohead.py \
--model_ind $1  \
--arch ClusterNet5gTwoHead \
--mode IID \
--dataset IMAGENET32 \
--dataset_root /h/yanxi/Disk/datasets/downsampled-imagenet-32 \
--gt_k 10 \
--output_k_A 70 \
--output_k_B 10 \
--lamb 1.0 \
--lr 0.01  \
--num_epochs 256 \
--batch_sz 5000 \
--num_dataloaders 1 \
--num_sub_heads 1 \
--crop_orig \
--rand_crop_sz 32 \
--input_sz 32 \
--head_A_first \
--head_B_epochs 2 \
--batchnorm_track \
