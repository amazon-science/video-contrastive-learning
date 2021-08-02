#!/bin/bash

data_dir="/home/ubuntu/data/kinetics400_30fps_frames"
output_dir="./results"
pretrained="pretrain/moco_v2_200ep_pretrain.pth.tar"
num_replica=8

mkdir -p ${output_dir}

python3 -m torch.distributed.launch --master_port 12857 --nproc_per_node=${num_replica} \
    train_vclr.py \
    --data_dir=${data_dir} \
    --datasplit=train \
    --pretrained_model=${pretrained} \
    --output_dir=${output_dir} \
    --model_mlp \
    --dataset KineticsClipFolderDatasetOrderTSN \
    --batch_size 32
