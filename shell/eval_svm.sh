#!/bin/bash

data_dir="/home/ubuntu/data/kinetics400_30fps_frames"
output_dir="./results"
eval_dir="./results/eval_svm"
pretrained="./results/current.pth"
num_replica=8

mkdir -p ${output_dir}
mkdir -p ${eval_dir}

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=${num_replica} \
    eval_svm_feature_extract.py \
    --data_dir=${data_dir} \
    --datasplit=train \
    --pretrained_model=${pretrained} \
    --output_dir=${eval_dir}

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=${num_replica} \
    eval_svm_feature_extract.py \
    --data_dir=${data_dir} \
    --datasplit=val \
    --pretrained_model=${pretrained} \
    --output_dir=${eval_dir}

python3 eval_svm_feature_perf.py \
    --trainsplit=train \
    --valsplit=val \
    --output-dir=${eval_dir} \
    --num_replica=${num_replica} \
    --primal
