#!/bin/bash

datasource="ucf"
data_dir="/home/ubuntu/data/ucf101"
#datasource="hmdb"
#data_dir="/home/ubuntu/data/hmdb51"
output_dir="./results"
eval_dir="./results/eval_retrieval"
pretrained="./results/current.pth"
layer=6
num_replica=1

mkdir -p ${output_dir}
mkdir -p ${eval_dir}

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=${num_replica} \
    eval_retrieval_feature_extract.py \
    --data_dir=${data_dir} \
    --datamode=train \
    --data-source=${datasource} \
    --layer=${layer} \
    --pretrained_model=${pretrained} \
    --output_dir=${eval_dir}

python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=${num_replica} \
    eval_retrieval_feature_extract.py \
    --data_dir=${data_dir} \
    --datamode=val \
    --data-source=${datasource} \
    --layer=${layer} \
    --pretrained_model=${pretrained} \
    --output_dir=${eval_dir}

python3 eval_retrieval_knn_pred.py \
    --trainsplit=train \
    --valsplit=val \
    --data-source=${datasource} \
    --output-dir=${eval_dir} \
    --num_replica=${num_replica}
