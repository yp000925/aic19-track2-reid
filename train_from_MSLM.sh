#!/usr/bin/env bash

IMAGE_ROOT=./image_train; shift
INIT_CKPT=./pre_trained_model/MSLM_resnet/MSLM_resnet; shift
EXP_ROOT=./experiments/MSLM_resNet; shift

python train.py\
    --experiment_root $EXP_ROOT \
    --train_set ./train_label.csv \
    --image_root $IMAGE_ROOT \
    --head_name fusion_resnet50 \
    --embedding_dim 128 \
    --initial_checkpoint $INIT_CKPT \
    --loss batch_hard \
    --model_name resnet_v1_50 \
    --learning_rate 3e-4 \
    --train_iterations 15000 \
    --decay_start_iteration 5000 \
    --lr_decay_factor 0.96 \
    --flip_augment \
    --crop_augment \
    --detailed_logs \
    --margin soft \
    --metric euclidean \
    --weight_decay_factor 0.0002 \
    --checkpoint_frequency 500 \
    "$@"

