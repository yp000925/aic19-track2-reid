#!/usr/bin/env bash

IMAGE_ROOT=./image_train; shift
EXP_ROOT=./experiments/MSLM_resNet101; shift
INIT_CKPT=../DaRi/pre_trained_model/resnet_v1_101.ckpt; shift

python train.py\
    --experiment_root $EXP_ROOT \
    --train_set ./train_label.csv \
    --image_root $IMAGE_ROOT \
    --head_name fusion_resnet101 \
    --embedding_dim 128 \
    --initial_checkpoint $INIT_CKPT \
    --loss batch_hard \
    --model_name resnet_v1_101 \
    --learning_rate 3e-4 \
    --train_iterations 70000 \
    --decay_start_iteration 15000 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --flip_augment \
    --crop_augment \
    --detailed_logs \
    --margin soft \
    --metric euclidean \
    --weight_decay_factor 0.001 \
    --checkpoint_frequency 5000 \
    "$@"

