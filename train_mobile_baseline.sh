#!/usr/bin/env bash

IMAGE_ROOT=./image_train; shift
INIT_CKPT=./pre_trained_model/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt; shift
EXP_ROOT=./experiments/baseline_mobileNet; shift

python train_triplet.py\
    --experiment_root $EXP_ROOT \
    --train_set ./train_label.csv \
    --image_root $IMAGE_ROOT \
    --head_name fc1024 \
    --embedding_dim 128 \
    --initial_checkpoint $INIT_CKPT \
    --loss batch_hard \
    --model_name mobilenet_v1_1_224 \
    --learning_rate 4e-4 \
    --train_iterations 25000 \
    --decay_start_iteration 5000 \
    --lr_decay_factor 0.96 \
    --flip_augment \
    --crop_augment \
    --detailed_logs \
    --margin soft \
    --metric euclidean \
    --weight_decay_factor 0.0002 \
    --checkpoint_frequency 500 \
    --lr_decay_steps 1500 \
    "$@"

