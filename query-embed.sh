#!/usr/bin/env bash

python embed.py \
    --experiment_root experiments/MSLM_resnet101 \
    --dataset ./name_query.txt \
    --image_root ./image_query/ \
    --checkpoint checkpoint-70000 \
    --batch_size 256 \
    --filename query_70000_embeddings.h5
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
