#!/usr/bin/env bash

python match_generation.py \
--query_dataset name_query.txt \
--query_embeddings ./experiments/MSLM_resnet101/query_70000_embeddings.h5 \
--gallery_dataset name_test.txt \
--gallery_embeddings ./experiments/MSLM_resnet101/test_70000_embeddings.h5 \
--metric euclidean \
--filename match70000_mslm_resnet101.txt \
--batch_size 256 \
--match_number 100