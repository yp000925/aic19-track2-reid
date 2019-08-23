#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from itertools import count
import os

import h5py
import numpy as np
import tensorflow as tf
import loss


import common

parser = ArgumentParser(description='Evaluate a ReID embedding.')

# parser.add_argument(
#     '--excluder', required=True, choices=('market1501', 'diagonal','veri776'),
#     help='Excluder function to mask certain matches. Especially for multi-'
#          'camera datasets, one often excludes pictures of the query person from'
#          ' the gallery if it is taken from the same camera. The `diagonal`'
#          ' excluder should be used if this is *not* required.')

parser.add_argument(
    '--query_dataset', required=True,
    help='Path to the query dataset txt file.')

parser.add_argument(
    '--query_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--gallery_dataset', required=True,
    help='Path to the gallery dataset txt file.')

parser.add_argument(
    '--gallery_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--metric', required=True, choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--filename', type=FileType('w'),
    help='Optional name of the txt file to store the results in.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on your memory usage.')

parser.add_argument(
    '--match_number', default=100, type=common.positive_int,
    help='Number of matches output in the txt file ')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    # Load the query and gallery data from the txt files.
    query_fids = common.load_from_txt(args.query_dataset, None)
    gallery_fids = common.load_from_txt(args.gallery_dataset, None)

    # Load the two datasets fully into memory.
    with h5py.File(args.query_embeddings, 'r') as f_query:
        query_embs = np.array(f_query['emb'])
    with h5py.File(args.gallery_embeddings, 'r') as f_gallery:
        gallery_embs = np.array(f_gallery['emb'])

    # Just a quick sanity check that both have the same embedding dimension!
    query_dim = query_embs.shape[1]
    gallery_dim = gallery_embs.shape[1]
    if query_dim != gallery_dim:
        raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                         'dimension'.format(query_dim, gallery_dim))

    # Setup the dataset specific matching function
    # excluder = import_module('excluders.' + args.excluder).Excluder(gallery_fids)

    # We go through the queries in batches, but we always need the whole gallery
    batch_fids, batch_embs = tf.data.Dataset.from_tensor_slices(
        (query_fids, query_embs)
    ).batch(args.batch_size).make_one_shot_iterator().get_next()

    batch_distances = loss.cdist(batch_embs, gallery_embs, metric=args.metric)

    with tf.Session(config=config) as sess:
        match_fids =[]
        query_figure = []
        for start_idx in count(step=args.batch_size):
            try:
                # Compute distance to all gallery embeddings
                distances, fids = sess.run([
                    batch_distances, batch_fids])
                print('\rEvaluating batch {}-{}/{}'.format(
                        start_idx, start_idx + len(fids), len(query_fids)),
                      flush=True, end='')
            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

            index = np.argsort(distances, axis=1)
            index.astype(int)
            #pre_fids = gallery_fids[index[:, 0:args.match_number]]
            pre_fids = index[:,0:args.match_number] +1
            if match_fids == []:
                match_fids = pre_fids
            else:
                match_fids = np.concatenate([match_fids, pre_fids], axis=0)
            query_figure = np.append(query_figure,fids.tolist())

    # match_fids, query_figure = np.array(match_fids,'|U'), np.array(query_figure,'|U')

    # Save important data
    if args.filename is not None:
        np.savetxt(args.filename, match_fids, delimiter=' ', fmt='%d')

    # Print out a short summary.
    print('The top {} match for each query have been outputed'.format(args.match_number))

if __name__ == '__main__':
    main()

