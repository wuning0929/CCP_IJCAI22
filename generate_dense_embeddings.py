#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os

import argparse
import csv
import logging
import pickle
import jsonlines
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device


def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str, str]], model: nn.Module, tensorizer: Tensorizer,
                    insert_title: bool = True) -> dict:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for batch_start in range(0, n, bsz):

        batch_token_tensors = [tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None) for ctx in
                               ctx_rows[batch_start:batch_start + bsz]]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), args.device)
        with torch.no_grad():
            out = model(ctx_ids_batch, ctx_attn_mask, ctx_seg_batch)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        for i in range(out.size(0)):
            results.append((ctx_ids[i], out[i].view(-1).numpy()))

        if total % 10 == 0:
            logger.info('Encoded passages %d', total)

    return results


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state, strict=False)

    if args.ctx_file.endswith('tsv'):  # for NQ dataset
        logger.info('reading data from %s', args.ctx_file)
        with open(args.ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            rows = [(row[0], row[1], row[2]) for row in reader if row[0] != 'id']
    elif args.ctx_file.endswith('jsonl'): # for Mr.TyDi dataset
        rows = []
        with jsonlines.open(args.ctx_file) as reader:
            for obj in reader:
                ctx = obj['contents'].split('\n\n')
                rows.append([obj['id'], ctx[1], ctx[0]])

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size if args.shard_id != args.num_shards - 1 else len(rows)

    logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]

    emb = gen_ctx_vectors(rows, encoder, tensorizer, True)

    file = args.out_file + '_' + str(args.shard_id)
    logger.info('Writing results to %s' % file)

    with open(file, "wb") as writer:
        pickle.dump(emb, writer)

    logger.info('Total passages processed %d. Written to %s', len(emb), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None, help='output .tsv file path to write results to ')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for the passage encoder forward pass")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    logger = logging.getLogger()

    logger.handlers.clear()
    
    if args.shard_id == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    log_format = '[%(asctime)s] [Rank {} - %(levelname)s] [%(filename)s - %(lineno)d] %(message)s'.format(args.local_rank)
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.addHandler(console)

    print_args(args)
    
    try:
        setup_args_gpu(args)
        main(args)
    except Exception as e:
        logger.exception(e)
