#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
import jsonlines
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                out = self.question_encoder(q_ids_batch, q_attn_mask, q_seg_batch)
                out = out.cpu()

                query_vectors.extend(out.split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)

        return query_tensor

    def index_encoded_data(self, vector_files: List[str], buffer_size: int = 50000):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            buffer.append((db_id, doc_vector))
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info('Data indexing completed.')

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results

def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.startswith(".gz"):
        with gzip.open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results_nq(
    passages: Dict[object, Tuple[str, str]], 
    questions: List[str], 
    answers: List[List[str]], 
    top_ids_and_scores: List[Tuple[List[object], List[float]]], 
    per_question_hits: List[List[bool]], 
    out_file: str
):

    merged_data = []
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_ids_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as fw:
        json.dump(merged_data, fw)


def save_results_mrtydi(
    q_ids: List[str], 
    top_ids_and_scores: List[Tuple[List[object], List[float]]], 
    out_file
):

    with open(out_file, "w") as fw:
        for q_id, top_id_and_score in zip(q_ids, top_ids_and_scores):
            for i, top_id, top_score in zip(
                range(len(top_id_and_score[0])), 
                top_id_and_score[0], 
                top_id_and_score[1].tolist()
            ):
                fw.write("{} Q0 {} {} {} FAISS\n".format(q_id, top_id, i + 1, top_score))


def save_results_xorqa(
    passages, 
    q_ids, 
    languages,
    top_ids_and_scores, 
    out_file: str
):
    results = []
    for ids_and_scores, q_id, lang in zip(top_ids_and_scores, q_ids, languages):
        ctxs = [passages[doc_id][0] for doc_id in ids_and_scores[0]]
        results.append({"q_id": q_id, "lang": lang, "ctxs" : ctxs})
    
    with open(out_file, 'w') as outfile:
        json.dump(results, outfile)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for file in vector_files:
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith('question_model.')}
    model_to_load.load_state_dict(question_encoder_state, strict=False)
    vector_size = model_to_load.backbone.config.hidden_size
    logger.info('Encoder vector_size=%d', vector_size)

    index_buffer_sz = args.index_buffer
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size)
        index_buffer_sz = -1  # encode all at once
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = ctx_files_pattern.replace('_*', '')
    if args.save_or_load_index and retriever.index.exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    
    if args.ctx_file is not None:
        all_passages = load_passages(args.ctx_file)

    for in_file, out_file in zip(args.input_file, args.out_file):
        q_ids, questions, languages = [], [], []

        if args.data == 'mrtydi':  # get questions for Mr.Tydi
            with open(in_file) as ifile:
                reader = csv.reader(ifile, delimiter='\t')
                for row in reader:
                    q_ids.append(row[0])
                    questions.append(row[1])
        elif args.data == 'xorqa':  # get questions for XOR-QA
            with jsonlines.open(in_file) as reader:
                for obj in reader:
                    q_ids.append(obj["id"])
                    questions.append(obj["question"])
                    languages.append(obj["lang"])

        questions_tensor = retriever.generate_question_vectors(questions)

        # get top k results
        top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

        if args.data == 'mrtydi':
            save_results_mrtydi(q_ids, top_ids_and_scores, out_file)
        elif args.data == 'xorqa':
            save_results_xorqa(all_passages, q_ids, languages, top_ids_and_scores, out_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--data', type=str, default='nq', help='Data format')
    parser.add_argument('--input_file', type=str, default=None, nargs='+',
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None, nargs='+',
                        help='output .tsv file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=500000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument('--log_file', type=str, default=None, help="log output")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file[0]), exist_ok=True)

    logger = logging.getLogger()

    logger.handlers.clear()
    
    logger.setLevel(logging.INFO)

    log_format = '[%(asctime)s] [Rank {} - %(levelname)s] [%(filename)s - %(lineno)d] %(message)s'.format(args.local_rank)
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.addHandler(console)

    file = logging.FileHandler(os.path.join(args.log_file), mode='a')
    file.setFormatter(log_format)
    logger.addHandler(file)

    print_args(args)

    try:
        setup_args_gpu(args)
        main(args)
    except Exception as e:
        logger.exception(e)
