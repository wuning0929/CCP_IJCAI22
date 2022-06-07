#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""
import re
import collections
import logging
import random
from typing import Tuple, List


import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.utils.data_utils import Tensorizer
from dpr.utils.data_utils import normalize_question

logger = logging.getLogger()

BiEncoderBatch = collections.namedtuple('BiENcoderInput',
                                        ['question_ids', 'question_segments', 'context_ids', 'ctx_segments', 'is_positive', 'hard_negatives'])


def dot_product_scores(q_vectors: T, ctx_vectors: T, positive: T, args) -> T:
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))


def cosine_scores(q_vectors: T, ctx_vectors: T, positive: T, args):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return args.scale * torch.matmul(F.normalize(q_vectors), torch.transpose(F.normalize(ctx_vectors), 0, 1))

class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, question_model: nn.Module, ctx_model: nn.Module, fix_q_encoder: bool = False,
                 fix_ctx_encoder: bool = False):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(sub_model: nn.Module, ids: T, segments: T, attn_mask: T, fix_encoder: bool = False) -> Tuple[T, T, T]:
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    output = sub_model(ids, attn_mask, segments)

                if sub_model.training:
                    output.requires_grad_(requires_grad=True)
            else:
                output = sub_model(ids, attn_mask, segments)

            return output
        else:
            return None

    def forward(self, question_ids: T, question_segments: T, question_attn_mask: T, context_ids: T, ctx_segments: T,
                ctx_attn_mask: T) -> Tuple[T, T]:

        q_pooled_out = self.get_representation(self.question_model, question_ids, question_segments, question_attn_mask, self.fix_q_encoder)
        ctx_pooled_out = self.get_representation(self.ctx_model, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder)

        return q_pooled_out, ctx_pooled_out

    @classmethod
    def create_biencoder_input(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            if shuffle and shuffle_positives:
                positive_ctxs = sample['positive_ctxs']
                index = np.random.choice(len(positive_ctxs))
                positive_ctx = positive_ctxs[index]
            else:
                positive_ctx = sample['positive_ctxs'][0]

            neg_ctxs = sample['negative_ctxs']
            hard_neg_ctxs = sample['hard_negative_ctxs']
            question = normalize_question(sample['question'])

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [tensorizer.text_to_tensor(ctx['text'], title=ctx['title'] if insert_title else None)
                                   for ctx in all_ctxs]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [i for i in
                 range(current_ctxs_len + hard_negatives_start_idx, current_ctxs_len + hard_negatives_end_idx)])

            question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(questions_tensor, question_segments, ctxs_tensor, ctx_segments, positive_ctx_indices, hard_neg_ctx_indices)


class BiEncoderNllLoss(object):

    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.similarity_function = dot_product_scores

    def calc(
        self, 
        q_vectors: T, 
        ctx_vectors: T, 
        positive_idx_per_question: list, 
        hard_negatice_idx_per_question: list = None, 
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negatice_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """

        positive_idx_tensor = torch.tensor(positive_idx_per_question, device=q_vectors.device).long()

        scores = self.similarity_function(q_vectors, ctx_vectors, positive_idx_tensor, self.args)

        loss = F.cross_entropy(scores, positive_idx_tensor, reduction='mean')

        max_idxs = torch.argmax(scores, 1)
        correct_predictions_count = torch.eq(max_idxs, positive_idx_tensor).long().sum()

        return loss, correct_predictions_count
