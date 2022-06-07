#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging

import torch
from torch import Tensor as T
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.optimization import AdamW

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


def get_bert_biencoder_components(args, inference_only: bool = False, **kwargs):
    question_encoder = HFBertEncoder(args)
    ctx_encoder = HFBertEncoder(args)

    fix_ctx_encoder = args.fix_ctx_encoder if hasattr(args, 'fix_ctx_encoder') else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = get_optimizer(biencoder, learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay) if not inference_only else None

    tensorizer = get_bert_tensorizer(args)

    return tensorizer, biencoder, optimizer


def get_bert_tensorizer(args):
    tokenizer = get_bert_tokenizer(args.pretrained_model_cfg)
    return BertTensorizer(tokenizer, args.sequence_length)


def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str):
    return AutoTokenizer.from_pretrained(pretrained_cfg_name)


class HFBertEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        dropout = args.dropout if hasattr(args, 'dropout') else 0.0

        cfg = AutoConfig.from_pretrained(args.pretrained_model_cfg)
        if dropout != 0:
            cfg.hidden_dropout_prob             = dropout
            cfg.attention_probs_dropout_prob    = dropout
        
        self.backbone = AutoModel.from_pretrained(args.pretrained_model_cfg, config=cfg)
    
    def forward(self, ids, attn_mask, segments):
        outputs = self.backbone(ids, attn_mask, segments)

        if self.args.pooling == "cls":
            pooling_output = outputs["last_hidden_state"][:, 0, :]
        elif self.args.pooling == "cls_pooler":
            pooling_output = outputs["pooler_output"]
        elif self.args.pooling == "avg":
            last_hidden_state = outputs["last_hidden_state"]
            last_hidden_state = last_hidden_state.masked_fill(torch.bitwise_not(attn_mask.unsqueeze(-1).expand_as(last_hidden_state)), 0)
            pooling_output = torch.sum(last_hidden_state, dim=1) / torch.clamp(torch.sum(attn_mask, dim=1, keepdim=True), min=1e-9)
        
        return pooling_output


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        if isinstance(text, list) and len(text) == 1:
            text = text[0]
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # FIXME: temporary enabling the tokenizer's truncation.
        if title:
            token_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=add_special_tokens,
                                              max_length=self.max_length, pad_to_max_length=False, truncation=True)
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad
