#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""


import argparse
import glob
import logging
import math
import os
import random
import time
import torch

from typing import Tuple
from torch import nn

from dpr.utils.dist_utils import all_gather_list
from dpr.models import init_biencoder_components

from dpr.models.biencoder import (
    BiEncoder, 
    BiEncoderNllLoss, 
    BiEncoderBatch
)

from dpr.options import (
    add_encoder_params,
    add_training_params, 
    setup_args_gpu, 
    set_seed, 
    print_args,
    get_encoder_params_state, 
    set_encoder_params_from_state
)

from dpr.utils.data_utils import (
    Tensorizer,
    ShardedDataIterator,
    MultiSetDataIterator, 
    read_data_from_json_files
)

from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    get_model_file, 
    get_model_obj, 
    load_states_from_checkpoint,
    CheckpointState
)


class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, args):
        self.args = args
        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, model, optimizer = init_biencoder_components(args.encoder_model_type, args)

        model, optimizer = setup_for_distributed_mode(
            model, 
            optimizer, 
            args.device, 
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level
        )

        self.biencoder              = model
        self.optimizer              = optimizer
        self.tensorizer             = tensorizer
        self.start_epoch            = 0
        self.start_batch            = 0
        self.scheduler_state        = None
        self.best_cp_name           = None
        self.best_validation_result = None
        self.val_avg_rank           = True
        self.restart                = self.args.restart

        if saved_state:
            self._load_saved_state(saved_state)
        
        self.create_input = BiEncoder.create_biencoder_input

    def get_data_iterator(
        self, 
        path: str, 
        batch_size: int, 
        shuffle=True, 
        shuffle_seed: int = 0,
        offset: int = 0, 
        upsample_rates: list = None
    ) -> ShardedDataIterator:

        data_files = glob.glob(path)
        data = read_data_from_json_files(data_files, upsample_rates)

        # filter those without positive ctx
        data = [[r for r in d if len(r['positive_ctxs']) > 0] for d in data]

        iterators = [ShardedDataIterator(
            d, 
            shard_id=self.shard_id, 
            num_shards=self.distributed_factor, 
            batch_size=batch_size, 
            shuffle=shuffle,
            shuffle_seed=shuffle_seed, 
        ) for d in data]

        iterator = MultiSetDataIterator(
            iterators,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed, 
            offset=offset
        )
        
        logger.info('Total cleaned data size: {}, total traing data size: {}'.format(
            sum([len(d) for d in data]), 
            iterator.total_data_len()
        ))

        return iterator

    def run_train(self):
        args = self.args
        upsample_rates = None
        if args.train_files_upsample_rates is not None:
            upsample_rates = eval(args.train_files_upsample_rates)

        train_iterator = self.get_data_iterator(
            args.train_file, 
            args.batch_size,
            shuffle=True,
            shuffle_seed=args.seed, 
            offset=self.start_batch,
            upsample_rates=upsample_rates
        )

        dev_iterator = self.get_data_iterator(
            args.dev_file, 
            args.dev_batch_size, 
            shuffle=False
        )

        logger.info("Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = train_iterator.max_iterations // args.gradient_accumulation_steps
        total_updates = max(updates_per_epoch * (args.num_train_epochs - self.start_epoch - 1), 0) + \
                        (train_iterator.max_iterations - self.start_batch) // args.gradient_accumulation_steps
        logger.info("Total updates=%d", total_updates)
        warmup_steps = args.warmup_steps
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        if self.scheduler_state and self.restart is False:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = int(updates_per_epoch / args.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        self.validate_and_save(0, train_iterator.get_iteration(), scheduler, dev_iterator)

        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator, dev_iterator)

        if args.local_rank in [-1, 0]:
            logger.info('Training finished. Best validation checkpoint %s', self.best_cp_name)

    def validate_and_save(self, epoch: int, iteration: int, scheduler, data_iterator: MultiSetDataIterator):
        args = self.args
        # for distributed mode, save checkpoint for only one process
        save_cp = args.local_rank in [-1, 0]

        if epoch == args.val_av_rank_start_epoch and self.val_avg_rank:
            self.best_validation_result = None
            self.val_avg_rank = False

        if epoch >= args.val_av_rank_start_epoch:
            validation_loss = self.validate_average_rank(data_iterator)
        else:
            validation_loss = self.validate_nll(data_iterator)

        if save_cp:

            if validation_loss < (self.best_validation_result or validation_loss + 1):

                cp_name = self._save_checkpoint(scheduler, epoch, iteration)

                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info('New Best validation checkpoint %s', cp_name)

    def validate_nll(self, data_iterator) -> float:
        logger.info('NLL validation ...')
        args = self.args
        self.biencoder.eval()
        
        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = args.hard_negatives
        num_other_negatives = args.other_negatives
        log_result_step = args.log_batch_step
        batches = 0
        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            biencoder_input = self.create_input(
                samples_batch,
                self.tensorizer, 
                not self.args.not_insert_title, 
                num_hard_negatives, 
                num_other_negatives, 
                shuffle=False,
            )

            loss, correct_cnt = _do_biencoder_fwd_pass(self.biencoder, biencoder_input, self.tensorizer, args)
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info('Eval step: %d , used_time=%f sec., loss=%f ', i, time.time() - start_time, loss.item())

        total_loss = total_loss / batches
        total_samples = data_iterator.total_data_len()
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            'NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', 
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio
        )
        return total_loss

    def validate_average_rank(self, data_iterator) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info('Average rank validation ...')

        args = self.args
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        sub_batch_size = args.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss(args).similarity_function
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = args.val_av_rank_hard_neg
        num_other_negatives = args.val_av_rank_other_neg

        log_result_step = args.log_batch_step

        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            # samples += 1
            if len(q_represenations) > args.val_av_rank_max_qs / distributed_factor:
                break

            biencoder_input = self.create_input(
                samples_batch,
                self.tensorizer, 
                not self.args.not_insert_title, 
                num_hard_negatives, 
                num_other_negatives, 
                shuffle=False,
            )
            biencoder_input = BiEncoderBatch(**move_to_device(biencoder_input._asdict(), args.device))

            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (biencoder_input.question_ids, biencoder_input.question_segments) if j == 0 \
                    else (None, None)

                if j == 0 and args.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start:batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[batch_start:batch_start + sub_batch_size]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids, 
                        q_segments, 
                        q_attn_mask, 
                        ctx_ids_batch, 
                        ctx_seg_batch, 
                        ctx_attn_mask
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend([total_ctxs + v for v in batch_positive_idxs])

            if (i + 1) % log_result_step == 0:
                logger.info('Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d', i,
                            len(ctx_represenations), len(q_represenations))

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info('Av.rank validation: total q_vectors size=%s', q_represenations.size())
        logger.info('Av.rank validation: total ctx_vectors size=%s', ctx_represenations.size())

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        positive_idx_tensor = torch.tensor(positive_idx_per_question, device=q_represenations.device).long()
        scores = sim_score_f(q_represenations, ctx_represenations, positive_idx_tensor, args)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != args.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        av_rank = float(rank / q_num)
        logger.info('Av.rank validation: average rank %s, total questions=%d', av_rank, q_num)
        return av_rank

    def _train_epoch(
        self, 
        scheduler, 
        epoch: int, 
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
        dev_data_iterator: MultiSetDataIterator):

        args = self.args
        rolling_train_loss = 0
        epoch_loss = 0
        epoch_correct_predictions = 0

        log_result_step = args.log_batch_step
        rolling_loss_step = args.train_rolling_loss_step
        num_hard_negatives = args.hard_negatives
        num_other_negatives = args.other_negatives
        seed = args.seed
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0
        for i, samples_batch in enumerate(train_data_iterator.iterate_data(epoch=epoch)):

            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)
            biencoder_batch = self.create_input(
                samples_batch, 
                self.tensorizer, 
                not self.args.not_insert_title, 
                num_hard_negatives, 
                num_other_negatives, 
                shuffle=True, 
                shuffle_positives=args.shuffle_positive_ctx
            )

            loss, correct_cnt = _do_biencoder_fwd_pass(self.biencoder, biencoder_batch, self.tensorizer, args)

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()

            rolling_train_loss += loss.item()

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), args.max_grad_norm)

            if (i + 1) % args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if (i + 1) % log_result_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info('Epoch: %d: Step: %d/%d, loss=%f, lr=%f', epoch, data_iteration, epoch_batches, loss.item(), lr)

            if (i + 1) % rolling_loss_step == 0:
                logger.info('Avg. loss per Last %d batches: loss=%f', rolling_loss_step, rolling_train_loss / rolling_loss_step)
                rolling_train_loss = 0

            if data_iteration % eval_step == 0:
                logger.info('Validation: Epoch: %d Step: %d/%d', epoch, data_iteration, epoch_batches)
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler, dev_data_iterator)
                self.biencoder.train()

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info('Av Loss per epoch=%f', epoch_loss)
        logger.info('epoch total correct predictions=%d', epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        args = self.args
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(args.output_dir, 'best.pt')

        meta_params = get_encoder_params_state(args)

        state = CheckpointState(model_to_save.state_dict(),
                                self.optimizer.state_dict(),
                                scheduler.state_dict(),
                                offset, epoch, meta_params)
        torch.save(state._asdict(), cp)
        logger.info('Saved checkpoint at %s', cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info('Loading checkpoint @ batch=%s and epoch=%s', offset, epoch)

        if self.restart is False:
            self.start_epoch = epoch 
            self.start_batch = offset

        model_to_load = get_model_obj(self.biencoder)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)

        if self.restart is False and saved_state.optimizer_dict:
            logger.info('Loading saved optimizer state ...')
            self.optimizer.load_state_dict(saved_state.optimizer_dict)

        if self.restart is False and saved_state.scheduler_dict:
            self.scheduler_state = saved_state.scheduler_dict


def _calc_loss(args, loss_function, local_q_vector, local_ctx_vectors, local_positive_idxs,
               local_hard_negatives_idxs: list = None):
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = args.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

        global_question_ctx_vectors = all_gather_list(
            [q_vector_to_send, ctx_vector_to_send, local_positive_idxs, local_hard_negatives_idxs],
            max_size=args.global_loss_buf_sz)

        global_q_vector = []
        global_ctxs_vector = []

        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != args.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negatives_idxs])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negatives_idxs])
                
            total_ctxs += ctx_vectors.size(0)

        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct = loss_function.calc(global_q_vector, global_ctxs_vector, positive_idx_per_question, hard_negatives_per_question)

    return loss, is_correct


def _do_biencoder_fwd_pass(model: nn.Module, input: BiEncoderBatch, tensorizer: Tensorizer, args) -> Tuple[torch.Tensor, int]:
    input = BiEncoderBatch(**move_to_device(input._asdict(), args.device))

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(input.question_ids, input.question_segments, q_attn_mask, input.context_ids,
                          input.ctx_segments, ctx_attn_mask)
    else:
        with torch.no_grad():
            model_out = model(input.question_ids, input.question_segments, q_attn_mask, input.context_ids,
                              input.ctx_segments, ctx_attn_mask)

    local_q_vector, local_ctx_vectors = model_out

    loss_function = BiEncoderNllLoss(args)

    loss, is_correct = _calc_loss(args, loss_function, local_q_vector, local_ctx_vectors, input.is_positive, input.hard_negatives)

    is_correct = is_correct.sum().item()

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss, is_correct


def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)

    # biencoder specific training features
    parser.add_argument("--eval_per_epoch", default=1, type=int,
                        help="How many times it evaluates on dev set per epoch and saves a checkpoint")

    parser.add_argument("--global_loss_buf_sz", type=int, default=1000000,
                        help='Buffer size for distributed mode representations al gather operation. \
                                Increase this if you see errors like "encoded data exceeds max_size ..."')

    parser.add_argument("--fix_ctx_encoder", action='store_true')
    parser.add_argument("--shuffle_positive_ctx", action='store_true')
    parser.add_argument("--not_insert_title", action='store_true')

    # input/output src params
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be written or resumed from")

    # data handling parameters
    parser.add_argument("--hard_negatives", default=1, type=int,
                        help="amount of hard negative ctx per question")
    parser.add_argument("--other_negatives", default=0, type=int,
                        help="amount of 'other' negative ctx per question")
    parser.add_argument("--train_files_upsample_rates", type=str,
                        help="list of up-sample rates per each train file. Example: [1,2,1]")

    # parameters for Av.rank validation method
    parser.add_argument("--val_av_rank_start_epoch", type=int, default=10000,
                        help="Av.rank validation: the epoch from which to enable this validation")
    parser.add_argument("--val_av_rank_hard_neg", type=int, default=30,
                        help="Av.rank validation: how many hard negatives to take from each question pool")
    parser.add_argument("--val_av_rank_other_neg", type=int, default=30,
                        help="Av.rank validation: how many 'other' negatives to take from each question pool")
    parser.add_argument("--val_av_rank_bsz", type=int, default=512,
                        help="Av.rank validation: batch size to process passages")
    parser.add_argument("--val_av_rank_max_qs", type=int, default=15000,
                        help="Av.rank validation: max num of questions")
    parser.add_argument('--checkpoint_file_name', type=str, default='dpr_biencoder', help="Checkpoints file prefix")
    parser.add_argument("--restart", action='store_true', 
                        help="set true if you want to reset the optimizer and schedular states when you start fine-tuning from pre-trained models")

    args = parser.parse_args()
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    if args.output_dir is not None and args.local_rank in [-1, 0]:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    log_format = '[%(asctime)s] [{} - %(levelname)s] [%(filename)s - %(lineno)d] %(message)s'.format(args.local_rank)
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.addHandler(console)

    if args.local_rank in [-1, 0]:
        logger.setLevel(logging.INFO)
        if args.train_file is not None:
            file = logging.FileHandler(os.path.join(args.output_dir, 'train.log'), mode='a')
            file.setFormatter(log_format)
            logger.addHandler(file)
    
    setup_args_gpu(args)
    set_seed(args)
    
    if args.local_rank != -1:
        torch.distributed.barrier()

    if args.local_rank not in [-1, 0]:
        logger.setLevel(logging.WARNING)
        if args.train_file is not None:
            file = logging.FileHandler(os.path.join(args.output_dir, 'train.log'), mode='a')
            file.setFormatter(log_format)
            logger.addHandler(file)

    print_args(args)

    try:
        trainer = BiEncoderTrainer(args)
        if args.train_file is not None:
            trainer.run_train()
        else:
            logger.warning("Nothing to do")
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":

    logger = logging.getLogger()

    main()
