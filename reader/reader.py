import argparse
import logging
import os
import random
import glob
import timeit
import json
import linecache
import faiss
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
import pytrec_eval
import scipy as sp
from copy import copy
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from model import Pipeline, AlbertForRetrieverOnlyPositivePassage, BertForOrconvqaGlobal, AlbertForOrconvqaGlobal
from utils import (LazyQuacDatasetGlobal, RawResult, 
                   write_predictions, write_final_predictions, 
                   get_retrieval_metrics, gen_reader_features, ReaderDataset)
from scorer import quac_eval


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
}
# MODEL_CLASSES = {
#     'reader': (AlbertConfig, AlbertForOrconvqaGlobal, AlbertTokenizer),
# }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))



    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(
    #     train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_portion)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    qa_tr_loss, qa_logging_loss = 0.0, 0.0
    rerank_tr_loss, rerank_logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)

    check = {}
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):

            qids = batch['qid']
            question_texts = batch['question_text']
            answer_texts = batch["answer_text"]
            answer_starts = batch["answer_start"]
            passages = batch["passages"]
            retrieval_labels = batch["retrieval_labels"]
            retrieval_labels = np.array([np.array(label) for label in retrieval_labels]).T
        
            passages = np.array([np.array(p) for p in passages]).T
            pids = np.array([[1, 2, 3, 4, 5] for _ in range(args.per_gpu_train_batch_size)])
            
            

            reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
                                        pids, passages, retrieval_labels,
                                        tokenizer, args.reader_max_seq_length, is_training=True)

            reader_batch = {k: v.to(args.device) for k, v in reader_batch.items()}
            model.train()
            inputs = {'input_ids':       reader_batch['input_ids'],
                      'attention_mask':  reader_batch['input_mask'],
                      'token_type_ids':  reader_batch['segment_ids'],
                      'start_positions': reader_batch['start_position'],
                      'end_positions':   reader_batch['end_position'],
                      'retrieval_label': reader_batch['retrieval_label']}
            
            reader_outputs = model(**inputs)
            
            loss, qa_loss, rerank_loss = reader_outputs[:3]

            if args.n_gpu > 1:
                loss = loss.mean()
                qa_loss = qa_loss.mean()
                rerank_loss = rerank_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                qa_loss = qa_loss / args.gradient_accumulation_steps
                rerank_loss = rerank_loss / args.gradient_accumulation_steps

            a = []
            for i in range(len(list(model.parameters()))):
                a.append(list(model.parameters())[i].clone())
                
            loss.backward()
            
            tr_loss += loss.item()
            qa_tr_loss += qa_loss.item()
            rerank_tr_loss += rerank_loss.item()

            print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                b = []
                for i in range(len(list(model.parameters()))):
                    b.append(list(model.parameters())[i].clone())
                
                for i in range(len(a)):
                    if not torch.equal(a[i].data, b[i].data):
                        print("changed")
                        break
                model.zero_grad()
                # print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))
                global_step += 1
                # print(scheduler.get_lr()[0])
                

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, mode="train")
                        # for key, value in results.items():
                        #     tb_writer.add_scalar(
                        #         'eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar(
                    #     'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'qa_loss', (qa_tr_loss - qa_logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'rerank_loss', (rerank_tr_loss - rerank_logging_loss)/args.logging_steps, global_step)

                    
                    logger.info(f"step: {step}, tr_loss: {tr_loss-logging_loss}, qa_tr_loss: {qa_tr_loss-qa_logging_loss}, rerank_tr_loss: {rerank_tr_loss-rerank_logging_loss}")
                    logging_loss = tr_loss
                    qa_logging_loss = qa_tr_loss
                    rerank_logging_loss = rerank_tr_loss
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    
                    reader_model_dir = os.path.join(output_dir, 'reader')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if not os.path.exists(reader_model_dir):
                        os.makedirs(reader_model_dir)

                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(reader_model_dir)

                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))

                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                logger.info("epoch_iterator close")
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            logger.info("train_iterator close")
            break
    if args.local_rank in [-1, 0]:
        logger.info("tb_writer close")
        tb_writer.close()

    return global_step, tr_loss / global_step


# mode=train if evaluating while training
def evaluate(args, model, tokenizer, prefix="", mode="eval"):
    retrieval_output = gen_retrieve_result(args)
    DatasetClass = ReaderDataset
    eval_dataset = DatasetClass(args.dev_file,
                                 args.load_small, args.history_num,
                                 given_query=True,
                                 given_passage=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    if mode == "train":
        predict_dir = os.path.join(args.output_dir, 'predictions_train')
    else:
        predict_dir = os.path.join(args.output_dir, 'predictions_eval')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    examples, features = {}, {}
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = batch['qid']
        question_texts = batch['question_text']
        answer_texts = batch["answer_text"]
        answer_starts = batch["answer_start"]
        retrieval_labels = batch["retrieval_labels"]
        retrieval_labels = np.array([np.array(label) for label in retrieval_labels]).T
        passages = batch["passages"]
        passages = np.array([np.array(p) for p in passages]).T
        pids = np.array([[1, 2, 3, 4, 5] for _ in range(args.per_gpu_eval_batch_size)])
        
        # retrieval_labels = np.zeros((args.per_gpu_eval_batch_size, args.top_k_for_reader))
        # passages = np.array([np.array(p) for p in passages]).T
        # passages = np.array([retrieval_output[qid]["docs"][:args.top_k_for_reader] for qid in qids])
        
        # pids = np.array([retrieval_output[qid]["pids"][:args.top_k_for_reader] for qid in qids])

        # retriever_probs = np.array([retrieval_output[qid]["retrieval_scores"][:args.top_k_for_reader] for qid in qids])
        retriever_probs = np.zeros((4, 5))
        # reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
        #                             pids, passages, retrieval_labels,
        #                             tokenizer, args.reader_max_seq_length, is_training=False)
        reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
                                        pids, passages, retrieval_labels,
                                        tokenizer, args.reader_max_seq_length, is_training=True)

        # print(reader_batch)
        # example_ids = reader_batch['example_id']
        # print('example_ids', example_ids)
        # examples.update(batch_examples)
        # features.update(batch_features)
        reader_batch = {k: v.to(args.device)
                        for k, v in reader_batch.items() if k != 'example_id'}
        with torch.no_grad():
            # inputs = {'input_ids': reader_batch['input_ids'],
            #           'attention_mask': reader_batch['input_mask'],
            #           'token_type_ids': reader_batch['segment_ids']}
            inputs = {'input_ids':       reader_batch['input_ids'],
                      'attention_mask':  reader_batch['input_mask'],
                      'token_type_ids':  reader_batch['segment_ids'],
                      'start_positions': reader_batch['start_position'],
                      'end_positions':   reader_batch['end_position'],
                      'retrieval_label': reader_batch['retrieval_label']}
            outputs = model(**inputs)

        loss, qa_loss, rerank_loss = outputs[:3]
        print(f"{loss}, {qa_loss}, {rerank_loss}")
        return 1
    #     retriever_probs = retriever_probs.reshape(-1).tolist()
    #     for i, example_id in enumerate(example_ids):
    #         result = RawResult(unique_id=example_id,
    #                            start_logits=to_list(outputs[0][i]),
    #                            end_logits=to_list(outputs[1][i]),
    #                            retrieval_logits=to_list(outputs[2][i]), 
    #                            retriever_prob=retriever_probs[i])
    #         all_results.append(result)
    # evalTime = timeit.default_timer() - start_time
    # logger.info("  Evaluation done in total %f secs (%f sec per example)",
    #             evalTime, evalTime / len(eval_dataset))

    # output_prediction_file = os.path.join(
    #     predict_dir, "instance_predictions_{}.json".format(prefix))
    # output_nbest_file = os.path.join(
    #     predict_dir, "instance_nbest_predictions_{}.json".format(prefix))
    # output_final_prediction_file = os.path.join(
    #     predict_dir, "final_predictions_{}.json".format(prefix))
    # if args.version_2_with_negative:
    #     output_null_log_odds_file = os.path.join(
    #         predict_dir, "instance_null_odds_{}.json".format(prefix))
    # else:
    #     output_null_log_odds_file = None

    # all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
    #                                     args.max_answer_length, args.do_lower_case, output_prediction_file,
    #                                     output_nbest_file, output_null_log_odds_file, args.verbose_logging,
    #                                     args.version_2_with_negative, args.null_score_diff_threshold)


    # write_final_predictions(all_predictions, output_final_prediction_file, 
    #                         use_rerank_prob=args.use_rerank_prob, use_retriever_prob=True)

    # eval_metrics = quac_eval(
    #     args.orig_dev_file, output_final_prediction_file)

    # with open(args.qrels) as handle:
    #     qrels = json.load(handle)
    # pytrec_eval_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'recall'})
    # rerank_metrics = get_retrieval_metrics(
    #     pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True)
    # logger.info(f"mrr: {rerank_metrics['rerank_mrr']}, recall_5: {rerank_metrics['rerank_recall']}")
    # eval_metrics.update(rerank_metrics)

    # metrics_file = os.path.join(
    #     predict_dir, "metrics_{}.json".format(prefix))

    # with open(metrics_file, 'w') as fout:
    #     json.dump(eval_metrics, fout)

    # return eval_metrics
    return 1

def gen_retrieve_result(args):
    outputs = {}
    with open(args.retrieve_result_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["query_id"]
            doc = obj["doc"]
            pid = obj["doc_id"]
            retrieval_score = obj["retrieval_score"]
            if qid not in outputs.keys():
                outputs[qid] = {"docs": [], "pids": [], "retrieval_scores": []}
            outputs[qid]["docs"].append(doc)
            outputs[qid]["pids"].append(pid)
            outputs[qid]["retrieval_scores"].append(retrieval_score)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/train.txt',
                        type=str, required=False,
                        help="open retrieval quac json for training. ")
    parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/dev.txt',
                        type=str, required=False,
                        help="open retrieval quac json for predictions.")
    parser.add_argument("--test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/test.txt',
                        type=str, required=False,
                        help="open retrieval quac json for predictions.")
    parser.add_argument("--orig_dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/dev.txt',
                        type=str, required=False,
                        help="open retrieval quac json for predictions.")
    parser.add_argument("--orig_test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/test.txt',
                        type=str, required=False,
                        help="original quac json for evaluation.")
    parser.add_argument("--qrels", default='/mnt/scratch/chenqu/orconvqa/v5/retrieval/qrels.txt', type=str, required=False,
                        help="qrels to evaluate open retrieval")
    parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
                        help="all blocks text")
    parser.add_argument("--passage_reps_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_reps.pkl',
                        type=str, required=False, help="passage representations")
    parser.add_argument("--passage_ids_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_ids.pkl',
                        type=str, required=False, help="passage ids")
    parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/orconvqa_output/release_test', type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--load_small", default=True, type=str2bool, required=False,
                        help="whether to load just a small portion of data during development")
    parser.add_argument("--num_workers", default=2, type=int, required=False,
                        help="number of workers for dataloader")

    parser.add_argument("--global_mode", default=True, type=str2bool, required=False,
                        help="maxmize the prob of the true answer given all passages")
    parser.add_argument("--history_num", default=1, type=int, required=False,
                        help="number of history turns to use")
    parser.add_argument("--prepend_history_questions", default=True, type=str2bool, required=False,
                        help="whether to prepend history questions to the current question")
    parser.add_argument("--prepend_history_answers", default=False, type=str2bool, required=False,
                        help="whether to prepend history answers to the current question")

    parser.add_argument("--do_train", default=True, type=str2bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=str2bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, type=str2bool,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--best_global_step", default=40, type=int, required=False,
                        help="used when only do_test")
    parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True, type=str2bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_portion", default=0.1, type=float,
                        help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                            "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=1,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=20,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", default=False, type=str2bool,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=106,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', default=False, type=str2bool,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")


    parser.add_argument("--given_query", default=True, type=str2bool,
                        help="Whether query is given.")
    parser.add_argument("--given_passage", default=False, type=str2bool,
                        help="Whether passage is given. Passages are not given when jointly train")
    parser.add_argument("--is_pretraining", default=False, type=str2bool,
                        help="Whether is pretraining. We fine tune the query encoder in retriever")
    # parser.add_argument("--only_positive_passage", default=True, type=str2bool,
    #                     help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
    # reader arguments
    parser.add_argument("--reader_config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--reader_model_name_or_path", default='bert-base-uncased', type=str, required=False,
                        help="reader model name")
    parser.add_argument("--reader_model_type", default='bert', type=str, required=False,
                        help="reader model type")
    parser.add_argument("--reader_tokenizer_name", default="bert-base-uncased", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--reader_cache_dir", default="tmp2/test", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--reader_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=384, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument('--version_2_with_negative', default=True, type=str2bool, required=False,
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--reader_max_query_length", default=125, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                            "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=40, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                            "and end predictions are not conditioned on one another.")
    parser.add_argument("--qa_loss_factor", default=1.0, type=float,
                        help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
    parser.add_argument("--retrieval_loss_factor", default=1.0, type=float,
                        help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
    parser.add_argument("--top_k_for_reader", default=5, type=int,
                        help="update the reader with top k passages")
    parser.add_argument("--use_rerank_prob", default=True, type=str2bool,
                        help="include rerank probs in final answer ranking")
    parser.add_argument("--retrieve_result_file", default="", type=str, 
                        help="path to retriever's result file")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
        # torch.cuda.set_device(0)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.reader_model_type = args.reader_model_type.lower()
    reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
    reader_config = reader_config_class.from_pretrained(args.reader_config_name if args.reader_config_name else args.reader_model_name_or_path,
                                                        cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
    reader_config.num_qa_labels = 2
    # this not used for BertForOrconvqaGlobal
    reader_config.num_retrieval_labels = 2
    reader_config.qa_loss_factor = args.qa_loss_factor
    reader_config.retrieval_loss_factor = args.retrieval_loss_factor

    tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                            do_lower_case=args.do_lower_case,
                                                            cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
    model = Pipeline()
    model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
                                                  from_tf=bool(
                                                      '.ckpt' in args.reader_model_name_or_path),
                                                  config=reader_config,
                                                  cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)


    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        DatasetClass = ReaderDataset
        train_dataset = DatasetClass(args.train_file,
                                 args.load_small, args.history_num,
                                 given_query=True,
                                 given_passage=True)
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer)
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        if not os.path.exists(args.retriever_tokenizer_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.retriever_tokenizer_dir)
        if not os.path.exists(args.reader_tokenizer_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.reader_tokenizer_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        final_checkpoint_output_dir = os.path.join(
            args.output_dir, 'checkpoint-{}'.format(global_step))
        final_reader_model_dir = os.path.join(
            final_checkpoint_output_dir, 'reader')
        if not os.path.exists(final_checkpoint_output_dir):
            os.makedirs(final_checkpoint_output_dir)
        if not os.path.exists(final_reader_model_dir):
            os.makedirs(final_reader_model_dir)
        
        model.save_pretrained(final_reader_model_dir)
        tokenizer.save_pretrained(args.reader_tokenizer_dir)

        torch.save(args, os.path.join(
            final_checkpoint_output_dir, 'training_args.bin'))

    if args.do_eval:
        result = evaluate(args, model, tokenizer)
        logger.info(f"f1 score: {result['f1']}")
if __name__ == "__main__":
    main()
