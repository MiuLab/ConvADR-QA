import argparse
import csv
import logging
import json
from model.models import MSMarcoConfigDict
import os
import pickle 
import time
import copy
import faiss
import torch
import numpy as np
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model, load_collection
from reader.model import BertForOrconvqaGlobal
from reader.utils import gen_reader_features, RawResult, write_predictions, write_final_predictions
from reader.scorer import quac_eval
from transformers import BertConfig, BertTokenizer
import pytrec_eval
from scipy import special
import subprocess
import os
from threading import Timer

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer)
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_gpu_memory():
    DEVICE = int(os.getenv('CUDA_VISIBLE_DEVICES'))
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used = output_to_list(subprocess.check_output(COMMAND.split()))[DEVICE+1]
    print(f"Memory Used: {memory_used}")

def EvalDevQuery(query_embedding2id,
                 merged_D,
                 dev_query_positive_id,
                 I_nearest_neighbor,
                 topN,
                 output_file,
                 output_trec_file,
                 offset2pid,
                 raw_data_dir,
                 output_query_type,
                 raw_sequences=None):
    prediction = {}

    qids_to_ranked_candidate_passages = {}
    qids_to_ranked_candidate_passages_ori = {}
    qids_to_raw_sequences = {}
    for query_idx in range(len(I_nearest_neighbor)):
        seen_pid = set()
        inputs = raw_sequences[query_idx]
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        top_ann_score = merged_D[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp
            qids_to_ranked_candidate_passages_ori[query_id] = tmp_ori
        qids_to_raw_sequences[query_id] = inputs

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid,
                                                                     score)
                qids_to_ranked_candidate_passages_ori[query_id][
                    rank] = pred_pid

                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    logger.info("Reading queries and passages...")
    queries = {}
    with open(
            os.path.join(raw_data_dir,
                         "queries." + output_query_type + ".tsv"), "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query
    collection = os.path.join(raw_data_dir, "collection.jsonl")
    if not os.path.exists(collection):
        collection = os.path.join(raw_data_dir, "collection.tsv")
        if not os.path.exists(collection):
            raise FileNotFoundError(
                "Neither collection.tsv nor collection.jsonl found in {}".
                format(raw_data_dir))
    all_passages = load_collection(collection)

    # Write to file
    with open(output_file, "w") as f, open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            ori_qid = qid
            query_text = queries[ori_qid]
            sequences = qids_to_raw_sequences[ori_qid]
            for i in range(topN):
                pid, score = passages[i]
                ori_pid = pid
                passage_text = all_passages[ori_pid]
                label = 0 if qid not in dev_query_positive_id else (
                    dev_query_positive_id[qid][ori_pid]
                    if ori_pid in dev_query_positive_id[qid] else 0)
                f.write(
                    json.dumps({
                        "query": query_text,
                        "doc": passage_text,
                        "label": label,
                        "query_id": str(ori_qid),
                        "doc_id": str(ori_pid),
                        "retrieval_score": score,
                        "input": sequences
                    }) + "\n")
                g.write(
                    str(ori_qid) + " Q0 " + str(ori_pid) + " " + str(i + 1) +
                    " " + str(-i - 1 + 200) + " ance\n")


def pad_input_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask

def prepare_dataset(args):
    with open(args.eval_file, encoding="utf-8") as f:
        inputs = []
        targets = []
        qids = []
        convids = []
        newConv = []
        for line in f:
            record = json.loads(line)
            input_sents = record['input'] # history quries + current query
            target_sent = record['answer']['text'] # manually rewritten query 
            auto_sent = record.get('output', "no")
            raw_sent = record["input"][-1] # current query
            # responses = record["manual_response"] # answer
            responses = record[
                "manual_response"] if args.query == "man_can" else (
                    record["automatic_response"]
                    if args.query == "auto_can" else [])
            topic_number = record.get('topic_number', None) 
            query_number = record.get('query_number', None) 
            qid = str(topic_number) + "_" + str(
                query_number) if topic_number != None else str(
                    record["qid"])
            convid = qid.split("#")[0]
            inputs.append(input_sents)
            targets.append(target_sent)
            qids.append(qid)
            if len(convids) == 0 or convid != convids[-1]:
                newConv.append(True)
            else:
                newConv.append(False)
            convids.append(convid)

        return inputs, targets, qids, convids, newConv


def search_one_by_one(ann_data_dir, gpu_index, query_embedding, topN, passage_embedding, passage_embedding2id):
    ts = time.time()
    D, I = gpu_index.search(query_embedding, topN)
    te = time.time()
    elapsed_time = te - ts
    # print({
    #     "total": elapsed_time,
    #     "data": query_embedding.shape[0],
    #     "per_query": elapsed_time / query_embedding.shape[0]
    # })
    I = passage_embedding2id[I]
    return D, I
    

def evaluate(args, retriever_model, retriever_tokenizer, 
            reader_model, reader_tokenizer, logger, index, offset2pid):
    with open(os.path.join("tmp2/datasets/or-quac/embeddings", "passage__emb_p__data_obj_0" + ".pb"), 'rb') as handle:
        passage_embedding_0 = pickle.load(handle)
    with open(os.path.join("tmp2/datasets/or-quac/embeddings", "passage__embid_p__data_obj_0" + ".pb"), 'rb') as handle:
        passage_embedding2id_0 = pickle.load(handle)
    with open(os.path.join("tmp2/datasets/or-quac/embeddings", "passage__emb_p__data_obj_1" + ".pb"), 'rb') as handle:
        passage_embedding_1 = pickle.load(handle)
    with open(os.path.join("tmp2/datasets/or-quac/embeddings", "passage__embid_p__data_obj_1" + ".pb"), 'rb') as handle:
        passage_embedding2id_1 = pickle.load(handle)

    passage_embedding = np.concatenate((passage_embedding_0, passage_embedding_1), axis=0)
    passage_embedding2id = np.concatenate((passage_embedding2id_0, passage_embedding2id_1), axis=0)
    print(passage_embedding.shape)
    print(passage_embedding2id.shape)
    index.add(passage_embedding)
    collection = os.path.join(args.raw_data_dir, "collection.jsonl")
    all_passages = load_collection(collection)
    print(type(all_passages))
    # get_gpu_memory()
    logger.info("***** Running evaluation *****")
    inputs, targets, qids, convids, newConv = prepare_dataset(args)
    answers = []
    gold_answers = []
    embedding = []
    embedding2id = []
    raw_sequences = []
    retriever_result = {"qids": [], "pids": []}
    qa_result = {}
    # output_prediction_file = os.path.join(
    #     predict_dir, "instance_predictions.json")
    # output_nbest_file = os.path.join(
    #     predict_dir, "instance_nbest_predictions.json")
    # output_final_prediction_file = os.path.join(
    #     predict_dir, "final_predictions.json")
    # if args.version_2_with_negative:
    #     output_null_log_odds_file = os.path.join(
    #         predict_dir, "instance_null_odds.json")
    # else:
    #     output_null_log_odds_file = None
    # j = 0
    cnt = 0
    for inp, target, qid, convid, isNew in tqdm(zip(inputs, targets, qids, convids, newConv), total=len(inputs)):
        assert(len(retriever_result["qids"]) == len(retriever_result["pids"]))
        retriever_result["qids"].append(qid)
        if isNew:
            answers = []
            gold_answers = []
        concat_ids = []
        concat_id_mask = []
        concat_ids.append(retriever_tokenizer.cls_token_id)
        assert (len(inp)-1) == len(answers), [len(inp), len(answers)]
        assert (len(inp)-1) == len(gold_answers), [len(inp), len(gold_answers)]
        # print(f"input: {inp}")
        # print(f"answers: {answers}")
        # print(f"gold_answers: {gold_answers}")
        if args.mode in ["ans", "no_ans"]:
            for q, a in zip(inp, answers):
                concat_ids.extend(
                    retriever_tokenizer.convert_tokens_to_ids(
                        retriever_tokenizer.tokenize(q)
                    ))
                concat_ids.append(retriever_tokenizer.sep_token_id)
                if args.mode == "ans":
                    if a not in ["CANNOTANSWER", "NOTRECOVERED", "empty"]:
                        concat_ids.extend(
                            retriever_tokenizer.convert_tokens_to_ids(
                                retriever_tokenizer.tokenize(a)))
                        concat_ids.append(retriever_tokenizer.sep_token_id)
        else: # args.mode == "gold"
            for q, a in zip(inp, gold_answers):
                concat_ids.extend(
                    retriever_tokenizer.convert_tokens_to_ids(
                        retriever_tokenizer.tokenize(q)
                    ))
                concat_ids.append(retriever_tokenizer.sep_token_id)
                if a not in ["CANNOTANSWER", "NOTRECOVERED", "empty"]:
                    concat_ids.extend(
                        retriever_tokenizer.convert_tokens_to_ids(
                            retriever_tokenizer.tokenize(a)))
                    concat_ids.append(retriever_tokenizer.sep_token_id)

        concat_ids.extend(
            retriever_tokenizer.convert_tokens_to_ids(
                            retriever_tokenizer.tokenize(inp[-1])))

        concat_ids.append(retriever_tokenizer.sep_token_id)
        concat_ids, concat_id_mask = pad_input_ids_with_mask(
            concat_ids, args.max_concat_length)
    
        assert len(concat_ids) == args.max_concat_length
        ids, id_mask = (
            ele.to(args.device)
            for ele in [torch.tensor([concat_ids]), torch.tensor([concat_id_mask])])

       
        retriever_model.eval()
        with torch.no_grad():
            embs = retriever_model(ids, id_mask)
        
        embedding.append(embs)
        embedding2id.append(qid)
        query_embedding = embs.cpu().detach().numpy()
        D, I = search_one_by_one(args.ann_data_dir, index, query_embedding, args.top_n, passage_embedding, passage_embedding2id)
        # I.shape: (1, top_n)
        I = I.squeeze()
        pids = [offset2pid[i] for i in I]
        retriever_result["pids"].append(pids)
        passages_for_reader = np.array([[all_passages[pid] for pid in pids]])
        passages_for_reader = passages_for_reader[:, :args.top_k_for_reader]
        scores = D[:, :args.top_k_for_reader]
        retriever_probs = special.softmax(scores, axis=1)
        pids_for_reader = np.array(pids).reshape(1, -1)
        pids_for_reader = pids_for_reader[:, :args.top_k_for_reader]
        # print(passages_for_reader)
        # print(pids_for_reader)
        
        question_texts = ["[CLS]" + " [SEP] ".join(inp)]
        # q_texts = "[CLS] "
        # for q, a in zip(inp, answers):
        #     q_texts += f"{q} [SEP] {a} [SEP] "
        answer_texts = [target]
        qids = [qid]
        answer_starts = [-1]
        labels_for_reader = np.zeros((1, args.top_k_for_reader))
        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                    answer_starts, pids_for_reader,
                                                                    passages_for_reader, labels_for_reader,
                                                                    reader_tokenizer,
                                                                    args.reader_max_seq_length,
                                                                    is_training=False)
        example_ids = reader_batch['example_id']
        reader_batch = {k: v.to(args.device) for k, v in reader_batch.items() if k != 'example_id'}

        
        examples = batch_examples
        features = batch_features
        reader_model.to(args.device)
        with torch.no_grad():
            batch = {
                'input_ids': reader_batch['input_ids'],
                'attention_mask': reader_batch['input_mask'],
                'token_type_ids': reader_batch['segment_ids']
            }
            outputs = reader_model(**batch)
        
        retriever_probs = retriever_probs.reshape(-1).tolist()
        all_results = []
        for i, example_id in enumerate(example_ids):
            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=to_list(outputs[2][i]), 
                               retriever_prob=retriever_probs[i])
            all_results.append(result)

        all_predictions, score_diff = write_predictions(examples, features, all_results, args.n_best_size,
                                        args.max_answer_length, args.do_lower_case, args.verbose_logging,
                                        args.version_2_with_negative, args.null_score_diff_threshold)

        # print(all_predictions)
        final_predictions = write_final_predictions(all_predictions, use_rerank_prob=args.use_rerank_prob, 
                            use_retriever_prob=args.use_retriever_prob)
        # print(final_predictions)

        prediction = list(final_predictions)[0]
        dialog_id = qid.split("#")[0]
        if dialog_id in qa_result:
            qa_result[dialog_id]['best_span_str'].append(prediction['best_span_str'][0])
            qa_result[dialog_id]['qid'].append(prediction['qid'][0])
            qa_result[dialog_id]['yesno'].append(prediction['yesno'][0])
            qa_result[dialog_id]['followup'].append(prediction['followup'][0])
        else:
            qa_result[dialog_id] = {}
            qa_result[dialog_id]['best_span_str'] = [prediction['best_span_str'][0]]
            qa_result[dialog_id]['qid'] = [prediction['qid'][0]]
            qa_result[dialog_id]['yesno'] = [prediction['yesno'][0]]
            qa_result[dialog_id]['followup'] = [prediction['followup'][0]]
            
        
        if score_diff > args.add_answer_diff_threshold:
            answer = "CANNOTANSWER"
        else:
            answer = prediction['best_span_str'][0]
            cnt += 1
        # print(answer)
        # print(qa_result)
        # print(retriever_result)
        answers.append(answer)
        gold_answers.append(target)
        # j += 1
        # if j == 10:
        #     return
    # write result to file
    with open(args.output_file, "w") as f:
        for pred in qa_result.values():
            f.write(json.dumps(pred) + '\n')

    with open(args.output_trec_file, "w") as f:
        for qid, pids in zip(retriever_result["qids"], retriever_result["pids"]):
            for i, pid in enumerate(pids):
                f.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) +
                        " " + str(-i - 1 + 200) + " ance\n")
    eval_metrics = quac_eval(
        args.orig_test_file, args.output_file)
    print(cnt)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model_path", type=str, help="The model checkpoint.")
    parser.add_argument("--reader_model_path", type=str, help="The model checkpoint.")
    parser.add_argument("--eval_file",
                        type=str,
                        help="The evaluation dataset.")
    parser.add_argument(
        "--max_concat_length",
        default=256,
        type=int,
        help="Max input concatenated query length after tokenization.")
    parser.add_argument("--max_query_length",
                        default=64,
                        type=int,
                        help="Max input query length after tokenization."
                        "This option is for single query input.")
    parser.add_argument("--cross_validate",
                        action='store_true',
                        help="Set when doing cross validation.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=4,
                        type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Avoid using CUDA when available (for pytorch).")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--ann_data_dir",
                        type=str,
                        help="Path to ANCE embeddings.")
    parser.add_argument("--processed_data_dir",
                        type=str,
                        help="Path to tokenized documents.")
    parser.add_argument("--raw_data_dir", type=str, help="Path to dataset.")
    parser.add_argument("--output_file",
                        type=str,
                        help="Output file for OpenMatch reranking.")
    parser.add_argument(
        "--output_trec_file",
        type=str,
        help="TREC-style run file, to be evaluated by the trec_eval tool.")
    parser.add_argument("--orig_test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/test.txt',
            type=str, required=False,
            help="original quac json for evaluation.")
    parser.add_argument(
        "--query",
        type=str,
        default="no_res",
        choices=["no_res", "man_can", "auto_can", "target", "output", "raw"],
        help="Input query format.")
    parser.add_argument("--output_query_type",
                        type=str,
                        help="Query to be written in the OpenMatch file.")
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold to evaluate on; set to -1 to evaluate all folds.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MSMarcoConfigDict.keys()),
    )
    parser.add_argument("--top_n",
                        default=100,
                        type=int,
                        help="Number of retrieved documents for each query.")
    parser.add_argument(
        "--add_answer",
        action='store_true',
        help="whether to add answer in query embedding"
    )
    parser.add_argument("--qrels", default='/mnt/scratch/chenqu/orconvqa/v5/retrieval/qrels.txt', type=str, required=False,
                    help="qrels to evaluate open retrieval")

    parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
                        help="all blocks text")
    parser.add_argument("--passage_reps_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_reps.pkl',
                        type=str, required=False, help="passage representations")
    parser.add_argument("--passage_ids_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_ids.pkl',
                        type=str, required=False, help="passage ids")
    parser.add_argument("--top_k_for_reader", default=10, type=int,
                    help="update the reader with top k passages")
    parser.add_argument("--top_k_for_retriever", default=100, type=int,
                    help="retrieve top k passages for a query, these passages will be used to update the query encoder")
    parser.add_argument("--use_gpu",
    action='store_true',
    help="Whether to use GPU for Faiss.")
    parser.add_argument("--reader_tokenizer_name", default="bert-base-uncased", type=str)
    parser.add_argument("--reader_max_seq_length", default=512, type=int)
    parser.add_argument("--reader_max_query_length", default=125, type=int)
    parser.add_argument("--do_lower_case", default=True, type=str2bool)
    parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=40, type=int,
                help="The maximum length of an answer that can be generated. This is needed because the start "
                        "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--version_2_with_negative', default=True, type=str2bool, required=False,
                    help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--add_answer_diff_threshold', type=float, default=0.0)
    parser.add_argument("--use_rerank_prob", default=True, type=str2bool,
                    help="include rerank probs in final answer ranking")
    parser.add_argument("--use_retriever_prob", default=False, type=str2bool,
                    help="include albert retriever probs in final answer ranking")
    parser.add_argument("--mode", type=str, default="ans", choices=["no_ans", "ans", "gold"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    ngpu = faiss.get_num_gpus()
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)



    set_seed(args)

    with open(os.path.join(args.processed_data_dir, "offset2pid.pickle"),
              "rb") as f:
        offset2pid = pickle.load(f)
    logger.info("Building index")
    cpu_index = faiss.IndexFlatIP(768) # inner product
    index = None
    if args.use_gpu:
        print("use gpu")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        print("use cpu")
        index = cpu_index

    dev_query_positive_id = {}
    if args.qrels is not None:
        with open(args.qrels, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                topicid = str(topicid)
                docid = int(docid)
                rel = int(rel)
                if topicid not in dev_query_positive_id:
                    if rel > 0:
                        dev_query_positive_id[topicid] = {}
                        dev_query_positive_id[topicid][docid] = rel
                else:
                    dev_query_positive_id[topicid][docid] = rel



    retriever_config, retriever_tokenizer, retriever_model = load_model(args, args.retriever_model_path)
    if args.max_concat_length <= 0:
        args.max_concat_length = retriever_tokenizer.max_len_single_sentence
    args.max_concat_length = min(args.max_concat_length,
                                     retriever_tokenizer.max_len_single_sentence)

    reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
    reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name,
                                                          do_lower_case=args.do_lower_case)
    reader_model = reader_model_class.from_pretrained(args.reader_model_path)
    logger.info("Training/evaluation parameters %s", args)
    evaluate(args, retriever_model, retriever_tokenizer, reader_model, reader_tokenizer, logger, index, offset2pid)    

    # eval
    

if __name__ == "__main__":
    main()
