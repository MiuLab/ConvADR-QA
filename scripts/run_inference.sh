cd ..

top_K=5

python drivers/pipeline_inference.py \
    --retriever_model_path=checkpoints/convdr-multi-orquac.cp \
    --eval_file=datasets/or-quac/test.jsonl  \
    --orig_test_file=datasets/raw/or-quac/test.txt \
    --reader_model_path= \
    --query=no_res  \
    --per_gpu_eval_batch_size=2 \
    --cache_dir=../ann_cache_dir  \
    --ann_data_dir=datasets/or-quac/embeddings  \
    --processed_data_dir=datasets/or-quac/tokenized  \
    --raw_data_dir=datasets/or-quac   \
    --top_k_for_reader=${top_K} \
    --output_file=tmp2/results/or-quac/multi_pipe_${top_K}.jsonl  \
    --output_trec_file=tmp2/results/or-quac/multi_pipe_${top_K}.trec  \
    --model_type=dpr  \
    --output_query_type=test.raw \
    --qrels=datasets/or-quac/qrels.tsv \
    --blocks_path=datasets/raw/or-quac/all_blocks.txt \
    --use_gpu \
    --null_score_diff_threshold -1 \
    --add_answer_diff_threshold -12