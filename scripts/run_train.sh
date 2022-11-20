cd ..

case $1 in
    find_neg ) 
        python drivers/run_convdr_inference.py  \
        --model_path=checkpoints/ad-hoc-ance-orquac.cp  \
        --eval_file=datasets/or-quac/train.jsonl  \
        --query=target  --per_gpu_eval_batch_size=8  \
        --ann_data_dir=datasets/or-quac/embeddings  \
        --qrels=datasets/or-quac/qrels.tsv  \
        --processed_data_dir=datasets/or-quac/tokenized  \
        --raw_data_dir=datasets/or-quac   --output_file=results/or-quac/manual_ance_train.jsonl \
        --output_trec_file=results/or-quac/manual_ance_train.trec  \
        --model_type=dpr  --output_query_type=train.manual  --use_gpu
        ;;
    gen_rank )  
        python data/gen_ranking_data.py  \
        --train=datasets/or-quac/train.jsonl  \
        --run=results/or-quac/manual_ance_train.trec  \
        --output=datasets/or-quac/train.rank.jsonl  \
        --qrels=datasets/or-quac/qrels.tsv  \
        --collection=datasets/or-quac/collection.jsonl
        ;;
    train )
        python drivers/run_convdr_train.py  \
        --output_dir=tmp2/checkpoints/convdr-multi-answer-orquac.cp \
        --model_name_or_path=tmp2/checkpoints/ad-hoc-ance-orquac.cp  \
        --train_file=tmp2/datasets/or-quac/train.rank.jsonl  --query=no_res \
        --per_gpu_train_batch_size=4 --learning_rate=1e-5 \
        --log_dir=logs/convdr_multi_orquac  --num_train_epochs=3  \
        --model_type=dpr  --log_steps=100  --ranking_task --add_answer \
        --gradient_accumulation_steps=10
        ;;
esac




