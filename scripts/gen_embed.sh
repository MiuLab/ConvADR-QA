cd ..

mkdir datasets/or-quac/tokenized
mkdir datasets/or-quac/embeddings

# tokenize documents
python data/tokenizing.py  --collection=datasets/or-quac/collection.tsv  --out_data_dir=datasets/or-quac/tokenized  --model_name_or_path=bert-base-uncased --model_type=dpr

# generate document embeddings (this step takes a lot of memory and time)
python -m torch.distributed.launch --nproc_per_node=$gpu_no python drivers/gen_passage_embeddings.py  --data_dir=datasets/or-quac/tokenized  --checkpoint=checkpoints/ad-hoc-ance-orquac.cp  --output_dir=datasets/or-quac/embeddings  --model_type=dpr