# ConvADR-QA

Source code for AACL-IJCNLP 2022 paper ["Open-Domain Conversational Question Answering with Historical Answers"](https://arxiv.org/abs/2211.09401)

Our code is based on [ConvDR](https://github.com/thunlp/ConvDR) and [ORConvQA](https://github.com/prdwb/orconvqa-release).

## Installation
* It is recommended to create a conda environment for the project with `conda create -n conadr_qa python=3.8`
* To install dependencies, run:
```bash
# Install torch (please check your CUDA version)
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu102

git clone https://github.com/MiuLab/ConvADR-QA.git
cd ConvADR-QA 
pip install -r requirements.txt
```

## Data Download
To download OR-QuAC dataset, run:
```
bash scripts/download.sh
```

## Data Preprocessing
To preprocess OR-QuAC dataset, run:
```
bash scripts/preprocessing.sh
```
## Generate Document Embedding
To generate embeddings for document, run:
```
bash scripts/gen_embed.sh
```
Note that this step could take a lot of memory and time.
## Training
To use ranking loss, we first need to find negative documents using manual queries:
```
bash scripts/run_train.sh find_neg
```
After the retrieval finishes, we can select negative documents using the following script:
```
bash scripts/run_train.sh gen_rank
```
Now we can start training:
```
bash scripts/run_train.sh train
```
## Inference
Run the following script to get inference results:
```
bash scripts/run_inference.sh
```
## Reference
Please cite the following paper:
```

```
