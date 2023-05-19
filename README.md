# ConvADR-QA: Open-Domain Conversational Question Answering with Historical Answers

- [AACL-IJCNLP 2022 Findings Paper](https://aclanthology.org/2022.findings-aacl.30/)
- Our code is based on [ConvDR](https://github.com/thunlp/ConvDR) and [ORConvQA](https://github.com/prdwb/orconvqa-release).

## Framework

<img width="790" alt="image" src="https://user-images.githubusercontent.com/2268109/202982724-7eadbe15-7861-415e-83d0-322c12f20881.png">


## Installation
* It is recommended to create a conda environment for the project with `conda create -n conadr_qa python=3.8`
* Install dependencies
```bash
# Install torch (please check your CUDA version)
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu102

git clone https://github.com/MiuLab/ConvADR-QA.git
cd ConvADR-QA 
pip install -r requirements.txt
```

## Data
- Download OR-QuAC data
```
bash scripts/download.sh
```
- Preprocessing
```
bash scripts/preprocessing.sh
```
- Document Embedding Geneation
```
bash scripts/gen_embed.sh
```
Note that this step could take a lot of memory and time.


## Model Training

### Training
- To use ranking loss, we first need to find negative documents using manual queries:
```
bash scripts/run_train.sh find_neg
```
- After the retrieval finishes, we can select negative documents using the following script:
```
bash scripts/run_train.sh gen_rank
```
- Model training:
```
bash scripts/run_train.sh train
```
### Inference
```
bash scripts/run_inference.sh
```
## Citation
```
@inproceedings{fang2022open,
  title={Open-Domain Conversational Question Answering with Historical Answers},
  author={Fang, Hung-Chieh and Hung, Kuo-Han and Huang, Chen-Wei and Chen, Yun-Nung},
  booktitle={Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022},
  pages={319--326},
  year={2022}
}
```
