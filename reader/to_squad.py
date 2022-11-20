import json
from tqdm.auto import tqdm
import argparse
import os

SPLITS = ['train', 'dev']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./tmp2/datasets/or-quac')
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    
    for split in SPLITS:
        origin = os.path.join(args.data_dir, f'{split}.rank.jsonl')
        output = os.path.join(args.data_dir, f'{split}_squad.jsonl')
        with open(origin, 'r') as f, open(output, 'w') as g:
            paragraphs = []
            for line in tqdm(f.readlines()):
                obj = json.loads(line)
                out_obj = {
                    "qas": [
                        {
                            "question": obj["target"],
                            "history_questions": obj["input"],
                            "id": obj["qid"],
                            "answers": [
                                {
                                    "text": obj["answer"]["text"],
                                    "answer_start": obj["answer"]["answer_start"]
                                }
                            ],
                            "is_impossible": obj["answer"]["text"] in ["CANNOTANSWER", "NOTRECOVERED"]
                        }
                    ],
                    "context": obj['doc_pos']
                }
                paragraphs.append(out_obj)
            data = [
                {
                    "title": "don'tcare",
                    "paragraphs": paragraphs
                }
            ]
            out_line = json.dumps({
                "data": data
            }) + '\n'
            g.write(out_line)

if __name__ == "__main__":
    main()


'''
"version": -
"data": [
    {
        "title": -,
        "paragraphs":[
            {
                "qas":[
                    {
                        "question": -
                        "id": -
                        "answers": [
                            {
                                "text": -
                                "answer_start": -
                            },
                            ...
                        ]
                        "is_impossible": -
                    },
                    ...
                ]
                "context": -
            },
            ...
        ],
    },
    ...
]
'''