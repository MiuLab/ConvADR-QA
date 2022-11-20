import json


RESULT_PATH = "../tmp2/results/predictions_.json"
DATA_PATH = "../tmp2/datasets/or-quac/dev.rank.jsonl"

with open(RESULT_PATH, "r") as f, open(DATA_PATH, "r") as g:
    k = 0
    num_ex = 3
    predictions = json.load(f)
    for i, line in enumerate(g):
        obj = json.loads(line)
        target = obj["target"]
        doc_pos = obj["doc_pos"]
        qid = obj["qid"]
        answer = obj["answer"]
        inp = obj["input"]
        prediction = predictions[qid]
        if len(inp) == 1:
            print(f"Questions {i+1}: {inp}")
            print()
            print(f"Manual Question: {target}")
            print()
            print(f"Answer: {answer}")
            print()
            print(f"prediction: {prediction}")
            print()
            print(f"doc: {doc_pos}")
            print("================")
            k += 1
        if k == num_ex:
            break
        
