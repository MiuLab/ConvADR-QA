import json
import argparse



def main(args):
    with open(args.data_path, 'r') as f, open(args.output_path, 'w') as g:
        for in_line in f.readlines():
            line = json.loads(in_line)
            query = line["input"][-1]
            doc_pos = line["doc_pos"]
            for doc_neg in line["doc_negs"]:
                out_obj = {
                    "query": query,
                    "doc_pos": doc_pos,
                    "doc_neg": doc_neg
                }
                out_line = json.dumps(out_obj) + "\n"
                g.write(out_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    main(args)