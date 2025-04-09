import os
import argparse
import numpy as np
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

import json
import multiprocessing as mp

from tqdm import tqdm, trange
from datasets import load_from_disk, load_dataset
from evaluator.MATH_evaluator_list import MATHEvaluator
# from evaluator.MC_evaluator_list import MCEvaluator


def check(evaluator, pred_ans, real_ans):
    print(len(pred_ans), len(real_ans))
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness


import random

random.seed(42)

name2path = {
    "MATH_OAI": "./dataset/MATH_OAI.jsonl",
    "MATH30": "./dataset/MATH30.jsonl",
    "AIME24": "./dataset/AIME24.jsonl",
    "AIME2023": "./dataset/AIME2023.jsonl",
    "AIME2022": "./dataset/AIME2022.jsonl",
    "AIME25": "./dataset/AIME25.jsonl",
    "hmmt_feb_2025": "./dataset/hmmt_feb_2025.jsonl",
    "OMNI": "./dataset/omni_math_num_500.jsonl",
    "LiveAOPSbench": "./dataset/liveaopsbench-2024-8-2024-12-num.jsonl",
}

name2eval = {
    "MATH_OAI": MATHEvaluator(),
    "MATH30": MATHEvaluator(),
    "OMNI": MATHEvaluator(),
    "AIME25": MATHEvaluator(),
    "AIME2023": MATHEvaluator(),
    "AIME2022": MATHEvaluator(),
    "AIMEzh": MATHEvaluator(),
    "hmmt_feb_2025": MATHEvaluator(),
    "LiveAOPSbench": MATHEvaluator(),
    "AIME24": MATHEvaluator(),
}

import sglang as sgl
set_default_backend(RuntimeEndpoint("http://0.0.0.0:60000"))
@function
def single_turn(s, question):
    s += user(question)
    stop_tokens = ["<｜end▁of▁sentence｜>"]

    s += assistant(gen("answer", temperature=0.6, top_p=0.95, max_tokens=32000, stop=stop_tokens))

def main(args, lines, gpu_idx):
    import os

    import torch

    import time
    tgt_path = os.path.join(args.target_path, "{}-L_{}-{}-part_{}-0123-{}.jsonl".format(args.data_name, args.max_tokens, args.decode, gpu_idx, args.system_prompt))
    fout = open(tgt_path, "w")

    bs = 1024
    num_data = len(lines)
    for st in trange(0, num_data, bs):
        tmp_lines = lines[st : st + bs]
        prompts = [{'question':data["input"]} for data in tmp_lines]
        responses = single_turn.run_batch(prompts,progress_bar=True,
            num_threads = 1000
        )
        # model.generate(prompts, sampling_params)

        for response, data in zip(responses, tmp_lines):
            new_data = {
                "input": data["input"],
                "output": data["output"],
                "prediction": [],
            }
            new_data["prediction"].append(
                {
                    "solution": response['answer'],
                }
            )
            print(response['answer'])
            fout.write(json.dumps(new_data) + "\n")
            fout.flush()
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_tokens", default=10000, type=int)
    parser.add_argument("--paralle_size", default=1, type=int)
    parser.add_argument("--part_num", default=1, type=int)
    parser.add_argument("--year", default=None, type=str)
    parser.add_argument("--number", default=1, type=int)
    parser.add_argument("--decode", default="sample", type=str)
    parser.add_argument("--system_prompt", default="qwen", type=str)
    args = parser.parse_args()

    os.makedirs(args.target_path, exist_ok=True)

    src_path = name2path[args.data_name]
    with open(src_path, "r") as fin:
        raw_dataset = fin.readlines()
        raw_dataset = [json.loads(d) for d in raw_dataset]
    new_dataset = []
    for i in range(args.number):
        new_dataset.extend(raw_dataset)
    dataset = []

    for data in new_dataset:
        dataset.append({"input": data["problem"], "output": data["solution"]})

    # random.shuffle(dataset)
    slice_idx = np.linspace(0, len(dataset), args.part_num + 1).astype('int')
    p = mp.Pool(args.part_num)
    for start_id in range(args.part_num):
        start, end = slice_idx[start_id], slice_idx[start_id + 1]
        new_lines = dataset[start:end]
        print("start process %s" % start_id)
        p.apply_async(main, args=(args, new_lines, start_id))
    p.close()
    p.join()
    print("All of the child processes over!")

    tgt_path = os.path.join(args.target_path, "{}-L_{}-{}-0123.jsonl".format(args.data_name, args.max_tokens, args.decode))
    fout = open(tgt_path, 'w')
    results = []
    for gpu_idx in range(args.part_num):
        src_path = os.path.join(args.target_path, "{}-L_{}-{}-part_{}-0123-{}.jsonl".format(args.data_name, args.max_tokens, args.decode, gpu_idx, args.system_prompt))
        # if (os.path.exists(src_path) == False):
        #     continue
        with open(src_path, 'r') as fin:
            results = results + fin.readlines()
    results = [json.loads(r) for r in results]

    pred_ans_list, real_ans_list = [], []
    for r in results:
        for preds in r['prediction']:
            pred = preds['solution']
            pred_ans_list.append(pred)
            real_ans_list.append(r["output"])
    evaluator = name2eval[args.data_name]
    correctness = check(evaluator, pred_ans_list, real_ans_list)
    pred2corr = {}
    for pred, c in zip(pred_ans_list, correctness):
        pred2corr[pred] = c

    total_correct, total_problem = 0, 0
    for r in results:
        for pred in r['prediction']:
            pred['correctness'] = pred2corr[pred['solution']]
            if (pred['correctness'] == True):
                total_correct = total_correct + 1
            total_problem = total_problem + 1
        fout.write(json.dumps(r) + '\n')
    results = {"results": round(total_correct / total_problem * 100, 2)}
    fout.write(json.dumps(results) + "\n")
    print(
        "{}: {}% ( {} / {} )".format(
            args.data_name,
            round(total_correct / total_problem * 100, 2),
            total_correct,
            total_problem,
        )
    )
