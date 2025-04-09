import json
import torch
from tqdm import tqdm
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process JSONL data and save weight information.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output PT file.')
    parser.add_argument('--target_number', type=int, default=128, help='Remaining expert number in each layer.')
    parser.add_argument('--expert_mask', type=str, required=True, help='Path to the mask file of experts.')
    args = parser.parse_args()

    random.seed(42)
    with open(args.input_file, 'r') as fp:
        lines = fp.readlines()
        data = [line for line in lines]
        random.shuffle(data)
    data = sorted(data, key=len, reverse=True)
    data = [json.loads(d) for d in data[:25]]

    expert_br_list = []
    for sample in tqdm(data):
        expert_br = torch.zeros((58, 256)).float()
        for layer in range(58):
            for token in range(len(sample['idxs'][0])):
                idxs = sample['idxs'][layer][token]
                weight = sample['weights'][layer][token]
                norm = sample['norms'][layer][token]
                simibr = max(1 - sample['simibr'][layer][0][token], 0)

                for i, idx in enumerate(idxs):
                    expert_br[layer, idx] += weight[i] * simibr * norm[i]
        expert_br_list.append(expert_br)

    export_scores = torch.sum(torch.stack(expert_br_list, dim=0), dim=0)
    torch.save(export_scores, args.output_file)
    topk_experts = torch.topk(export_scores, args.target_number, dim=-1)[1]
    mask = torch.zeros_like(export_scores).float()
    mask.scatter_(1, topk_experts, 1)
    with open(args.expert_mask,'w') as fw:
        import json
        fw.write(json.dumps(mask.float().tolist()))


if __name__ == "__main__":
    main()