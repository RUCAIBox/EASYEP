import json
import torch
from tqdm import tqdm
import random
import argparse
import os
def main():
    parser = argparse.ArgumentParser(description='Process JSONL data and save weight information.')
    parser.add_argument('--expert_info_dir', type=str, default = 'expert_statistics/expert_information/', help='Path to the directory of expert information')
    parser.add_argument('--target_number', type=int, default=128, help='Remaining expert number in each layer.')
    parser.add_argument('--expert_mask', type=str, required=True, help='Path to the mask file of experts.')
    args = parser.parse_args()
    score_list = []
    for file_path in os.listdir(args.expert_info_dir):
        score = torch.load(os.path.join(args.expert_info_dir,file_path))


        score = score/torch.sum(score, dim=-1, keepdim=True)
        score_list.append(score)
    tmp =  torch.zeros_like(w)
    for w in score_list:
        tmp = tmp+w
    topk_experts = torch.topk(tmp,k=128, dim=-1)[1]
    mask = torch.zeros_like(tmp).float()
    print(mask.shape)
    print(topk_experts.shape)
    mask.scatter_(1, topk_experts, 1)
    with open(args.expert_mask,'w') as fw:
        import json
        fw.write(json.dumps(mask.float().tolist()))

