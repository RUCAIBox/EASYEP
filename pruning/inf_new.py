import os
import json
from argparse import ArgumentParser
from typing import List
from datasets import load_from_disk
import torch
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model
class Hook_gate():
    def __init__(self):
        self.topk_idxs = []
        self.topk_weights = []
        self.topk_norms = []
        
    def hook_fn(self, module, input, output):
        self.topk_idxs.append(output[1].cpu())
        self.topk_weights.append(output[0].cpu())
        self.topk_norms.append(output[2].cpu())

class Hook_gate2():
    def __init__(self):
        self.simi1 = []
        self.simi2 = []
        self.simi3 = []
        
    def hook_fn(self, module, input, output):
        self.simi1.append(output[0].cpu())
        self.simi2.append(output[1].cpu())
        self.simi3.append(output[2].cpu())

from pruning.model_new import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

@torch.inference_mode()
def inference(model, dataset,rank, output):
    count = 0
    for data in tqdm(dataset):

        count+=1

        Hook = Hook_gate()
        hooks = []
        Hook2 = Hook_gate2()
        hooks2 = []
        print(len(model.layers))
        for layer in range(3,61):
            # 1, 59 for deepseek-v2
            hooks.append(model.layers[layer].ffn.tmp.register_forward_hook(Hook.hook_fn))
            hooks2.append(model.layers[layer].tmp.register_forward_hook(Hook2.hook_fn))
        
        tokens = torch.tensor([data['input_ids']],device="cuda")
        print(tokens.shape)
        with torch.no_grad():
            logits = model.forward(tokens[:, :], 0)

        for hook in hooks:
            hook.remove()
        for hook in hooks2:
            hook.remove()
        if rank == 0:
            with open(output,'a') as fp:
                idxs = []
                weights = []
                norms = []
                simi1 = []
                simi2 = []
                simi3 = []
                for layer in range(len(Hook.topk_idxs)):
                    idxs.append(Hook.topk_idxs[layer].tolist())
                    weights.append(Hook.topk_weights[layer].tolist())
                    norms.append(Hook.topk_norms[layer].tolist())
                for layer in range(len(Hook2.simi1)):
                    simi1.append(Hook2.simi1[layer].tolist())
                    simi2.append(Hook2.simi2[layer].tolist())
                    simi3.append(Hook2.simi3[layer].tolist())
                fp.write(json.dumps({"idxs":idxs, "weights":weights, "norms": norms, 'simibr': simi1, 'simisf': simi2, 'simirf': simi3})+'\n')
 

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    
    dataset = load_from_disk(input_file)
    inference(model, dataset,rank, args.output)


    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    main(args.ckpt_path, args.config, args.input_file, args.max_new_tokens, args.temperature)
