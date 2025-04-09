# Table of Contents
<!-- - [1. Preparation](#1-preparation)
- [2. Expert Selection](#2-expert-selection)
- [3. Model Pruning](#3-model-pruning)
- [4. Evaluation](#4-evaluation)
- [5. Next Steps](#5-next-steps)
- [6. Citation](#6-citation) -->

## 1. Preparation

### 1.1 Requirements
```bash
cd EasyEP
conda create -n easyep python=3.10
conda activate easyep
pip install -r requirements.txt
```
### 1.2 Model Preparation

#### System Requirements
> [!NOTE] 
> Linux with Python 3.10 only. Mac and Windows are not supported.

Dependencies:
```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

#### Model Weights Conversion
Our code is based on the official inference demo provided by [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#61-inference-with-deepseek-infer-demo-example-only). It requires Model Weights Conversion. Here, using an 8 x H200 141GB node, the conversion method is as follows:

Download the model weights from Hugging Face, and put them into /path/to/DeepSeek-R1 folder. Then convert Hugging Face model weights to a specific format:

```shell
python pruning/convert.py --hf-ckpt-path /path/to/DeepSeek-R1 --save-path /path/to/DeepSeek-R1-Demo --n-experts 256 --model-parallel 8
```

### Data Preparation


## 2. Expert Selection
This part primarily involves extracting and calculating internal MoE hidden states using calibration data, which will be used for later pruning.
```bash
torchrun --nproc_per_node=8 pruning/inf_new.py \
    --ckpt-path /path/to/DeepSeek-R1-Demo \
    --config configs/config_671B.json \
    --input-file dataset/aime23_full \
    --output expert_statistics/token_information/aime.jsonl
```

The expert mask matrix is then derived using statistical information. 
```bash
python pruning/expert_selection.py \
    --input_file expert_statistics/token_information/aime.jsonl \
    --output_file expert_statistics/expert_information/aime23.pt \
    --expert_mask expert_statistics/expert_mask/aime23_128_mask.json \
    --target_number 128
```

## 3. Model Pruning
* 我们使用sglang推理框架，提供两个不同的版本，快速评测只mask相应的expert（load模型参数量不变），实际剪枝则只load剪枝后的MoE参数。

### 3.1 Quick Start
> [!NOTE] 
> 我们的sglang版本为0.4.3

首先将 path-to-your-conda/envs/easyep/lib/python3.10/site-packages/sglang 部分的代码替换为 sglang/sglang_full/sglang,

之后需要修改sglang/srt/models/deepseek_v2.py line 68中的mask文件：
```python
current_fp = "EASYEP/expert_statistics/expert_mask/aime23_full_br_128.json" # 替换为你自己的mask文件
```

最后直接启动sglang服务
```bash
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0   python3 -m sglang.launch_server     --model-path /export/ruc/EASY-Prune/pruned_model --tp 8 --dist-init-addr localhost:5002   --trust-remote-code --mem-fraction-static 0.9 --host 0.0.0.0 --port 60000 --context-length 32768 --max-prefill-token 32500
```
注意该版本为了适配性，依然每层加载256个routed experts，只是在gate输出部分进行mask，完全剪枝后的方案见3.2。

### 3.2 实际剪枝
#### 模型剪枝
首先运行模型剪枝程序，根据输入的expert_mask文件裁剪模型权重(这里的DeepSeek-R1选取原始hf版本即可)
```bash
python pruning/model_prune.py \
    --mask_json expert_statistics/expert_mask/aime23_full_br_128.json \
    --input_dir path/to/DeepSeek-R1 \
    --output_dir pruned_model
```

此时需要手动修改config.json中的n_routed_experts，适配裁剪后每层的专家数量
```json
"n_routed_experts": 128
```

#### 部署剪枝后的模型
我们使用sglang部署剪枝后的模型，以进行进一步的评测:
你可以直接复用我们的sglang/srt文件
将 path-to-your-conda/envs/easyep/lib/python3.10/site-packages/sglang 部分的代码替换为 sglang/sglang_pruned/sglang,

如果你所使用的sglang版本有所不同，这里我们详细列举了需要修改的几处代码:
1. sglang/srt/models/deepseek_v2.py
2. sglang/srt/layers/moe/topk.py

## 4. Evaluation
这里我们开源math相关的评测代码，直接运行脚本`evaluation/scripts/run_eval.sh`即可

## 5. Citation