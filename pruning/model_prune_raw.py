#!/usr/bin/python3
import os
import json
import numpy as np
import re
from glob import glob
from safetensors.torch import load_file, save_file

# 读取 JSON 文件
with open("aime23_full_br_64.json", "r") as f:
    data = json.load(f)

# 转换为 NumPy 数组
mask = np.array(data)
print(mask)

# 获取目标文件夹路径
file_dir = "/data/DeepSeek-R1/"
file_path = os.path.join(file_dir, "model.safetensors.index.json")

with open(file_path,"r") as f:
    json_data=json.load(f)

def json_key_to_indices(key):
    """
    model.layers.<layer_num>.mlp.experts.<expert_num>.* 
    """
    # 正则表达式解析（非贪婪模式）
    pattern = re.compile(
        r"^model\.layers\."              # 固定前缀
        r"(\d+)\."                       # 提取层号
        r"mlp\.experts\."                 # 固定中间层
        r"(\d+)"                         # 提取专家号
        r"(?:\..+)?$"                    # 可选的后缀部分
    )
    
    match = pattern.match(key)
    if not match:
        return (None, None, False)
    
    # 提取数字部分
    try:
        layer = int(match.group(1))
        expert = int(match.group(2))
    except ValueError:
        print(f"数值转换错误：{key} → {match.groups()}")
        return (None, None, False)
    
    # 返回匹配结果及标识
    return (layer, expert, True)

def filter_json_with_mask(json_data, mask):
    """
    - 匹配正则的键：根据掩码保留
    - 不匹配正则的键：无条件保留
    """
    filtered = {}
    newnamedict = {}
    old_layer = 0
    pattern = re.compile(
        r"^model\.layers\."              # 固定前缀
        r"(\d+)\."                       # 提取层号
        r"mlp\.experts\."                 # 固定中间层
        r"(\d+)"                         # 提取专家号
        r"(?:\..+?)$"                    # 可选的后缀部分
    )
    for key, value in json_data["weight_map"].items():
        layer, expert, is_matched = json_key_to_indices(key)
        if is_matched:
            # 仅当索引合法时应用掩码
            if 3 <= layer < 61 and 0 <= expert < 256:
                layer_mask = mask[layer-3]
                nzind = list(np.nonzero(layer_mask)[0])
                if mask[layer-3, expert] == 1:
                    expertid = nzind.index(expert)
                    filtered[key] = value
                    key_tail = key.replace("model.layers."+str(layer),"")
                    key_tail = key_tail.replace(".mlp.experts."+str(expert),"")
                    newname = "model.layers." + str(layer) + ".mlp.experts."+str(expertid) + key_tail
                    newnamedict[key] = newname      
            else:
                # 索引越界的匹配项直接保留
                filtered[key] = value
                newnamedict[key] = key
        else:
            # 非匹配项直接保留
            filtered[key] = value
            newnamedict[key] = key
    return filtered, newnamedict
mask_data, newnamedict =filter_json_with_mask(json_data,mask)

json_data_new = {}
metadata = {}
for key, value in json_data["metadata"].items():
    metadata[key] = value
json_data_new["metadata"] = metadata
json_data_new["weight_map"] = mask_data

with open(os.path.join("/data/DeepSeek-R1", "filtered64.json"), "w") as f:
    json.dump(json_data_new, f, indent=4)


json_data_new = {}
metadata = {}
for key, value in json_data["metadata"].items():
    metadata[key] = value
json_data_new["metadata"] = metadata
mask_data_new = {}
for key, value in mask_data.items():
    newkey = newnamedict[key] 
    mask_data_new[newkey] = value
json_data_new["weight_map"] = mask_data_new
with open(os.path.join("/data/newR1", "model.safetensors.index.json"), "w") as f:
    json.dump(json_data_new, f, indent=4)

safetensor_files = sorted(list(glob("/data/DeepSeek-R1/*.safetensors")))

def parse_layer_number(key):
    match = re.match(r"^model\.layers\.(\d+)\.", key)
    if not match:
        raise ValueError(f"无法解析层数: {key}")
    return int(match.group(1))

for safetensor_file in safetensor_files:
    print("safetensor_file:", safetensor_file)
    state_dict = load_file(safetensor_file)
    new_state_dict = {}
    file_name = os.path.basename(safetensor_file)
    for weight_name, weight in state_dict.items():
        print("weight_name:", weight_name)
        if weight_name in mask_data:
            newname = newnamedict[weight_name]
            new_state_dict[newname] = weight


    new_safetensor_file = os.path.join("/data/newR1/", file_name)
    save_file(new_state_dict, new_safetensor_file)