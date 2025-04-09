# 将本地文件scp到远程服务器上
# H200 upload
scp -r -P 2222 -i ~/.ssh/h200_id_rsa /Users/panospeng/Desktop/lcb_runner ybsk2@root@10.200.89.223@10.99.134.25:/export/ruc/penghan/LiveCodeBench-main
scp -r -P 2222 -i ~/.ssh/h200_id_rsa /Users/panospeng/Desktop/AgentBench-main/src ybsk2@root@10.200.89.223@10.99.134.25:/export/ruc/AgentBench-main
scp -r -P 2222 -i ~/.ssh/h200_id_rsa /Users/panospeng/Desktop/AgentBench-main/configs ybsk2@root@10.200.89.223@10.99.134.25:/export/ruc/AgentBench-main
scp -r -P 2222 -i ~/.ssh/h200_id_rsa /Users/panospeng/Desktop/src/client/agents ybsk2@root@10.200.89.223@10.99.134.25:/export/ruc/AgentBench-main/src/client/agents
scp -r /Users/panospeng/Desktop/src/client/agents/http_agent.py h200-223:/export/ruc/AgentBench-main/src/client/agents

scp -r /Users/panospeng/Desktop/USMLE h200-223:/export/ruc

scp -r /Users/panospeng/Desktop/EASY-Prune/sglang/sglang h200-223:/usr/local/anaconda3/envs/train/lib/python3.10/site-packages

scp -r /Users/panospeng/Desktop/EASY-Prune/sglang/sglang_full/sglang h200-223:/usr/local/anaconda3/envs/expert/lib/python3.12/site-packages

scp -r /Users/panospeng/Desktop/EASY-Prune/sglang/sglang_pruned/sglang h200-223:/usr/local/anaconda3/envs/train/lib/python3.10/site-packages

scp -r /Users/panospeng/Desktop/EASY-Prune/sglang/sglang_pruned/sglang/srt h200-223:/usr/local/anaconda3/envs/expert4/lib/python3.12/site-packages/sglang

/usr/local/anaconda3/envs/expert4/lib/python3.12/site-packages/sglang
# H200 download
scp -r h200-223:/usr/local/anaconda3/envs/expert2/lib/python3.12/site-packages/sglang /Users/panospeng/Desktop/EASY-Prune/sglang/sglang_pruned

scp -r h200-223:/usr/local/anaconda3/envs/expert2/lib/python3.12/site-packages/sglang /Users/panospeng/Desktop/EASY-Prune/final_sglang/sglang_prune

scp  h200-223:/export/ruc/expert.yaml /Users/panospeng/Desktop/EASY-Prune/sglang


scp -r h200-223:/usr/local/anaconda3/envs/train/lib/python3.10/site-packages/sglang /Users/panospeng/Desktop/EASY-Prune/sglang

scp -r h200-223:/usr/local/anaconda3/envs/expert/lib/python3.12/site-packages/sglang /Users/panospeng/Desktop/EASY-Prune/sglang/sglang_raw

# moe_cpu download
scp -P 35133 root@ssh-cn-beijing2.ebtech.com:/data/H20_copy/DeepSeek-V3.zip /Users/panospeng/Desktopn/