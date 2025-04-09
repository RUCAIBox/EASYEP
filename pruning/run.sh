cd /export/ruc/EASY-Prune
python pruning/model_prune.py \
    --mask_json expert_statistics/expert_mask/aime23_full_br_64.json \
    --input_dir /export/models/DeepSeek-R1 \
    --output_dir pruned_model