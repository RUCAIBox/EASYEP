
model_list=(\
    "/path/DeepSeek-R1" \
)
L=32768
num_model=${#model_list[@]}

tgt_path=easyep/evaluation
for ((i=0;i<$num_model;i++)) do
{
    MODEL_PATH=${model_list[$i]}
    SRC_PATH=${MODEL_PATH}

    python run_sglang.py \
        --data_name AIME24 \
        --target_path ${tgt_path} \
        --model_name_or_path ./outputs \
        --max_tokens ${L} \
        --system_prompt none \
        --paralle_size 1 \
        --number 5 & 
        
    python run_sglang.py \
         --data_name hmmt_feb_2025 \
         --target_path ${tgt_path} \
         --model_name_or_path ./outputs \
         --max_tokens ${L} \
         --system_prompt none \
         --paralle_size 1  --number 5 &
        
     python run_sglang.py \
        --data_name AIME25 \
         --target_path ${tgt_path} \
        --model_name_or_path ./outputs \
         --max_tokens ${L} \
         --system_prompt none \
         --paralle_size 1  --number 5 & 

}
done
