#!/bin/bash
#SBATCH --job-name=easyep_v4
#SBATCH --account=rrg-tayebi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/sinam/deepseek_easy_ep/easyep-%j.out

set -euo pipefail
module load cuda/13.2

BIG=/project/6114328/sinam/dsv4
SMALL=/project/6084786/sinam/dsv4
E=/scratch/sinam/deepseek_easy_ep

export PYTHONUNBUFFERED=1
export TILELANG_CACHE_DIR=/home/sinam/.tilelang/cache
export TRITON_CACHE_DIR=/home/sinam/.triton
export TORCHINDUCTOR_CACHE_DIR=/home/sinam/.inductor
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "host=$(hostname) start=$(date -u +%FT%TZ)"

cd "$BIG/code/inference"
"$SMALL/venv/bin/torchrun" --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29514 \
  "$E/easyep_v4.py" pipeline \
  --ckpt-path "$BIG/checkpoint" \
  --config "$BIG/code/inference/config.json" \
  --code-dir "$BIG/code/inference" \
  --calib-dir "$E/inputs/vulnerable-js-files" \
  --questions "$E/inputs/questions_used.json" \
  --out "$E/results/easyep_keep128" \
  --n-calib 25 \
  --keep 128 \
  --max-seq-len 4096 \
  --max-new-tokens 256

echo "end=$(date -u +%FT%TZ)"
