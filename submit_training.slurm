#!/bin/bash
#SBATCH --job-name=train_qwen_scratch
#SBATCH --partition=uri-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -o pipefail
mkdir -p logs

source activate /work/pi_kaleel_mahmood_uri_edu/syed_uri/.conda/envs/torch311
pip install -q matplotlib

OUTPUT_DIR="./output"
MODEL_NAME="qwen2.5_scratch"
TARGET_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
mkdir -p "${TARGET_DIR}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=${NUM_GPUS} train_scratch.py \
    --data_dir "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --batch_size 4 \
    --grad_accum 8 \
    --epochs 30 \
    --patience 4 2>&1 | tee "${TARGET_DIR}/train.log"
