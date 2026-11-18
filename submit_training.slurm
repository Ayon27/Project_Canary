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

export MPLCONFIGDIR="/tmp/matplotlib_${USER}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Detect GPUs allocated to this specific SLURM job
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -v '^$' | wc -l)
elif [ -n "${SLURM_GPUS_ON_NODE}" ]; then
    NUM_GPUS=${SLURM_GPUS_ON_NODE}
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS=${NUM_GPUS:-1}
fi
if [ "${NUM_GPUS}" -eq 0 ]; then
    NUM_GPUS=1
fi

echo "Detected ${NUM_GPUS} GPU(s) available for training."

torchrun --nproc_per_node=${NUM_GPUS} train_scratch.py \
    --data_dir "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --batch_size 4 \
    --grad_accum 8 \
    --epochs 10 \
    --lr 4e-4 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --eval_steps 300 \
    --save_steps 300 2>&1 | tee "${TARGET_DIR}/train.log"
