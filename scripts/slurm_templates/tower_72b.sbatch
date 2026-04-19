#!/usr/bin/env bash
#SBATCH --job-name=mtm-tower-72b
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --exclude=aisurrey26
#SBATCH --output=outputs/slurm-logs/%x-%j.out
#SBATCH --error=outputs/slurm-logs/%x-%j.err

# Tower-Plus-72B (Qwen2 backbone) with vLLM tensor-parallel=4.
# A100 80GB × 4 recommended; if only 40GB available, set:
#   --override scorers.0.params.gpu_memory_utilization=0.85
CONFIG="${1:?usage: sbatch ... <run-config.yaml>}"
source "$(dirname "$0")/common_header.sbatch"
mt-metrix score --config "$CONFIG" --output-root "$SCRATCH/outputs"
