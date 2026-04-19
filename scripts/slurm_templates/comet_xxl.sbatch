#!/usr/bin/env bash
#SBATCH --job-name=mtm-comet-xxl
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=aisurrey26
#SBATCH --output=outputs/slurm-logs/%x-%j.out
#SBATCH --error=outputs/slurm-logs/%x-%j.err

# For cometkiwi-da-xxl and XCOMET-XXL (XLM-R-XXL backbone, ~10.7B params).
# Requires an A100 80GB for fp16 inference without offload.
CONFIG="${1:?usage: sbatch ... <run-config.yaml>}"
source "$(dirname "$0")/common_header.sbatch"
mt-metrix score --config "$CONFIG" --output-root "$SCRATCH/outputs"
