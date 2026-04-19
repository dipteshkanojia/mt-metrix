#!/usr/bin/env bash
#SBATCH --job-name=mtm-tower-7b
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=aisurrey26
#SBATCH --output=outputs/slurm-logs/%x-%j.out
#SBATCH --error=outputs/slurm-logs/%x-%j.err

# TowerInstruct-7B (any v0.1/v0.2), TowerInstruct-Mistral-7B, Tower-Plus-2B,
# Tower-Plus-9B (at 9B you may want tower_13b.sbatch for 2×GPU headroom).
CONFIG="${1:?usage: sbatch ... <run-config.yaml>}"
source "$(dirname "$0")/common_header.sbatch"
mt-metrix score --config "$CONFIG" --output-root "$SCRATCH/outputs"
