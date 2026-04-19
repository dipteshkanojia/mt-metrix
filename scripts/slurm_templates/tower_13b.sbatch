#!/usr/bin/env bash
#SBATCH --job-name=mtm-tower-13b
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=aisurrey26
#SBATCH --output=outputs/slurm-logs/%x-%j.out
#SBATCH --error=outputs/slurm-logs/%x-%j.err

# TowerInstruct-13B, Tower-Plus-9B with vLLM tensor-parallel=2.
CONFIG="${1:?usage: sbatch ... <run-config.yaml>}"
source "$(dirname "$0")/common_header.sbatch"
mt-metrix score --config "$CONFIG" --output-root "$SCRATCH/outputs"
