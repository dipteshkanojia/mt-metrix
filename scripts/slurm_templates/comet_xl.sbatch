#!/usr/bin/env bash
#SBATCH --job-name=mtm-comet-xl
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --exclude=aisurrey26
#SBATCH --output=outputs/slurm-logs/%x-%j.out
#SBATCH --error=outputs/slurm-logs/%x-%j.err

# For cometkiwi-da-xl and XCOMET-XL (XLM-R-XL backbone, ~3.5B params).
CONFIG="${1:?usage: sbatch ... <run-config.yaml>}"
source "$(dirname "$0")/common_header.sbatch"
mt-metrix score --config "$CONFIG" --output-root "$SCRATCH/outputs"
