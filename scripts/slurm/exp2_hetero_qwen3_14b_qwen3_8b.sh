#!/bin/bash
#SBATCH --job-name=exp2_hetero_qwen3_14b_qwen3_8b
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/exp2_hetero_qwen3_14b_qwen3_8b_%j.out
#SBATCH --error=logs/exp2_hetero_qwen3_14b_qwen3_8b_%j.err
#SBATCH --account=EUHPC_D34_189

# Exp2: Modular zero-shot pipeline for hetero_qwen3_14b_qwen3_8b
# DST: Qwen3-14B (strong slot extraction)
# ResponseGen: Qwen3-8B (faster response generation)
cd $WORK/mlp4cs

# Load conda
source ~/.bashrc
conda activate mlp4cs

# Offline mode (no internet on compute nodes) + suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run experiment
python -m src.main
