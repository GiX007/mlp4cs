#!/bin/bash
#SBATCH --job-name=ft_llama32_3b
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/ft_llama32_3b_%j.out
#SBATCH --error=logs/ft_llama32_3b_%j.err
#SBATCH --account=EUHPC_D34_189

# Exp3: Fine-tune DST + ResponseGen LoRA adapters for Llama-3.2-3B
cd $WORK/mlp4cs

# Load conda
source ~/.bashrc
conda activate mlp4cs

# Offline mode (no internet on compute nodes) + suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fine-tune both adapters sequentially
python -m scripts.finetune --role dst --model llama32_3b
python -m scripts.finetune --role response_generator --model llama32_3b