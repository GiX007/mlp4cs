"""Build fine-tuning datasets for Exp3."""
from src.data.ft_datasets_builder import build_dst_dataset, build_respgen_dataset, save_dataset
from src.config import FINETUNE_DST_FILE, FINETUNE_RESPGEN_FILE

# Run with: python -m scripts.build_finetune_data
if __name__ == "__main__":
    dst_samples = build_dst_dataset("train")
    save_dataset(dst_samples, FINETUNE_DST_FILE)

    respgen_samples = build_respgen_dataset("train")
    save_dataset(respgen_samples, FINETUNE_RESPGEN_FILE)
