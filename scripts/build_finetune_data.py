"""Build fine-tuning datasets for Exp3."""
from src.data.ft_datasets_builder import build_dst_dataset, build_respgen_dataset, save_dataset
from src.config import FINETUNE_DST_TRAIN_FILE, FINETUNE_RESPGEN_TRAIN_FILE, FINETUNE_DST_DEV_FILE, FINETUNE_RESPGEN_DEV_FILE

# Run with: python -m scripts.build_finetune_data
if __name__ == "__main__":
    # Train split (used to fit LoRA weights)
    dst_train = build_dst_dataset("train")
    save_dataset(dst_train, FINETUNE_DST_TRAIN_FILE)

    respgen_train = build_respgen_dataset("train")
    save_dataset(respgen_train, FINETUNE_RESPGEN_TRAIN_FILE)

    # Dev split (used for in-training eval loss and early stopping)
    dst_dev = build_dst_dataset("dev")
    save_dataset(dst_dev, FINETUNE_DST_DEV_FILE)

    respgen_dev = build_respgen_dataset("dev")
    save_dataset(respgen_dev, FINETUNE_RESPGEN_DEV_FILE)
