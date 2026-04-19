"""Fine-tune a local model on a role-specific dataset using Unsloth + (Q)LoRA."""
import json
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from src.config import (
    FINETUNE_DST_TRAIN_FILE, FINETUNE_RESPGEN_TRAIN_FILE, FINETUNE_DST_DEV_FILE, FINETUNE_RESPGEN_DEV_FILE, FINETUNED_MODELS_DIR, FINETUNE_EPOCHS,
    OPEN_SOURCE_MODELS, MODELS_DIR, LOCAL_MAX_SEQ_LENGTH, LOCAL_LOAD_IN_4BIT, LOCAL_DTYPE,
)

ROLE_DATASET_MAP: dict[str, dict[str, str]] = {
    "dst": {
        "train": str(FINETUNE_DST_TRAIN_FILE),
        "dev": str(FINETUNE_DST_DEV_FILE),
    },
    "response_generator": {
        "train": str(FINETUNE_RESPGEN_TRAIN_FILE),
        "dev": str(FINETUNE_RESPGEN_DEV_FILE),
    },
}


def format_example(ex: dict, eos_token: str) -> dict:
    """
    Format a single training example into Alpaca-style text with EOS appended.

    Args:
        ex: Dict with keys 'instruction', 'input', 'output'
        eos_token: The tokenizer's end-of-sequence string (e.g. '<|end_of_text|>')

    Returns:
        Dict with single 'text' key in Alpaca format, terminated by EOS
    """
    # EOS is required: without it, the model never learns to stop generating.
    text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}{eos_token}"
    )
    return {"text": text}


def load_ft_dataset(role: str, split: str, eos_token: str) -> Dataset:
    """
    Load and format a fine-tuning dataset for a given role and split.

    Args:
        role: Pipeline role, either 'dst' or 'response_generator'
        split: Data split, either 'train' or 'dev'
        eos_token: The tokenizer's end-of-sequence string

    Returns:
        HuggingFace Dataset with a single 'text' field in Alpaca format
    """
    path = ROLE_DATASET_MAP[role][split]
    with open(path, "r") as f:
        examples = json.load(f)

    dataset = Dataset.from_list([format_example(e, eos_token) for e in examples])
    print(f"Loaded {len(dataset)} {split} examples for role '{role}'")
    return dataset


def plot_loss_curves(log_history: list[dict], save_path: str) -> None:
    """
    Parse log_history and save a train + eval loss plot to disk.

    Args:
        log_history: trainer.state.log_history as a list of per-step log dicts
        save_path: path where the PNG file will be written

    Returns:
        None
    """
    # Training logs have 'loss' key, eval logs have 'eval_loss' key
    train_steps = [log["step"] for log in log_history if "loss" in log]
    train_losses = [log["loss"] for log in log_history if "loss" in log]
    eval_steps = [log["step"] for log in log_history if "eval_loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Train loss", alpha=0.7)
    plt.plot(eval_steps, eval_losses, label="Eval loss", marker="o", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training vs Eval Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to {save_path}")


def finetune(role: str, model_alias: str) -> None:
    """
    Fine-tune a model for a given pipeline role using (Q)LoRA.

    Args:
        role: Pipeline role, either 'dst' or 'response_generator'
        model_alias: Model alias from OPEN_SOURCE_MODELS (e.g., 'llama31_8b')

    Returns:
        None
    """
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    if model_alias not in OPEN_SOURCE_MODELS:
        raise ValueError(f"Unknown model alias: {model_alias}. Choose from {list(OPEN_SOURCE_MODELS)}")

    if role not in ROLE_DATASET_MAP:
        raise ValueError(f"Unknown role: {role}. Choose from {list(ROLE_DATASET_MAP)}")

    model_local_path = str(MODELS_DIR / model_alias)
    save_path = FINETUNED_MODELS_DIR / f"{model_alias}_{role}"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nFinetune role={role} | model={model_alias} | epochs={FINETUNE_EPOCHS}")
    print(f"\nLoading base model from {model_local_path} ...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_local_path,
        max_seq_length=LOCAL_MAX_SEQ_LENGTH,
        dtype=LOCAL_DTYPE,
        load_in_4bit=LOCAL_LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Trainable parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = round(100 * trainable_params / total_params, 2)
    print(f"\nModel's total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct}% of total)")

    train_dataset = load_ft_dataset(role, "train", tokenizer.eos_token)
    eval_dataset = load_ft_dataset(role, "dev", tokenizer.eos_token)

    optim = "adamw_8bit" if LOCAL_LOAD_IN_4BIT else "adamw_torch"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=LOCAL_MAX_SEQ_LENGTH,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,  # samples per forward pass
            per_device_eval_batch_size=2,  # same for eval, keeps VRAM predictable
            gradient_accumulation_steps=4,  # effective batch = 2*4 = 8
            warmup_steps=10,  # ramp LR up gradually at start
            num_train_epochs=FINETUNE_EPOCHS,  # max epoch cap (early stopping decides actual end)
            learning_rate=2e-4,  # Unsloth/QLoRA default
            logging_steps=10,  # log train loss every 10 steps
            eval_strategy="steps",  # eval every N steps (not per epoch)
            eval_steps=100,  # run eval every 100 steps (~40 evals total)
            save_strategy="steps",  # checkpoint every N steps
            save_steps=100,  # must match eval_steps for best-checkpoint logic
            save_total_limit=3,  # keep only 3 most recent checkpoints on disk
            load_best_model_at_end=True,  # reload lowest-eval-loss checkpoint at the end
            metric_for_best_model="eval_loss",  # which metric ranks checkpoints
            greater_is_better=False,  # lower loss is better
            optim=optim,  # adamw_8bit for 4-bit QLoRA
            weight_decay=0.001,  # L2 regularization
            lr_scheduler_type="linear",  # LR decays linearly to 0
            seed=3407,  # Unsloth-recommended seed
            output_dir=str(FINETUNED_MODELS_DIR / f"{model_alias}_{role}_outputs"),  # checkpoint dir
            report_to="none",  # disable tensorboard logging
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU {gpu_stats.name} | Max VRAM = {max_memory} GB")
    print(f"GPU reserved before training = {start_gpu_memory} GB")

    print(f"\n>>> Finetune training ...")
    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"\nGPU {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training")
    print(f"GPU Peak reserved memory = {used_memory} GB")
    print(f"GPU Peak reserved memory for LoRA = {used_memory_for_lora} GB")

    # Save training logs + parameter counts
    log_history = trainer.state.log_history
    training_summary = {
        "role": role,
        "model_alias": model_alias,
        "epochs": FINETUNE_EPOCHS,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": trainable_pct,
        "train_runtime_min": round(trainer_stats.metrics["train_runtime"] / 60, 2),
        "peak_vram_gb": used_memory,
        "lora_vram_gb": used_memory_for_lora,
        "log_history": log_history,
    }
    with open(save_path / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\nTraining summary saved to {save_path}/training_summary.json")

    plot_loss_curves(log_history, str(save_path / "loss_plot.png"))

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Adapter saved to {save_path}")


# Run one job per combination on EuroHPC:
# python -m scripts.finetune --role dst --model llama32_3b
# python -m scripts.finetune --role response_generator --model llama32_8b
# python -m scripts.finetune --role dst --model qwen3_8b
# python -m scripts.finetune --role response_generator --model qwen3_8b

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["dst", "response_generator"])
    parser.add_argument("--model", required=True, choices=list(OPEN_SOURCE_MODELS.keys()))
    args = parser.parse_args()
    finetune(role=args.role, model_alias=args.model)
