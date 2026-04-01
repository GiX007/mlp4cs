"""Fine-tune a local model on a role-specific dataset using Unsloth + (Q)LoRA."""
import json
import torch
from datasets import Dataset
from src.config import (
    FINETUNE_DST_FILE, FINETUNE_RESPGEN_FILE, FINETUNED_MODELS_DIR, FINETUNE_EPOCHS,
    OPEN_SOURCE_MODELS, MODELS_DIR, LOCAL_MAX_SEQ_LENGTH, LOCAL_LOAD_IN_4BIT, LOCAL_DTYPE,
)

ROLE_DATASET_MAP: dict[str, str] = {
    "dst": str(FINETUNE_DST_FILE),
    "response_generator": str(FINETUNE_RESPGEN_FILE),
}


def format_example(ex: dict) -> dict:
    """Format a single training example into Alpaca-style text.

    Args:
        ex: Dict with keys 'instruction', 'input', 'output'

    Returns:
        Dict with single 'text' key in Alpaca format
    """
    text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}"
    )
    return {"text": text}


def load_ft_dataset(role: str) -> Dataset:
    """Load and format the fine-tuning dataset for a given role.

    Args:
        role: Pipeline role, either 'dst' or 'response_generator'

    Returns:
        HuggingFace Dataset with a single 'text' field in Alpaca format
    """
    path = ROLE_DATASET_MAP[role]
    with open(path, "r") as f:
        examples = json.load(f)

    dataset = Dataset.from_list([format_example(e) for e in examples])
    print(f"Data loaded {len(dataset)} examples for role '{role}'")
    return dataset


def finetune(role: str, model_alias: str) -> None:
    """Fine-tune a model for a given pipeline role using (Q)LoRA.

    Args:
        role: Pipeline role, either 'dst' or 'response_generator'
        model_alias: Model alias from OPEN_SOURCE_MODELS (e.g., 'llama31_8b')

    Returns:
        None
    """
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

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

    dataset = load_ft_dataset(role)

    optim = "adamw_8bit" if LOCAL_LOAD_IN_4BIT else "adamw_torch"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=LOCAL_MAX_SEQ_LENGTH,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=FINETUNE_EPOCHS,
            learning_rate=2e-4,
            logging_steps=10,
            optim=optim,
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(FINETUNED_MODELS_DIR / f"{model_alias}_{role}_outputs"),
            report_to="none",
        ),
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

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Adapter saved to {save_path}")


# Run one job per combination on EuroHPC:
# python -m scripts.finetune --role dst --model llama32_3b
# python -m scripts.finetune --role dst --model llama31_8b
# python -m scripts.finetune --role response_generator --model llama31_8b
# python -m scripts.finetune --role dst --model qwen25_7b
# python -m scripts.finetune --role response_generator --model qwen25_7b
# python -m scripts.finetune --role response_generator --model mistral_12b

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["dst", "response_generator"])
    parser.add_argument("--model", required=True, choices=list(OPEN_SOURCE_MODELS.keys()))
    args = parser.parse_args()
    finetune(role=args.role, model_alias=args.model)
