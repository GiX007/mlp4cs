"""Inspect a fine-tuned model's training summary.

Reminder: Fine-Tuning Step Math

Train set: 2,850 dialogues (hotel + restaurant only).
Each dialogue has ~10-11 turns on average, but we only create training examples from USER turns with an active hotel/restaurant intent.
Turns like "thank you" / "goodbye" (intent=NONE) are excluded.
Result: 10,846 DST training examples | 13,174 RespGen training examples.

Batch size = 8 (2 per device × 4 gradient accumulation).
One epoch = one full pass through all examples, 8 at a time. Total epochs = 3.
DST: 10,846 / 8 = 1,356 steps per epoch → 1,356 × 3 = 4,068 max steps.
RespGen: 13,174 / 8 = 1,647 steps per epoch → 1,647 × 3 = 4,940 max steps.

Eval happens every 100 steps (not per epoch).
DST gets ~13 eval points per epoch (step 100, 200, ..., 1300). So for 4068 max steps, we expect ~40 evals total.
Early stopping (patience=3) counts consecutive evals with no improvement, NOT epochs. After 3 bad evals in a row (= 300 steps), training stops, even mid-epoch. The "3 epochs" setting is just a max cap.

Example: qwen3_14b_dst stopped at step 900.
900 / 1,356 = 0.66 → 66% through epoch 1, never finished it.
Best eval_loss at step 600, then 3 worse evals (700, 800, 900) → STOP.
If it had survived all 3 epochs, it would have reached step 4,068 (epoch 3.00) with 40 eval checkpoints. It only made it to step 900 (epoch 0.66) with 9.
"""
import json
import argparse
from pathlib import Path

from src.config import FINETUNED_MODELS_DIR
from src.utils import print_separator


def inspect_training(model_dir: Path) -> None:
    """
    Print key training diagnostics from a training_summary.json.

    Params:
        model_dir: path to a finetuned model folder

    Returns:
        None
    """
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        print(f"No training_summary.json in {model_dir.name}, skipping.\n")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print_separator(f"Inspecting {model_dir.name}")

    # Basic info
    print(f"Model: {summary['model_alias']} | Role: {summary['role']}")
    print(f"Trainable params: {summary['trainable_params']:,} / {summary['total_params']:,} ({summary['trainable_pct']})")
    print(f"Runtime: {summary['train_runtime_min']} min | Peak VRAM: {summary['peak_vram_gb']} GB | LoRA VRAM: {summary['lora_vram_gb']} GB")

    # Extract log entries
    log_history = summary["log_history"]
    eval_entries = [e for e in log_history if "eval_loss" in e]
    train_entries = [e for e in log_history if "loss" in e]

    if not eval_entries:
        print("No eval entries found in log_history!\n")
        return

    # Epoch and step info
    max_epochs = summary["epochs"]
    last_eval = eval_entries[-1]
    last_epoch_reached = last_eval.get("epoch", None)
    last_step = last_eval["step"]

    print(f"\nEpochs & Steps")
    print(f"Max epochs configured: {max_epochs}")
    print(f"Last epoch reached: {last_epoch_reached}")
    print(f"Last step reached: {last_step}")
    print(f"Total eval points: {len(eval_entries)}")

    if last_epoch_reached is not None and last_epoch_reached < max_epochs - 0.1:
        print(f">> EARLY STOPPING FIRED (stopped before epoch {max_epochs})")
    else:
        print(f">> Training ran full {max_epochs} epochs (no early stopping)")

    # Best checkpoint
    best = min(eval_entries, key=lambda e: e["eval_loss"])
    print(f"\nBest Checkpoint")
    print(f"Best eval_loss: {best['eval_loss']:.4f} at step {best['step']} (epoch {best.get('epoch', '?')})")
    print(f"Last eval_loss: {last_eval['eval_loss']:.4f} at step {last_step} (epoch {last_epoch_reached})")

    # Overfitting check
    print(f"\nLast 5 Eval Points")
    for entry in eval_entries[-5:]:
        marker = " << best" if entry["step"] == best["step"] else ""
        print(f"  Step {entry['step']}: eval_loss = {entry['eval_loss']:.4f}{marker}")

    print()


# Inspect a single model:
#   python -m scripts.inspect_ft --model llama31_8b_dst
#   python -m scripts.inspect_ft --model llama31_8b_response_generator
#   python -m scripts.inspect_ft --model llama32_3b_dst
#   python -m scripts.inspect_ft --model llama32_3b_response_generator
#   python -m scripts.inspect_ft --model phi4_14b_dst
#   python -m scripts.inspect_ft --model phi4_14b_response_generator
#   python -m scripts.inspect_ft --model qwen3_4b_dst
#   python -m scripts.inspect_ft --model qwen3_4b_response_generator
#   python -m scripts.inspect_ft --model qwen3_8b_dst
#   python -m scripts.inspect_ft --model qwen3_8b_response_generator
#   python -m scripts.inspect_ft --model qwen3_14b_dst
#   python -m scripts.inspect_ft --model qwen3_14b_response_generator
#
# Inspect all finetuned models at once:
#   python -m scripts.inspect_ft --all

if __name__ == "__main__":
    all_models = sorted([d.name for d in FINETUNED_MODELS_DIR.iterdir() if d.is_dir()])

    parser = argparse.ArgumentParser(description="Inspect fine-tuning results.")
    parser.add_argument("--model", choices=all_models, help="Model folder name to inspect")
    parser.add_argument("--all", action="store_true", help="Inspect all finetuned models")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model <name> or --all")

    if args.all:
        for name in all_models:
            inspect_training(FINETUNED_MODELS_DIR / name)
    else:
        inspect_training(FINETUNED_MODELS_DIR / args.model)
