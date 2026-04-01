"""Download all local models from HuggingFace and save to data/models/."""
import subprocess
from pathlib import Path
from src.config import OPEN_SOURCE_MODELS, MODELS_DIR


def download_model(alias: str, model_name: str) -> None:
    """Download a single model using huggingface-cli and save to local directory.

    Args:
        alias: Short name used as folder name (e.g., 'llama31_8b').
        model_name: Full HuggingFace model ID (e.g., 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit').
    """
    save_path: Path = MODELS_DIR / alias

    if save_path.exists():
        print(f"\nSkip {alias} as it already exists at {save_path}")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Downloading {alias} ({model_name}) ...")

    subprocess.run(["huggingface-cli", "download", model_name, "--local-dir", str(save_path)], check=True)
    print(f"\n{alias} saved to {save_path}")


def download_all_models() -> None:
    """Download all open-source models defined in OPEN_SOURCE_MODELS."""
    for alias, hf_id in OPEN_SOURCE_MODELS.items():
        download_model(alias, hf_id)


# Run with: python -m scripts.download_models
if __name__ == "__main__":
    download_all_models()
