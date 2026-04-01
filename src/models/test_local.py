"""Verify that local base models and fine-tuned LoRA adapters load and generate correctly."""
from src.config import OPEN_SOURCE_MODELS, MODELS_DIR, FINETUNED_MODELS_DIR
from src.models.llm import call_model
from src.utils import print_separator


def test_local_models() -> None:
    """Load each local model and run a simple prompt to verify it works."""
    prompt = "Say exactly: hello"

    for alias in OPEN_SOURCE_MODELS:
        local_path = str(MODELS_DIR / alias)
        print(f"\nTesting {alias} from {local_path} ...")
        try:
            response = call_model(model_name=local_path, prompt=prompt)
            print(f"\nResponse={response.text!r} | tokens={response.input_tokens}/{response.output_tokens} | time={response.response_time:.2f}s")
        except Exception as e:
            print(f"Fail {alias}: {e}")


# NOTE: Cannot run locally because it requires sm_70+ GPU and ≥8GB VRAM. Run on Kaggle, Colab (T4), or EuroHPC only.
def test_llama32_local_models() -> None:
    """Test Llama 3.2 3B base model and both of its fine-tuned LoRA adapters."""
    prompt = "Say exactly: hello"
    print_separator("TEST CALLING OPEN SOURCE MODELS")
    print(f"\nPrompt: {prompt}")

    local_path = str(MODELS_DIR / "llama32_3b")
    print(f"\nTesting llama32_3b from {local_path} ...")

    try:
        response = call_model(model_name=local_path, prompt=prompt)
        print(f"Response={response.text!r} | tokens={response.input_tokens}/{response.output_tokens} | time={response.response_time:.2f}s")
    except Exception as e:
        print(f"FAIL: {e}")


    dst_path = str(FINETUNED_MODELS_DIR / "dummy/dst_lora")
    print(f"\nTesting fine-tuned dst_lora from {dst_path} ...")
    try:
        response = call_model(model_name=dst_path, prompt=prompt)
        print(f"\nResponse={response.text!r} | tokens={response.input_tokens}/{response.output_tokens} | time={response.response_time:.2f}s")
    except Exception as e:
        print(f"FAIL: {e}")

    respgen_path = str(FINETUNED_MODELS_DIR / "dummy/respgen_lora")
    print(f"\nTesting fine-tuned respgen_lora from {respgen_path} ...")
    try:
        response = call_model(model_name=respgen_path, prompt=prompt)
        print(f"\nResponse={response.text!r} | tokens={response.input_tokens}/{response.output_tokens} | time={response.response_time:.2f}s")
    except Exception as e:
        print(f"FAIL: {e}")

    print_separator("END OF TEST CALLING OPEN SOURCE MODELS")


# Run with: python -m src.models.test_local
if __name__ == "__main__":
    test_llama32_local_models()
