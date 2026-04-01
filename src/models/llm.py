"""LLM routing: single entry point for OpenAI, Anthropic, and local Unsloth models."""
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import LLM_MAX_TOKENS, LLM_TEMPERATURE, LOCAL_MAX_SEQ_LENGTH, LOCAL_LOAD_IN_4BIT, LOCAL_DTYPE
from src.utils import calculate_cost

# Load .env so os.getenv() works for all API keys in this file
load_dotenv()

# Module-level cache: avoids reloading the same model on every call
_local_model_cache: dict = {}

@dataclass
class ModelResponse:
    """Standardized response from any LLM backend."""
    text: str
    input_tokens: int
    output_tokens: int
    response_time: float
    cost: float


def call_model(model_name: str, prompt: str, system_prompt: str = "", max_tokens: int = LLM_MAX_TOKENS, temperature: float = LLM_TEMPERATURE) -> ModelResponse:
    """Route a prompt to the correct LLM backend and return a standardized ModelResponse.

    Args:
        model_name: Model identifier name (e.g., 'gpt-4o-mini') or local path
        prompt: User message to send
        system_prompt: System instruction for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        ModelResponse with text, token counts, cost, and response time
    """
    if model_name.startswith("gpt-"):
        return _call_openai(model_name, prompt, system_prompt, max_tokens, temperature)
    elif model_name.startswith("claude-"):
        return _call_anthropic(model_name, prompt, system_prompt, max_tokens, temperature)
    else:
        return _call_unsloth(model_name, prompt, system_prompt, max_tokens, temperature)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_openai(model_name: str, prompt: str, system_prompt: str = "", max_tokens: int = LLM_MAX_TOKENS, temperature: float = LLM_TEMPERATURE) -> ModelResponse:
    """
    Call an OpenAI model and return a standardized ModelResponse.

    Args:
        model_name: OpenAI model identifier (e.g., 'gpt-4o-mini')
        prompt: User message to send
        system_prompt: System instruction for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        ModelResponse with text, token counts, cost, and response time
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - start

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return ModelResponse(
        text=response.choices[0].message.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=calculate_cost(model_name, input_tokens, output_tokens),
        response_time=elapsed,
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def _call_anthropic(model_name: str, prompt: str, system_prompt: str = "", max_tokens: int = LLM_MAX_TOKENS, temperature: float = LLM_TEMPERATURE) -> ModelResponse:
    """Call an Anthropic model and return a standardized ModelResponse.

    Args:
        model_name: Anthropic model identifier (e.g., 'claude-3-haiku-20240307')
        prompt: User message to send
        system_prompt: System instruction for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        ModelResponse with text, token counts, cost, and response time.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.time()
    response = client.messages.create(
        model=model_name,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - start

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return ModelResponse(
        text=response.content[0].text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=calculate_cost(model_name, input_tokens, output_tokens),
        response_time=elapsed,
    )


def _call_unsloth(model_name: str, prompt: str, system_prompt: str = "", max_tokens: int = LLM_MAX_TOKENS, temperature: float = LLM_TEMPERATURE) -> ModelResponse:
    """Load a local Unsloth model (base or LoRA fine-tuned) and return a standardized ModelResponse.

    Args:
        model_name: Local path to model directory (base model or LoRA adapter)
        prompt: User message to send
        system_prompt: System instruction for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        ModelResponse with text, token counts, cost 0.0, and response time
    """
    import peft
    from unsloth import FastLanguageModel

    if model_name not in _local_model_cache:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=LOCAL_MAX_SEQ_LENGTH,
            dtype=LOCAL_DTYPE,
            load_in_4bit=LOCAL_LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)
        _local_model_cache[model_name] = (model, tokenizer)

    model, tokenizer = _local_model_cache[model_name]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Legacy tokenizer returns BatchEncoding, not a raw tensor so we extract input_ids explicitly
    input_ids = inputs["input_ids"].to("cuda")
    input_length = input_ids.shape[1]

    # Use correct generate path depending on model type (base vs LoRA fine-tuned)
    # model.base_model.model.generate bypasses Unsloth's fast Triton kernel which breaks on repeated calls with growing prompt lengths (multi-turn generation)
    if isinstance(model, peft.PeftModel):
        generate_fn = model.base_model.model.generate
    else:
        generate_fn = model.generate

    start = time.time()
    outputs = generate_fn(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else None,
        use_cache=False,  # Disables KV cache to prevent shape mismatch errors in Unsloth's fast attention kernel when called repeatedly with growing prompts
    )
    elapsed = time.time() - start

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = decoded.split("assistant")[-1].strip()

    input_tokens = input_length
    output_tokens = len(outputs[0]) - input_length

    return ModelResponse(
        text=generated_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=0.0,
        response_time=elapsed,
    )
