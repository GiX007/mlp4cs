"""Model Exploration Script: Inspect raw API responses before ModelResponse wrapping."""
import os
import torch
from openai import OpenAI
from anthropic import Anthropic
from src.utils import print_separator

from dotenv import load_dotenv
load_dotenv()


def explore_openai_raw() -> None:
    """Explore raw OpenAI API response structure."""
    print_separator("RAW OPENAI RESPONSE STRUCTURE")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "Say love in 2 different languages."
    print(f"\nPrompt: {prompt}\n")

    print("Calling OpenAI API...")
    raw_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )

    print("\n" + "=" * 60)
    print("\nRAW RESPONSE OBJECT\n")
    print(f"Type: {type(raw_response)}")
    print(f"\nFull object response:\n{raw_response}\n")
    print("=" * 60)

    print("\nUNPACKING STRUCTURE\n")

    # Step 1: Get all top-level fields
    print(f"Top-level fields: {list(raw_response.model_dump().keys())}")

    # Step 2: Convert to dict and inspect each field's type
    response_dict = raw_response.model_dump()
    print("\nField Types")
    for key, value in response_dict.items():
        print(f"  {key}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 3: Dive into content
    print(f"\nDive into 'choices':\n{raw_response.choices}\n")
    print(f"choices: {type(raw_response.choices)} with {len(raw_response.choices)} item(s)")
    if raw_response.choices:
        first_choice = raw_response.choices[0]
        print(f"  choices[0]: {type(first_choice)}")
        print(f"  choices[0] fields: {list(first_choice.model_dump().keys())}")

        print(f"\nDive into 'message':\n{first_choice.message}\n")
        print(f"  choices[0].message: {type(first_choice.message)}")
        print(f"  choices[0].message fields: {list(first_choice.message.model_dump().keys())}")

        print(f"\n  Dive into 'content':")
        print(f"    message.content type: {type(first_choice.message.content)} = {first_choice.message.content}")

        print(f"\n  Other 'message' fields:")
        print(f"    message.role type: {type(first_choice.message.role)} = {first_choice.message.role}")
        print(f"    message.tool_calls: {type(first_choice.message.tool_calls)}")

    print("\n" + "=" * 60)

    # Step 4: Usage inspection
    print(f"\nDive into 'usage':\n{raw_response.usage}\n")
    print(f"usage: {type(raw_response.usage)}")
    print(f"usage fields: {list(raw_response.usage.model_dump().keys())}")
    for field, value in raw_response.usage.model_dump().items():
        print(f"  {field}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 5: Simple fields
    print("\nSimple Fields")
    print(f"id: {type(raw_response.id)} = {raw_response.id}")
    print(f"model: {type(raw_response.model)} = {raw_response.model}")
    print(f"created: {type(raw_response.created)} = {raw_response.created}")
    print(f"object: {type(raw_response.object)} = {raw_response.object}")

    print("\n" + "=" * 60)
    print("\nOpenAI raw exploration complete!")


def explore_anthropic_raw() -> None:
    """Explore raw Anthropic API response structure."""
    print_separator("RAW ANTHROPIC RESPONSE STRUCTURE")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = "Say love in 2 different languages."
    print(f"\nPrompt: {prompt}\n")

    print("Calling Anthropic API...")
    raw_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n" + "=" * 60)
    print("\nRAW RESPONSE OBJECT\n")
    print(f"Type: {type(raw_response)}")
    print(f"\nFull object response:\n{raw_response}\n")
    print("=" * 60)

    print("\nUNPACKING STRUCTURE\n")
    # print(dir(raw_response))  # Print all attribute names

    # Step 1: Top-level fields
    top_fields = list(raw_response.model_dump().keys())
    print(f"Top-level fields: {top_fields}")

    # Step 2: Field types
    response_dict = raw_response.model_dump()
    print("\nField Types")
    for key, value in response_dict.items():
        print(f"  {key}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 3: Dive into content
    print(f"\nDive into 'content':\n{raw_response.content}\n")
    print(f"content: {type(raw_response.content)} with {len(raw_response.content)} block(s)")

    if raw_response.content:
        first_block = raw_response.content[0]
        print(f"  content[0]: {type(first_block)}")
        print(f"  content[0] fields: {list(first_block.model_dump().keys())}")

        print(f"\n  Dive into content[0].text:")
        print(f"    type: {type(first_block.text)} = {first_block.text}")

    print("\n" + "=" * 60)

    # Step 4: Usage inspection
    print(f"\nDive into 'usage':\n{raw_response.usage}\n")
    print(f"usage: {type(raw_response.usage)}")
    print(f"usage fields: {list(raw_response.usage.model_dump().keys())}")

    for field, value in raw_response.usage.model_dump().items():
        print(f"  {field}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 5: Simple fields
    print("\nSimple Fields")
    print(f"id: {type(raw_response.id)} = {raw_response.id}")
    print(f"model: {type(raw_response.model)} = {raw_response.model}")
    print(f"role: {type(raw_response.role)} = {raw_response.role}")
    print(f"stop_reason: {type(raw_response.stop_reason)} = {raw_response.stop_reason}")

    print("\n" + "=" * 60)
    print("\nAnthropic raw exploration complete!")


# Note: Requires GPU with compute capability >= 7.0 (sm_70+, i.e., Volta/Ampere/Hopper) and will crash on older GPUs like GTX 1050 Ti (sm_61) due to Unsloth's Triton kernels
def explore_unsloth_raw() -> None:
    """Explore raw Unsloth/Transformers response structure."""
    print_separator("RAW UNSLOTH/TRANSFORMERS RESPONSE STRUCTURE")

    from unsloth import FastLanguageModel

    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    prompt = "Say love in 2 different languages."
    max_seq_length = 4096
    print(f"\nModel: {model_name}")
    print(f"\nPrompt: {prompt}")
    print("\n" + "=" * 60)

    # Step 0: Load model and tokenizer via Unsloth
    print("\nStep 0: LOADING MODEL VIA UNSLOTH")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"model type: {type(model)}")
    print(f"tokenizer type: {type(tokenizer)}")

    print("\n" + "=" * 60)

    print("\nUNPACKING STRUCTURE\n")

    # Step 1: Build chat messages and apply chat template
    print("Step 1: APPLY CHAT TEMPLATE")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    print(f"messages: {messages}")

    raw_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to("cuda")

    print(f"\nraw_inputs type: {type(raw_inputs)}")

    # apply_chat_template may return a raw tensor OR a BatchEncoding (dict-like) depending on tokenizer version
    if isinstance(raw_inputs, torch.Tensor):
        inputs = raw_inputs
    else:
        inputs = raw_inputs["input_ids"]

    print(f"inputs shape: {tuple(inputs.shape)}")
    print(f"inputs dtype: {inputs.dtype}")
    print(f"inputs device: {inputs.device}")
    print(f"inputs sample (first 16 tokens): {inputs[:, :min(16, inputs.shape[1])]}")

    input_length = inputs.shape[1]
    print(f"\ninput_length: {input_length}")

    print("\n" + "=" * 60)

    # Step 2: Generate output tensor
    print("\nStep 2: MODEL.GENERATE (RAW OUTPUT)")
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        use_cache=True,
    )

    print(f"\noutputs type: {type(outputs)}")
    print(f"outputs shape: {tuple(outputs.shape)}")
    print(f"outputs dtype: {outputs.dtype}")
    print(f"outputs device: {outputs.device}")
    print(f"outputs sample (first 16 tokens): {outputs[:, :min(16, outputs.shape[1])]}")

    print("\n" + "=" * 60)

    # Step 3: Decode and extract response
    print("\nStep 3: DECODE AND EXTRACT RESPONSE")
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"decoded type: {type(decoded)}")
    print(f"\n[DECODED FULL TEXT]\n{decoded}\n")

    # Extract response only (split on 'assistant' role marker)
    generated_text = decoded.split("assistant")[-1].strip()
    print("[RESPONSE ONLY]")
    print(generated_text)

    print("\n" + "=" * 60)

    # Step 4: Token counts
    print("\nStep 4: TOKEN COUNTS")
    input_tokens = input_length
    output_tokens = len(outputs[0]) - input_length
    print(f"input_tokens: {input_tokens}")
    print(f"output_tokens: {output_tokens}")
    print(f"total_tokens: {input_tokens + output_tokens}")

    print("\n" + "=" * 60)
    print("\nUnsloth raw exploration complete!")


def explore_all() -> None:
    """Run all explorations sequentially."""
    explore_openai_raw()
    explore_anthropic_raw()
    # explore_unsloth_raw()


# Run with: python -m scripts.sandbox.explore_models
if __name__ == "__main__":
    explore_all()