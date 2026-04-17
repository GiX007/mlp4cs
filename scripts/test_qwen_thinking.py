"""
Test the thinking mechanism of Qwen3-14B.
The thinking mechanism is designed to enhance the model's reasoning capabilities by allowing it to generate intermediate thoughts before producing the final answer.
"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "data/models/qwen3_14b",
    max_seq_length=4096,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

msgs = [{"role": "user", "content": "What is 2+2? Explain briefly."}]

# Thinking ON (default)
ids_on = tokenizer.apply_chat_template(
    msgs,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

out_on = model.generate(input_ids=ids_on, max_new_tokens=200, use_cache=True)
print(">>> Thinking ON")
print(tokenizer.decode(out_on[0], skip_special_tokens=False))

# Thinking OFF
ids_off = tokenizer.apply_chat_template(
    msgs,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,
).to("cuda")

out_off = model.generate(input_ids=ids_off, max_new_tokens=200, use_cache=True)
print("\n>>> Thinking OFF")
print(tokenizer.decode(out_off[0], skip_special_tokens=False))
