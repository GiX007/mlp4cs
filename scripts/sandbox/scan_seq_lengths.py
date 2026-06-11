"""
Reconstruct the exact Exp1/Exp2/Exp3 prompts for every saved turn, tokenize them, and report overflows vs LOCAL_MAX_SEQ_LENGTH.

Context window basics: The model processes ONE continuous token sequence per forward pass: prompt + generated response. Both count toward the cap:
    - peak_seq_len = prompt_tokens + max_new_tokens
    - peak_seq_len <= LOCAL_MAX_SEQ_LENGTH <= model_context_window,

where prompt_tokens covers system + user + history + chat-template special tokens, and (for Qwen3 thinking mode) max_new_tokens also includes <think> trace tokens.
This script measures prompt_tokens, so the peak is then prompt + max_new_tokens (=512 here). Overflow at either side means silent truncation.

Note: API-based runs (gpt-4o-mini, gpt-4.1-nano, claude-haiku-4-5) are not scanned. Their context windows are far larger than any prompt we send (gpt-4o-mini: 128K tokens, gpt-4.1-nano: ~1M tokens,
claude-haiku-4-5: 200K tokens), so truncation is impossible given our worst-case prompt of ~25K tokens. They also lack a user-set LOCAL_MAX_SEQ_LENGTH to compare against.
"""
import json
from pathlib import Path
from transformers import AutoTokenizer

from src.config import TARGET_DOMAINS, BOOKING_REQUIRED_SLOTS
from src.pipeline.dst import DST_SLOTS
from src.db import load_db
from src.utils import format_history, format_slots

from src.pipeline.dst import build_dst_prompt
from src.pipeline.response_generator import build_respgen_prompt
from src.db import find_entity, book_entity


_HOTEL_DB = load_db("hotel")
_RESTAURANT_DB = load_db("restaurant")


def build_exp1_messages(user_utterance: str, history: list[dict], slots: dict) -> list[dict]:
    """
    Purpose: Recreate the exact (system, user) message pair from run_turn_single.

    Args:
        user_utterance: saved 'user_utterance' for this turn
        history: list of prior {speaker, utterance} dicts for this dialogue
        slots: saved 'slots' (belief state) for this turn
    Return: list of role/content dicts ready for apply_chat_template
    """
    system_prompt = (
        f"You are an end-to-end customer service assistant for {', '.join(TARGET_DOMAINS)} bookings.\n\n"
        f"HOTEL DATABASE:\n{json.dumps(_HOTEL_DB, indent=2)}\n\n"
        f"RESTAURANT DATABASE:\n{json.dumps(_RESTAURANT_DB, indent=2)}"
    )

    history_str = format_history(history)
    slots_str = format_slots(slots)
    slot_list = ", ".join(DST_SLOTS["hotel"] + DST_SLOTS["restaurant"])

    user_prompt = (
        f"{history_str}\n"
        f"USER: {user_utterance}\n\n"
        f"Current belief state: {slots_str}\n\n"
        f"Valid domains: {', '.join(TARGET_DOMAINS)}\n"
        f"Valid intents: find_hotel, book_hotel, find_restaurant, book_restaurant\n"
        f"Valid slots: {slot_list}\n\n"
        f"Booking policy — required slots:\n"
        f"- book_hotel: {', '.join(BOOKING_REQUIRED_SLOTS['book_hotel'])}\n"
        f"- book_restaurant: {', '.join(BOOKING_REQUIRED_SLOTS['book_restaurant'])}\n\n"
        f"Extraction rules:\n"
        f"- ONLY extract slot values EXPLICITLY stated by the user.\n"
        f"- DO NOT infer or assume missing values.\n\n"
        f"Response rules:\n"
        f"1. Find the best matching entity from the DATABASE in the system prompt based on the current belief state constraints.\n"
        f"2. If no matching entity found, tell user nothing was found.\n"
        f"3. If booking intent and ALL required slots present, confirm booking with [ref] placeholder.\n"
        f"4. If booking intent but required slots MISSING, ask ONLY for those missing slots.\n"
        f"5. NEVER use real entity names or details — ALWAYS use placeholders:\n"
        f"   [hotel_name], [hotel_phone], [hotel_address], [hotel_postcode]\n"
        f"   [restaurant_name], [restaurant_phone], [restaurant_address], [restaurant_postcode]\n"
        f"   [ref] for booking reference.\n\n"
        # f"   DO NOT invent placeholders like [hotel_bookday], [food_type], [price_range], [restaurant_signature], etc.\n"
        # f"   For booking attributes (day, time, people, stay), refer to them in plain text\n"
        # f"   (e.g., 'for monday at 19:00 for 2 people') — NOT as placeholders.\n\n"
        f"6. Recommend ONE entity only and use each placeholder ONCE.\n"
        f"7. If user says goodbye or thanks, respond with a farewell message. Set intent to None.\n"
        f"Respond with valid JSON only:\n"
        f'{{"domain": "...", "intent": "...", "slots": {{}}, "response": "..."}}'
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def scan_run(model_name: str, results_path: str, cap: int) -> dict:
    """
    Purpose: For one saved results file, reconstruct each turn's prompt and count overflows.

    Args:
        model_name: HuggingFace tokenizer ID matching the run
        results_path: path to the saved JSON with 'turns' list
        cap: LOCAL_MAX_SEQ_LENGTH used during the run

    Return: dict with max_len, overflow_count, total_turns, and a few example overflows
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(results_path) as f:
        data = json.load(f)

    # Group turns by dialogue so we can rebuild history turn-by-turn
    by_dialogue: dict[str, list[dict]] = {}
    for t in data["turns"]:
        by_dialogue.setdefault(t["dialogue_id"], []).append(t)

    max_len = 0
    overflow_count = 0
    total = 0
    examples: list[dict] = []

    for dialogue_id, turns in by_dialogue.items():
        history: list[dict] = []
        for t in turns:
            messages = build_exp1_messages(t["user_utterance"], history, t["predicted_slots"])
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            n = len(ids)

            total += 1
            max_len = max(max_len, n)
            if n > cap:
                overflow_count += 1
                if len(examples) < 3:
                    examples.append({"dialogue_id": dialogue_id, "tokens": n, "over_by": n - cap})

            # Update history exactly like memory() does in the runner
            history.append({"speaker": "USER", "utterance": t["user_utterance"]})
            history.append({"speaker": "SYSTEM", "utterance": t["lex_response"]})

    return {"max_len": max_len, "overflow_count": overflow_count, "total_turns": total, "examples": examples}


def scan_run_exp2(model_name: str, results_path: str, cap: int) -> dict:
    """
    Purpose: For one saved Exp2 results file, reconstruct DST + RespGen prompts per turn and count overflows against the cap.
    Args:
        model_name: HuggingFace tokenizer ID
        results_path: path to the saved Exp2 JSON with 'turns' list
        cap: LOCAL_MAX_SEQ_LENGTH used during the run (2048)

    Return: dict with max_len, overflow_count, total_turns (counts BOTH calls per turn), examples
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    by_dialogue: dict[str, list[dict]] = {}
    for t in data["turns"]:
        by_dialogue.setdefault(t["dialogue_id"], []).append(t)

    max_len = 0
    overflow_count = 0
    total = 0
    examples: list[dict] = []

    for dialogue_id, turns in by_dialogue.items():
        history: list[dict] = []
        for t in turns:
            user_utt = t["user_utterance"]
            domain = t.get("predicted_domain")
            intent = t.get("predicted_intent")
            slots = t.get("predicted_slots", {})
            violations = t.get("violations", [])

            # Call 1: DST prompt
            dst_sys, dst_user = build_dst_prompt(user_utt, history)
            dst_msgs = [{"role": "system", "content": dst_sys}, {"role": "user", "content": dst_user}]
            dst_rendered = tokenizer.apply_chat_template(dst_msgs, tokenize=False, add_generation_prompt=True)
            dst_n = len(tokenizer.encode(dst_rendered, add_special_tokens=False))

            # Call 2: ResponseGen prompt
            domain_slots = {k: v for k, v in slots.items() if domain and k.startswith(domain)}
            db_results = t.get("db_results", [])  # use what was actually retrieved during the run
            rg_sys, rg_user = build_respgen_prompt(
                history, user_utt, domain, intent, domain_slots, db_results, violations, zeroshot=True
            )
            rg_msgs = [{"role": "system", "content": rg_sys}, {"role": "user", "content": rg_user}]
            rg_rendered = tokenizer.apply_chat_template(rg_msgs, tokenize=False, add_generation_prompt=True)
            rg_n = len(tokenizer.encode(rg_rendered, add_special_tokens=False))

            # Track each call as its own data point against the cap
            for call_name, n in [("dst", dst_n), ("rg", rg_n)]:
                total += 1
                max_len = max(max_len, n)
                if n > cap:
                    overflow_count += 1
                    if len(examples) < 3:
                        examples.append({
                            "dialogue_id": dialogue_id, "call": call_name,
                            "tokens": n, "over_by": n - cap,
                        })

            history.append({"speaker": "USER", "utterance": user_utt})
            history.append({"speaker": "SYSTEM", "utterance": t["lex_response"]})

    return {"max_len": max_len, "overflow_count": overflow_count, "total_turns": total, "examples": examples}


RUNS = {
    # Exp1: single-LLM, cap 32768
    "exp1_qwen3_8b": (
        "unsloth/Qwen3-8B-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp1_qwen3_8b\overall\exp1_qwen3_8b_test_20260428_162751_turns.json",
        32768,
    ),
    "exp1_llama31_8b": (
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp1_llama31_8b\overall\exp1_llama31_8b_test_20260430_095359_turns.json",
        32768,
    ),
    "exp1_qwen25_14b": (
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp1_qwen25_14b\overall\exp1_qwen25_14b_test_20260428_204345_turns.json",
        32768,
    ),
    "exp1_qwen3_14b": (
        "unsloth/Qwen3-14B-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp1_qwen3_14b\overall\exp1_qwen3_14b_test_20260419_095454_turns.json",
        32768,
    ),

    # Exp2: two-agent zero-shot, cap 2048
    "exp2_homo_qwen3_8b": (
        "unsloth/Qwen3-8B-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp2_homo_qwen3_8b\overall\exp2_homo_qwen3_8b_test_20260429_051651_turns.json",
        2048,
    ),
    "exp2_homo_qwen3_14b": (
        "unsloth/Qwen3-14B-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp2_homo_qwen3_14b\overall\exp2_homo_qwen3_14b_test_20260428_214220_turns.json",
        2048,
    ),
    "exp2_hetero_qwen25_qwen3_14b": (
        # heterogeneous: DST=qwen25_14b, RG=qwen3_14b. Scanned with DST tokenizer only as RG tokenizer counts may differ by a few tokens (same family, short prompts)
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp2_hetero_qwen25_qwen3_14b\overall\exp2_hetero_qwen25_qwen3_14b_test_20260429_042215_turns.json",
        2048,
    ),
    "exp2_hetero_qwen3_14b_qwen3_8b": (
        "unsloth/Qwen3-14B-bnb-4bit",
        r"archived_results\open_source_test_20260418\exp2_hetero_qwen3_14b_qwen3_8b\overall\exp2_hetero_qwen3_14b_qwen3_8b_test_20260429_132629_turns.json",
        2048,
    ),
}


def main() -> None:
    """Run scan_run for every configured experiment and print a summary."""
    for run_name, (model_name, path, cap) in RUNS.items():
        if not Path(path).exists():
            print(f"{run_name}: SKIP (not found: {path})")
            continue

        # Pick the right scanner based on run name
        if run_name.startswith("exp1_"):
            r = scan_run(model_name, path, cap)
        elif run_name.startswith("exp2_"):
            r = scan_run_exp2(model_name, path, cap)
        else:
            print(f"{run_name}: SKIP (no scanner for this experiment)")
            continue

        pct = 100 * r["overflow_count"] / r["total_turns"]
        print(f"\n{run_name}")
        print(f"  total turns: {r['total_turns']}")
        print(f"  max prompt: {r['max_len']} tokens (cap {cap})")
        print(f"  overflows: {r['overflow_count']} ({pct:.1f}%)")
        for ex in r["examples"]:
            print(f"    {ex['dialogue_id']}: {ex['tokens']} tokens (over by {ex['over_by']})")


# Run with: python -m scripts.sandbox.scan_seq_lengths
if __name__ == "__main__":
    main()
