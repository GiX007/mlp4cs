"""Shared utility functions for MLP4CS."""
from pathlib import Path
from src.config import SLOT_VALUE_NORMALIZATION, MODEL_COSTS


def print_separator(message: str) -> None:
    """
    Print a labeled separator for readable console output.

    Args:
        message: title to display between separator lines
    """
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)


def capture_and_save(func, output_path: str) -> None:
    """
    Run func(), print to terminal in real time, and save to file.

    Args:
        func: callable that prints output
        output_path: path to save the output .txt file
    """
    import sys

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        class Tee:
            @staticmethod
            def write(msg):
                sys.__stdout__.write(msg)
                f.write(msg)

            @staticmethod
            def flush():
                sys.__stdout__.flush()
                f.flush()

        sys.stdout = Tee()
        try:
            func()
        finally:
            sys.stdout = sys.__stdout__

    print(f"\nSaved to {output_path}")


def format_slots(slot_values: dict) -> str:
    """
    Convert slot values dict to a flat string for LLM prompts and fine-tuning output.

    Args:
        slot_values: dict like {"restaurant-area": ["centre"], "restaurant-pricerange": ["expensive"]}
                     or already flattened {"restaurant-area": "centre"} from pipeline accumulation
    Returns:
        string like "restaurant-area=centre, restaurant-pricerange=expensive"

    Note on multi-value lists:
        MultiWOZ stores spelling variants of the same value as a list, e.g., ["rosas bed and breakfast", "rosa's"] or ["17:45", "5:45 pm"].
        These are NOT conflicting preferences as they refer to the same entity. We always take v[0] (canonical form).
        This affects ~1.7% of slots across all splits and has negligible impact on metrics.
    """
    if not slot_values:
        return "none"
    parts = []
    for k, v in slot_values.items():
        value = v[0] if isinstance(v, list) else v
        if value:
            parts.append(f"{k}={value}")
    return ", ".join(parts)


def format_history(turns: list[dict]) -> str:
    """
    Format a list of previous turns into a conversation history string.

    Args:
        turns: list of turn dicts with 'speaker' and 'utterance' keys
    Returns:
        formatted history string, empty string if no turns
    """
    if not turns:
        return ""
    lines = [f"{t['speaker']}: {t['utterance']}" for t in turns]
    return "<history>\n" + "\n".join(lines) + "\n</history>"


def normalize_slot_value(value: str) -> str:
    """
    Normalize a slot value for consistent DB matching and metric computation.

    Lowercases, strips whitespace, then applies canonical replacements defined in SLOT_VALUE_NORMALIZATION.

    Args:
        value: raw slot value string
    Returns:
        normalized string

    Example:
        "Center" → "centre", "don't care" → "dontcare", "B&B" → "bed and breakfast"
    """
    value = value.lower().strip()
    for raw, normalized in SLOT_VALUE_NORMALIZATION.items():
        if value == raw:
            return normalized
    return value


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate API cost in USD given model name and token counts.

    Args:
        model_name: Model identifier, must match a key in MODEL_COSTS
        input_tokens: Number of prompt tokens used
        output_tokens: Number of completion tokens generated

    Returns:
        Cost in USD as a float. Returns 0.0 for local/unknown models
    """
    if model_name not in MODEL_COSTS:
        return 0.0
    price_in, price_out = MODEL_COSTS[model_name]
    return (input_tokens / 1000 * price_in) + (output_tokens / 1000 * price_out)


def build_tomiinek_turn(delex_response: str, accumulated_slots: dict[str, str], domain: str | None) -> dict:
    """
    Build a single turn dict in Tomiinek evaluator format.

    Args:
        delex_response: delexicalized system response
        accumulated_slots: full belief state with prefixed keys e.g., {"hotel-area": "north"}
        domain: active domain for this turn, can be None
    Returns:
        dict with keys: response, state, active_domains
    """
    from src.config import TOMIINEK_SLOT_MAP, TARGET_DOMAINS
    from src.pipeline.dst import DST_SLOTS

    # Valid slot names without domain prefix e.g., {"area", "pricerange", "booking-day", ...}
    valid_slot_names = {s.split("-", 1)[1] for slots in DST_SLOTS.values() for s in slots}
    valid_slot_names.update(TOMIINEK_SLOT_MAP.values())

    # Convert prefixed slots to nested Tomiinek format: {"hotel-bookday": "friday"} → {"hotel": {"booking-day": "friday"}}
    state: dict[str, dict[str, str]] = {}
    for key, value in accumulated_slots.items():
        key = key.replace("_", "-")  # Normalize underscore: restaurant_bookday → restaurant-bookday
        if "-" in key:
            d, slot = key.split("-", 1)
            if d not in TARGET_DOMAINS:  # Skip non-domain keys
                continue
            slot = TOMIINEK_SLOT_MAP.get(slot, slot)  # Remap booking slots
            if slot not in valid_slot_names:  # Drop hallucinated slot names not in schema, e.g., price_double, total_cost
                continue
            if d not in state:
                state[d] = {}
            state[d][slot] = value

    return {
        "response": delex_response,
        "state": state,
        "active_domains": [domain] if domain else [],
    }


def build_custom_turn(
    domain: str | None, intent: str | None, accumulated_slots: dict[str, str], violations: list[str], delex_response: str, lex_response: str,
    db_results: list[dict], valid: bool, attempts: int, user_utterance: str, cost: float, response_time: float,
) -> dict:
    """
    Build a single turn dict for custom metric evaluation.

    Args:
        domain: active domain from DST
        intent: active intent from DST
        accumulated_slots: full belief state from DST
        violations: missing required slots from policy
        delex_response: delexicalized response from response_generator
        lex_response: lexicalized response from lexicalizer
        db_results: matching entities from DB lookup
        valid: whether supervisor approved the response
        attempts: number of retry attempts made
        user_utterance: current user message
        cost: total API cost for this turn in USD
        response_time: total response time for this turn in seconds
    Returns:
        dict with all turn-level fields needed for custom metrics
    """
    return {
        "domain": domain,
        "intent": intent,
        "slots": accumulated_slots.copy(),
        "violations": violations,
        "delex_response": delex_response,
        "lex_response": lex_response,
        "db_results": db_results,
        "valid": valid,
        "attempts": attempts,
        "user_utterance": user_utterance,
        "cost": cost,
        "response_time": response_time,
    }
