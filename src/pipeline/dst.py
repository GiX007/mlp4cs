"""Dialogue State Tracker for MLP4CS pipeline."""
from src.utils import format_history, normalize_slot_value
from src.config import DST_INSTRUCTION, TARGET_DOMAINS
from src.models.llm import call_model


# Valid slots per domain that guides zero-shot model, mirrors fine-tuning data
DST_SLOTS = {
    "hotel": [
        "hotel-area", "hotel-pricerange", "hotel-type", "hotel-stars", "hotel-internet",
        "hotel-parking", "hotel-name", "hotel-bookday", "hotel-bookpeople", "hotel-bookstay",
    ],
    "restaurant": [
        "restaurant-area", "restaurant-pricerange", "restaurant-food", "restaurant-name",
        "restaurant-bookday", "restaurant-bookpeople", "restaurant-booktime",
    ],
}

DST_INTENTS = ["find_hotel", "book_hotel", "find_restaurant", "book_restaurant"]


def build_dst_prompt(user_utterance: str, history: list[dict]) -> tuple[str, str]:
    """
    Build system and user prompts for the DST LLM call. Same format used in fine-tuning data so Exp2 and Exp3 produce identical output format.

    Args:
        user_utterance: latest user message
        history: list of previous turns as dicts with 'speaker' and 'utterance'
    Returns:
        tuple of (system_prompt, user_prompt)
    """
    history_str = format_history(history)
    slot_list = ", ".join(DST_SLOTS["hotel"] + DST_SLOTS["restaurant"])
    domain_list = ", ".join(TARGET_DOMAINS)
    intent_list = ", ".join(DST_INTENTS)

    user_prompt = (
        f"{history_str}\nUSER: {user_utterance}\n\n"
        f"Valid domains: {domain_list}\n"
        f"Valid intents: {intent_list}\n"
        f"Valid slots: {slot_list}\n\n"
        
        # Chain-of-thought reasoning to force structured thinking before output
        f"Think step by step:\n"
        f"1. What is the user asking about? (domain)\n"
        f"2. What action do they want: find or book? (intent)\n"
        f"3. What specific values did they explicitly mention? (slots)\n\n"
        f"Then output ONLY:\n"
        f"DOMAIN: <domain>\n"
        f"INTENT: <intent>\n"
        f"SLOTS: <slot1=value1, slot2=value2> or SLOTS: none\n\n"

        f"\nRules:\n"
        f"- ONLY extract slot values EXPLICITLY stated by the user.\n"
        f"- If user expresses no preference for a specific attribute (e.g. 'any food', 'no particular type', 'doesn't matter') → identify WHICH slot they refer to and extract slot=dontcare.\n"
        f"  Example: 'no particular food type' → restaurant-food=dontcare\n"
        f"  Example: 'any area is fine' → hotel-area=dontcare\n"
        f"  Example: 'no preference on price' → restaurant-pricerange=dontcare\n"
        f"  Example: 'No particular type of food' → restaurant-food=dontcare\n"
        f"  Example: 'any hotel type is fine' → hotel-type=dontcare\n"
        f"- When user mentions BOTH a preference AND a non-preference in the same sentence, extract BOTH.\n"
        f"  Example: 'No particular food type but moderate price' → restaurant-food=dontcare, restaurant-pricerange=moderate\n"
        f"  Example: 'Any area is fine but I need cheap price' → restaurant-area=dontcare, restaurant-pricerange=cheap\n"
        f"- Valid area values: north, south, east, west, centre, dontcare. DO NOT extract city names as area values.\n"
        f"- Only extract a slot when user is PROVIDING information, not REQUESTING it.\n"
        f"  Example: 'what price range are they?' → do NOT extract pricerange.\n"
        f"  Example: 'I want a cheap restaurant' → extract restaurant-pricerange=cheap.\n"
        f"- If user references previous context without mentioning a domain explicitly (e.g. 'same group', 'same day', 'same people', 'it', 'there', 'that one') → keep the PREVIOUS domain from conversation history.\n"
        f"- DO NOT infer or assume missing values.\n\n"
    )
    return DST_INSTRUCTION, user_prompt


def parse_dst_output(raw: str, accumulated_slots: dict[str, str]) -> tuple[str | None, str | None, dict[str, str]]:
    """
    Parse raw LLM output into domain, intent, and merged belief state.

    Args:
        raw: raw LLM response string
        accumulated_slots: belief state built up across all previous turns
    Returns:
        tuple of (domain, intent, slots), domain and intent are None if parsing fails

    Edge cases:
    1. domain and intent both valid, matching: normal case, use both as-is
    2. domain valid, intent None (LLM output multiple or invalid values): use domain, intent=None. Policy skipped for this turn
    3. domain None, intent valid (LLM output multiple or invalid domain): infer domain from intent as "book_hotel" → "hotel"
    4. domain None, intent None (both multiple or invalid): both returned as None. Slots still accumulated correctly
    5. domain valid, intent valid but mismatched (e.g. domain=hotel, intent=find_restaurant): trust domain, set intent=None. Policy skipped for this turn

    In all cases, slots are accumulated correctly regardless of domain/intent outcome.
    """
    domain = None
    intent = None
    new_slots: dict[str, str] = {}

    for line in raw.strip().splitlines():
        line = line.strip()

        if line.startswith("DOMAIN:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in TARGET_DOMAINS:
                domain = value

        elif line.startswith("INTENT:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in DST_INTENTS:
                intent = value

        elif line.startswith("SLOTS:"):
            value = line.split(":", 1)[1].strip()
            if value.lower() != "none":
                for pair in value.split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        k, v = k.strip(), normalize_slot_value(v.strip().lower())
                        if k in ("hotel-bookstay", "hotel-bookpeople", "restaurant-bookpeople", "restaurant-bookstay"):
                            # v = v.split()[0]  # "2 nights" → "2"
                            parts = v.split()
                            v = parts[0] if parts else v

                        # Slot-specific normalization
                        if k in ("hotel-parking", "restaurant-parking", "hotel-internet") and v in ["free", "free wifi", "free internet"]:
                            v = "yes"

                        if k and v:
                            new_slots[k] = v

    # Edge case 3: domain None, intent valid → infer domain from intent
    if domain is None and intent is not None:
        domain = intent.split("_")[1]  # "book_hotel" → "hotel"

    # Edge case 5: domain and intent mismatch → trust domain, discard intent
    if domain is not None and intent is not None:
        if not intent.endswith(domain):
            intent = None

    # Strip booking slots during find_ turns, GT does not have them yet
    if intent and intent.startswith("find_"):
        booking_keys = {"bookday", "bookpeople", "booktime", "bookstay"}
        new_slots = {k: v for k, v in new_slots.items() if k.split("-")[-1] not in booking_keys}

    # Safety net: keep confirmed slots even if LLM forgets them
    merged = {**accumulated_slots, **new_slots}
    return domain, intent, merged


def dst(user_utterance: str, history: list[dict], accumulated_slots: dict[str, str], model_config: dict[str, str]) -> tuple[str | None, str | None, dict[str, str], float, float]:
    """
    Run the full DST step: build prompt, call LLM, parse output.

    Args:
        user_utterance: latest user message
        history: list of previous turns as dicts with 'speaker' and 'utterance'
        accumulated_slots: belief state built up across all previous turns
        model_config: dict with key 'dst' pointing to the model name or path
    Returns:
        tuple of (domain, intent, slots, cost, response_time)
    """
    # print(f"\nUser utterance: {user_utterance}\nHistory: {history}")

    system_prompt, user_prompt = build_dst_prompt(user_utterance, history)
    # print(f"\nSystem prompt: {system_prompt}\nUser prompt: {user_prompt}")

    response = call_model(model_config["dst"], user_prompt, system_prompt)
    # print(f"\nRaw DST LLM output:\n {response}")

    domain, intent, slots = parse_dst_output(response.text, accumulated_slots)
    # print(f"\nParsed Output:\nDomain: {domain} | Intent: {intent} | Slots: {slots}")

    return domain, intent, slots, response.cost, response.response_time
