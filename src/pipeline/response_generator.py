"""Response generator for MLP4CS pipeline."""
from src.config import RESPGEN_INSTRUCTION, TARGET_DOMAINS
from src.db import find_entity, book_entity
from src.models.llm import call_model


def _format_db_results(db_results: list[dict], domain: str) -> str:
    """
    Format DB results into a readable string for the LLM prompt.

    Args:
        db_results: list of entity dicts from find_entity() or book_entity()
        domain: 'hotel' or 'restaurant'
    Returns:
        formatted string of entity attributes

    Example:
        Input: [{"name": "bedouin", "area": "centre", "food": "african", "phone": "01223367660"}]
        Output: "1. name=bedouin | area=centre | food=african | phone=01223367660"
    """
    if not db_results:
        return "no matching entities found"

    # Shared fields for both domains
    shared = ["name", "area", "pricerange", "phone", "address", "postcode"]

    # Domain-specific extra fields
    extra = {
        "hotel": ["type", "stars", "parking", "internet"],
        "restaurant": ["food"],
    }

    fields = shared + extra.get(domain, [])
    lines = []

    for i, entity in enumerate(db_results, 1):
        parts = [f"{f}={entity[f]}" for f in fields if f in entity]
        lines.append(f"{i}. {' | '.join(parts)}")

    return "\n".join(lines)


def build_respgen_prompt(history: list[dict], user_utterance: str, domain: str, intent: str | None, domain_slots: dict[str, str], db_results: list[dict], violations: list[str], zeroshot: bool = True, feedback: str | None = None) -> tuple[str, str]:
    """
    Build system and user prompts for the response generator LLM call.

    Args:
        history: list of previous turns as dicts with 'speaker' and 'utterance'
        user_utterance: current user message to keep LLM focused on what was just asked
        domain: active domain for this turn
        intent: active intent for this turn
        domain_slots: belief state filtered to active domain only
        db_results: list of matching entities from DB
        violations: list of missing required slot names from policy
        zeroshot: if True, include DB results, user message, and rules (used by both Exp2 inference and Exp3 training data build + inference). If False, return minimal history+slots format only (unused in current pipeline)
        feedback: supervisor feedback from previous attempt, None on first try
    Returns:
        tuple of (system_prompt, user_prompt)
    """
    from src.utils import format_history, format_slots

    history_str = format_history(history)
    slots_str = format_slots(domain_slots)
    intent_str = intent if intent else "unknown"

    # Base input mirrors fine-tuning format exactly
    user_prompt = (
        f"{history_str}\n"
        f"DOMAIN: {domain}\n"
        f"INTENT: {intent_str}\n"
        f"SLOTS: {slots_str}"
    )

    # Zero-shot only: add DB results and violations
    if zeroshot:
        db_str = _format_db_results(db_results, domain)
        user_prompt += f"\n\nDB results:\n{db_str}"

        if violations:
            missing = ", ".join(violations)
            user_prompt += f"\n\nMissing required information: {missing}. Ask the user for these."

        if feedback:
            user_prompt += f"\n\nPrevious response was rejected. Supervisor feedback: {feedback}"

        user_prompt += f"\n\nUser message: \"{user_utterance}\""

        user_prompt += (
            f"\nRules:\n"
            f"1. CRITICAL: NEVER use real entity names, addresses, phone numbers or any real values in your response.\n"
            f"   ALWAYS use ONLY these placeholders:\n"
            f"   [{domain}_name], [{domain}_phone], [{domain}_address], [{domain}_postcode], [ref]\n"
            f"2. If DB results are available, recommend the first entity using placeholders only.\n"
            f"3. If no DB results found, do not just say 'nothing found'. Ask user to relax one constraint (e.g. area, price range).\n"
            f"4. If booking reference [ref] is available, confirm the booking using [ref] placeholder.\n"
            f"5. If missing required information is listed, ask ONLY for those missing details.\n"
            f"6. Keep your response concise with 1-2 sentences maximum.\n"
        )

    return RESPGEN_INSTRUCTION, user_prompt


def response_generator(history: list[dict], user_utterance: str, domain: str | None, intent: str | None, accumulated_slots: dict[str, str], violations: list[str], model_config: dict[str, str], zeroshot: bool = True, feedback: str | None = None) -> tuple[str, list[dict], float, float]:
    """
    Generate a delexicalized response for the current turn.

    Args:
        history: list of previous turns as dicts with 'speaker' and 'utterance'
        user_utterance: current user message to keep LLM focused on what was just asked
        domain: active domain from DST, can be None
        intent: active intent from DST, can be None
        accumulated_slots: full belief state from DST
        violations: missing required slots from policy
        model_config: dict with key 'response_generator'
        zeroshot: if True, add DB results and violations to prompt (Exp2)
                  if False, mirror fine-tuning format exactly (Exp3)
        feedback: supervisor feedback from previous attempt, None on first try
    Returns:
        tuple of (delex_response, db_results, cost, response_time)

    Edge cases:
        1. domain is None: cannot query DB or build prompt. Returns hardcoded fallback response, empty db_results.
        2. violations exist: skip DB entirely, ask user for missing slots. Returns LLM response with violation guidance, empty db_results.
        3. booking intent, no violations, entity found: call book_entity(). Returns LLM response with [ref] placeholder, db_results=[entity].
        4. booking intent, no violations, no entity found: booking fails. Returns LLM response informing user, empty db_results.
        5. find intent, results found: call find_entity(). Returns LLM response with entity recommendations, db_results=matches.
        6. find intent, no results: call find_entity(), returns empty list. Returns LLM response asking user to relax constraints, empty db_results.
    """
    # print(f"\nHistory: {history} | Domain: {domain} | Intent: {intent} | Accumulated Slots: {accumulated_slots} | Violations: {violations} | Feedback: {feedback}")

    # Edge case 1: domain is None, cannot proceed
    if domain is None:
        return "I'm sorry, I didn't understand. Could you please clarify?", [], 0.0, 0.0

    # Filter accumulated slots to active domain only
    domain_slots = {k: v for k, v in accumulated_slots.items() if k.startswith(domain)}

    # Edge case 2: violations exist, skip DB, ask user for missing slots, see in build_respgen_prompt function
    if violations:
        db_results = []

    else:
        # Edge cases 3,4: booking intent, no violations
        if intent in ("book_hotel", "book_restaurant"):
            booking = book_entity(domain, domain_slots)
            if booking["success"]:
                db_results = [booking["entity"]]  # edge case 3: booking succeeded
            else:
                db_results = []  # edge case 4: no matching entity

        # Edge cases 5, 6: find intent or intent=None, no violations
        else:
            db_results = find_entity(domain, domain_slots)  # empty list = edge case 6

    # Build prompt and call LLM
    # print(f"\nDomain slots: {domain_slots} | DB results {len(db_results)}: {db_results}")
    system_prompt, user_prompt = build_respgen_prompt(
        history, user_utterance, domain, intent, domain_slots, db_results[:1], violations, zeroshot=zeroshot, feedback=feedback
    )  # show only first entity to LLM to reduce verbosity
    # print(f"\nSystem prompt: {system_prompt}\nUser prompt: {user_prompt}")

    response = call_model(model_config["response_generator"], user_prompt, system_prompt)
    # print(f"\nRaw Response Generator LLM output:\n {response}")

    delex_response = response.text.strip()
    # print(f"\nDelex response: {delex_response}")

    return delex_response, db_results, response.cost, response.response_time

