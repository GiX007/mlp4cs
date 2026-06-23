"""Rule-based response validator for MLP4CS pipeline."""

# Farewell keywords
FAREWELL_KEYWORDS = {"bye", "goodbye", "thank you", "thanks", "that's all", "good bye"}


def supervisor(delex_response: str, violations: list[str], db_results: list[dict], intent: str | None, user_utterance: str, domain: str | None) -> tuple[bool, str | None]:
    """
    Validate the response generator output using rule-based checks.

    Args:
        delex_response: delexicalized response from response_generator
        violations: missing required slots from policy
        db_results: matching entities from DB lookup
        intent: active intent for this turn
        user_utterance: current user message
        domain: active domain for this turn
    Returns:
        tuple of (valid, feedback): valid=False triggers retry in runner.py

    Checks:
        0. Farewell turn: any farewell keyword in user utterance → always valid, skip all checks.
        1. Response not empty: delex_response must contain text. Feedback: "Response is empty. Generate a response."
        2. Hallucination: real entity name from db_results appears literally in delex_response. Feedback: "Response contains real entity name '{name}'. Use [{domain}_name] instead."
        3. Booking reference: booking intent, no violations, db_results non-empty → [ref] must appear. Feedback: "Booking succeeded but [ref] placeholder is missing from response."
        4. Policy compliance: violations exist → response must not be empty. Covered by Check 1.

    Example of a hallucination retry:

    User: "I want a cheap restaurant in the centre."
    DST: domain=restaurant, intent=find_restaurant, slots={area: centre, price_range: cheap} -> DB returns: [{name: "the missing sock", ...}]

    Attempt 1 -> ResponseGen output: "I found The Missing Sock, a cheap restaurant in the centre."
    Supervisor: rejected. Feedback: "Response contains real entity name 'the missing sock'. Use [restaurant_name] instead."

    Attempt 2 -> ResponseGen output: "I found [restaurant_name], a cheap restaurant in the [restaurant_area]."
    Supervisor: valid. Forwarded to lexicalize.
    """
    # Check 0: farewell turn, always valid
    if any(kw in user_utterance.lower() for kw in FAREWELL_KEYWORDS):
        return True, None

    # Check 1: response not empty
    if not delex_response.strip():
        return False, "Response is empty. Generate a response."

    # Check 2: hallucination, real entity name in delex response
    for entity in db_results:
        name = entity.get("name", "").lower()
        if name and name in delex_response.lower():
            return False, f"Response contains real entity name '{name}'. Use [{domain}_name] instead."

    # Check 3: booking ref missing
    if intent in ("book_hotel", "book_restaurant") and not violations and db_results:
        if "[ref]" not in delex_response:
            return False, "Booking succeeded but [ref] placeholder is missing from response."

    return True, None
