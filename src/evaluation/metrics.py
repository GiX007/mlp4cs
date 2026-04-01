"""Custom evaluation metrics for MLP4CS pipeline."""


def calculate_domain_accuracy(predicted: str | None, gt: set[str]) -> tuple[bool, float, float, float]:
    """
    Calculate domain accuracy between predicted and ground truth domains.

    Args:
        predicted: single domain from dst(), can be None
        gt: set of active GT domains from dataset e.g., {"restaurant", "hotel"}
    Returns:
        tuple of (exact_match, precision, recall, f1)
    """
    if predicted is None or not gt:
        return False, 0.0, 0.0, 0.0

    pred_set = {predicted.lower()}
    gt_set = {d.lower() for d in gt}

    exact = pred_set == gt_set
    correct = len(pred_set & gt_set)
    precision = correct / len(pred_set)
    recall = correct / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return exact, precision, recall, f1


def calculate_intent_accuracy(predicted: str | None, gt: set[str]) -> tuple[bool, float, float, float]:
    """
    Calculate intent accuracy between predicted and ground truth intents.

    Args:
        predicted: single intent from dst(), can be None
        gt: set of active GT intents from dataset e.g., {"find_restaurant", "find_hotel"}
    Returns:
        tuple of (exact_match, precision, recall, f1)
    """
    if predicted is None or not gt:
        return False, 0.0, 0.0, 0.0

    pred_set = {predicted.lower()}
    gt_set = {i.lower() for i in gt}

    exact = pred_set == gt_set
    correct = len(pred_set & gt_set)
    precision = correct / len(pred_set)
    recall = correct / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return exact, precision, recall, f1


def calculate_jga(predicted_slots: dict[str, str], gt_slots: dict[str, str]) -> bool:
    """
    Check if predicted belief state exactly matches ground truth.

    JGA = True only if every slot-value pair matches exactly.

    Args:
        predicted_slots: flat prefixed dict e.g., {"restaurant-area": "centre"}
        gt_slots: flat prefixed dict normalized from dataset e.g., {"restaurant-area": "centre"}
    Returns:
        True if full belief state matches, False otherwise
    """
    return predicted_slots == gt_slots


def calculate_slot_f1(predicted_slots: dict[str, str], gt_slots: dict[str, str]) -> tuple[float, float, float]:
    """
    Calculate slot-level precision, recall and F1.

    Args:
        predicted_slots: flat prefixed dict e.g., {"restaurant-area": "centre"}
        gt_slots: flat prefixed dict normalized from dataset e.g., {"restaurant-area": "centre"}
    Returns:
        tuple of (precision, recall, f1)
    """
    # {"restaurant-area": "centre"} → {("restaurant-area", "centre"), ...}
    pred_set = {(k, v) for k, v in predicted_slots.items() if v}
    gt_set = {(k, v) for k, v in gt_slots.items()}

    correct = len(pred_set & gt_set)
    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def calculate_hallucination(lex_response: str, db_results: list[dict]) -> tuple[bool, bool]:
    """
    Check if lexicalized response hallucinates an entity not returned by DB.

    Args:
        lex_response: lexicalized system response for this turn
        db_results: entities returned by DB for this turn from custom_turn["db_results"]
    Returns:
        tuple of (hallucinated, entity_mentioned)
        hallucinated: True if response mentions an entity not in db_results
        entity_mentioned: True if any known entity name appears in response
    """
    from src.db import load_db

    response_lower = lex_response.lower()

    # Load all known entity names from both domains
    all_known = []
    for domain in ["hotel", "restaurant"]:
        db = load_db(domain)
        all_known.extend([e["name"].lower() for e in db if "name" in e])

    # Check which known names appear in response
    mentioned = [name for name in all_known if name in response_lower]

    if not mentioned:
        return False, False  # No entity mentioned, skip this turn for hall rate

    valid_names = {e["name"].lower() for e in db_results if "name" in e}

    # Hallucination: mentioned name not in valid DB results for this turn
    hallucinated = any(name not in valid_names for name in mentioned)

    return hallucinated, True


def calculate_booking_success(custom_turns: list[dict]) -> bool:
    """
    Calculate booking success rate for a dialogue.

    Args:
        custom_turns: list of custom_turn dicts for the full dialogue
    Returns:
        fraction of booking attempts that were successfully confirmed, None if no booking was attempted.
    """
    booking_attempts = [t for t in custom_turns if t["intent"] in ("book_hotel", "book_restaurant")]
    if not booking_attempts:
        return None  # No booking attempted, exclude from metric
    successful = sum(1 for t in booking_attempts if not t["violations"] and "[ref]" in t.get("delex_response", "").lower())
    return successful / len(booking_attempts)


def calculate_system_correctness(hallucinated: bool, policy_compliant: bool) -> bool:
    """
    Check if system responded correctly for a single turn.

    A turn is correct only if no hallucination occurred AND policy was followed.

    Args:
        hallucinated: True if hallucination detected this turn
        policy_compliant: True if policy was followed this turn
    Returns:
        True if system response was correct, False otherwise
    """
    if hallucinated or not policy_compliant:
        return False
    return True


def calculate_action_accuracy(intent: str | None, violations: list[str], dialog_act: dict) -> tuple[bool | None, str | None, str | None]:
    """
    Check if predicted action type matches ground truth action type.

    Predicted action mapping:
        find_hotel / find_restaurant → "inform"
        book_hotel / book_restaurant + no violations → "book"
        book_hotel / book_restaurant + violations exist → "request"
        None → None (skip turn)

    Args:
        intent: active intent from dst(), can be None
        violations: missing required slots from policy
        dialog_act: dialog_act dict from SYSTEM turn e.g., {"Restaurant-Inform": [...]}
    Returns:
        tuple of (correct, predicted_action, gt_action)
        correct is None if either action is undetermined
    """
    # Step 1: derive predicted action from intent and violations
    predicted = None
    if intent is not None:
        if intent.startswith("find_"):
            predicted = "inform"
        elif intent.startswith("book_") and violations:
            predicted = "request"
        elif intent.startswith("book_") and not violations:
            predicted = "book"

    # Step 2: derive GT action from SYSTEM turn dialog_act keys
    gt = None
    for act_key in dialog_act:
        act_lower = act_key.lower()
        if "booking-book" in act_lower:
            gt = "book"
            break
        if "request" in act_lower:
            gt = "request"
            break
        if any(x in act_lower for x in ("inform", "recommend", "select", "booking-inform")):
            gt = "inform"
            break

    # Step 3: compare, None if either undetermined
    if predicted is None or gt is None:
        return None, predicted, gt

    return predicted == gt, predicted, gt
