"""Hierarchical evaluation for MLP4CS pipeline."""
from src.evaluation.metrics import (
    calculate_domain_accuracy, calculate_intent_accuracy, calculate_jga, calculate_slot_f1, calculate_hallucination,
    calculate_action_accuracy, calculate_system_correctness, calculate_booking_success
)
from src.data.loader import load_split
from src.config import TARGET_DOMAINS


def evaluate_turn(custom_turn: dict, gt_domains: set[str], gt_intents: set[str], gt_slots: dict[str, str], gt_dialog_act: dict) -> dict:
    """
    Evaluate a single turn against ground truth.

    Args:
        custom_turn: turn dict from run_turn() with keys: domain, intent, slots, violations, delex_response, lex_response, db_results, valid, attempts
        gt_domains: set of active GT domains e.g., {"restaurant", "hotel"}
        gt_intents: set of active GT intents e.g., {"find_restaurant", "find_hotel"}
        gt_slots: flat prefixed GT slots normalized from dataset e.g., {"restaurant-area": "centre"}
        gt_dialog_act: GT dialog act from next SYSTEM turn e.g., {"Restaurant-Inform": [...]}
    Returns:
        dict with all turn-level metric results
    """
    predicted_domain = custom_turn["domain"]
    predicted_intent = custom_turn["intent"]
    predicted_slots = custom_turn["slots"]
    violations = custom_turn["violations"]
    lex_response = custom_turn["lex_response"]
    delex_response = custom_turn["delex_response"]
    db_results = custom_turn["db_results"]

    # Skip domain/intent/action on ambiguous turns, both None means multi-domain or farewell
    skip = predicted_domain is None and predicted_intent is None

    # 1. Domain accuracy
    domain_exact, domain_p, domain_r, domain_f1 = (
        (None, None, None, None) if skip else calculate_domain_accuracy(predicted_domain, gt_domains)
    )

    # 2. Intent accuracy
    intent_exact, intent_p, intent_r, intent_f1 = (
        (None, None, None, None) if skip else calculate_intent_accuracy(predicted_intent, gt_intents)
    )

    # 3. JGA, always computed
    jga = calculate_jga(predicted_slots, gt_slots)

    # 4. Slot F1, always computed
    slot_p, slot_r, slot_f1 = calculate_slot_f1(predicted_slots, gt_slots)

    # 5. Hallucination, checked on lex_response
    hallucinated, entity_mentioned = calculate_hallucination(lex_response, db_results)

    # 6. Policy compliance, violations list tells us directly
    # policy_compliant = len(violations) == 0
    # policy_compliant=False only when pipeline confirmed booking despite missing slots, if pipeline correctly asked for missing slots → compliant
    if violations:
        # Check delex_response for [ref] placeholder as the most reliable booking confirmation signal and also check lex_response for common confirmation words as fallback
        delex_lower = delex_response.lower()
        lex_lower = lex_response.lower()
        booking_confirmed = (
                "[ref]" in delex_lower or
                any(w in lex_lower for w in ("booked", "confirmed", "reservation", "your booking", "your table"))
        )
        policy_compliant = not booking_confirmed
    else:
        policy_compliant = True

    # 7. Action accuracy, skip on ambiguous turns
    action_correct, predicted_action, gt_action = (
        (None, None, None) if skip else calculate_action_accuracy(predicted_intent, violations, gt_dialog_act)
    )
    # 8. System correctness
    system_correct = calculate_system_correctness(hallucinated, policy_compliant)

    return {
        "predicted_domain": predicted_domain,
        "predicted_intent": predicted_intent,
        "gt_domains": list(gt_domains),
        "gt_intents": list(gt_intents),
        "domain_exact": domain_exact,
        "domain_p": domain_p,
        "domain_r": domain_r,
        "domain_f1": domain_f1,
        "intent_exact": intent_exact,
        "intent_p": intent_p,
        "intent_r": intent_r,
        "intent_f1": intent_f1,
        "jga": jga,
        "slot_p": slot_p,
        "slot_r": slot_r,
        "slot_f1": slot_f1,
        "gt_slots": gt_slots,
        "predicted_slots": predicted_slots,
        "hallucinated": hallucinated,
        "entity_mentioned": entity_mentioned,
        "policy_compliant": policy_compliant,
        "predicted_action": predicted_action,
        "gt_action": gt_action,
        "action_correct": action_correct,
        "system_correct": system_correct,
        "violations": violations,
        "attempts": custom_turn["attempts"],
        "user_utterance": custom_turn["user_utterance"],
        "lex_response": custom_turn["lex_response"],
        "cost": custom_turn["cost"],
        "response_time": custom_turn["response_time"],
        "db_results": db_results,
    }


def evaluate_dialogue(turn_results: list[dict], custom_turns: list[dict]) -> dict:
    """
    Aggregate turn-level metrics to dialogue level.

    Args:
        turn_results: list of dicts from evaluate_turn()
        custom_turns: list of custom_turn dicts from run_dialogue()
    Returns:
        dict with dialogue-level aggregated metrics
    """
    num_turns = len(turn_results)
    if num_turns == 0:
        return {"num_turns": 0}

    # Skip None turns for domain/intent/action, ambiguous/multi-domain turns
    domain_turns = [t for t in turn_results if t["domain_exact"] is not None]
    intent_turns = [t for t in turn_results if t["intent_exact"] is not None]
    action_turns = [t for t in turn_results if t["action_correct"] is not None]

    # Hallucination: only turns where entity was mentioned
    hall_turns = [t for t in turn_results if t["entity_mentioned"]]

    # Custom metrics averages
    avg_domain_f1 = sum(t["domain_f1"] for t in domain_turns) / len(domain_turns) if domain_turns else None
    avg_domain_p = sum(t["domain_p"] for t in domain_turns) / len(domain_turns) if domain_turns else None
    avg_intent_f1 = sum(t["intent_f1"] for t in intent_turns) / len(intent_turns) if intent_turns else None
    avg_intent_p = sum(t["intent_p"] for t in intent_turns) / len(intent_turns) if intent_turns else None
    avg_action = sum(t["action_correct"] for t in action_turns) / len(action_turns) if action_turns else None
    avg_hall = sum(t["hallucinated"] for t in hall_turns) / len(hall_turns) if hall_turns else 0.0
    avg_policy = sum(t["policy_compliant"] for t in turn_results) / num_turns
    avg_system = sum(t["system_correct"] for t in turn_results) / num_turns

    # Official metrics averages
    avg_jga = sum(t["jga"] for t in turn_results) / num_turns
    avg_slot_f1 = sum(t["slot_f1"] for t in turn_results) / num_turns
    avg_slot_p = sum(t["slot_p"] for t in turn_results) / num_turns
    avg_slot_r = sum(t["slot_r"] for t in turn_results) / num_turns

    # Booking success at dialogue level
    booking_success = calculate_booking_success(custom_turns)

    # Policy violations count
    policy_violations = sum(1 for t in turn_results if not t["policy_compliant"])

    return {
        "num_turns": num_turns,
        "avg_domain_f1": avg_domain_f1,
        "avg_domain_p": avg_domain_p,
        "avg_intent_f1": avg_intent_f1,
        "avg_intent_p": avg_intent_p,
        "avg_action": avg_action,
        "avg_hall": avg_hall,
        "avg_policy": avg_policy,
        "avg_system": avg_system,
        "avg_jga": avg_jga,
        "avg_slot_f1": avg_slot_f1,
        "avg_slot_p": avg_slot_p,
        "avg_slot_r": avg_slot_r,
        "booking_success": booking_success,
        "policy_violations": policy_violations,
        "turn_results": turn_results,
    }


def evaluate_dataset(dialogue_results: list[dict]) -> dict:
    """
    Aggregate dialogue-level metrics to dataset level.

    Args:
        dialogue_results: list of dicts from evaluate_dialogue()
    Returns:
        dict with dataset-level aggregated metrics
    """
    num_dialogues = len(dialogue_results)
    if num_dialogues == 0:
        return {"num_dialogues": 0}

    # MACRO AVERAGE
    # domain_dialogues_f1 = [d for d in dialogue_results if d["avg_domain_f1"] is not None]
    # domain_dialogues = [d for d in dialogue_results if d["avg_domain_p"] is not None]
    # intent_dialogues_f1 = [d for d in dialogue_results if d["avg_intent_f1"] is not None]
    # intent_dialogues = [d for d in dialogue_results if d["avg_intent_p"] is not None]
    # action_dialogues = [d for d in dialogue_results if d["avg_action"] is not None]
    # hall_dialogues = [d for d in dialogue_results if d["avg_hall"] > 0.0]
    # avg_domain_f1 = sum(d["avg_domain_f1"] for d in domain_dialogues_f1) / len(domain_dialogues_f1) if domain_dialogues_f1 else None
    # avg_domain_p = sum(d["avg_domain_p"] for d in domain_dialogues) / len(domain_dialogues) if domain_dialogues else None
    # avg_intent_f1 = sum(d["avg_intent_f1"] for d in intent_dialogues_f1) / len(intent_dialogues_f1) if intent_dialogues_f1 else None
    # avg_intent_p = sum(d["avg_intent_p"] for d in intent_dialogues) / len(intent_dialogues) if intent_dialogues else None
    # avg_action = sum(d["avg_action"] for d in action_dialogues) / len(action_dialogues) if action_dialogues else None
    # avg_hall = sum(d["avg_hall"] for d in hall_dialogues) / len(hall_dialogues) if hall_dialogues else 0.0
    # avg_policy = sum(d["avg_policy"] for d in dialogue_results) / num_dialogues
    # avg_system = sum(d["avg_system"] for d in dialogue_results) / num_dialogues
    # avg_jga = sum(d["avg_jga"] for d in dialogue_results) / num_dialogues
    # avg_slot_f1 = sum(d["avg_slot_f1"] for d in dialogue_results) / num_dialogues
    # avg_slot_p = sum(d["avg_slot_p"] for d in dialogue_results) / num_dialogues
    # avg_slot_r = sum(d["avg_slot_r"] for d in dialogue_results) / num_dialogues
    # total_violations = sum(d["policy_violations"] for d in dialogue_results)

    # MICRO AVERAGE: standard in MultiWOZ literature
    all_turns = [t for d in dialogue_results for t in d.get("turn_results", [])]
    total_turns = len(all_turns)
    domain_turns = [t for t in all_turns if t["domain_p"] is not None]
    intent_turns = [t for t in all_turns if t["intent_p"] is not None]
    action_turns = [t for t in all_turns if t["action_correct"] is not None]
    hall_turns = [t for t in all_turns if t["entity_mentioned"]]

    avg_domain_p = sum(t["domain_p"] for t in domain_turns) / len(domain_turns) if domain_turns else None
    avg_domain_f1 = sum(t["domain_f1"] for t in domain_turns) / len(domain_turns) if domain_turns else None
    avg_intent_p = sum(t["intent_p"] for t in intent_turns) / len(intent_turns) if intent_turns else None
    avg_intent_f1 = sum(t["intent_f1"] for t in intent_turns) / len(intent_turns) if intent_turns else None
    avg_action = sum(t["action_correct"] for t in action_turns) / len(action_turns) if action_turns else None
    avg_hall = sum(t["hallucinated"] for t in hall_turns) / len(hall_turns) if hall_turns else 0.0
    avg_policy = sum(t["policy_compliant"] for t in all_turns) / len(all_turns)
    avg_system = sum(t["system_correct"] for t in all_turns) / len(all_turns)
    avg_jga = sum(t["jga"] for t in all_turns) / len(all_turns)
    avg_slot_p = sum(t["slot_p"] for t in all_turns) / len(all_turns)
    avg_slot_r = sum(t["slot_r"] for t in all_turns) / len(all_turns)
    avg_slot_f1 = sum(t["slot_f1"] for t in all_turns) / len(all_turns)
    total_violations = sum(1 for t in all_turns if not t["policy_compliant"])

    # Booking: skip dialogues with no booking intent
    booking_dialogues = [d for d in dialogue_results if d["booking_success"] is not None]
    booking_rate = sum(d["booking_success"] for d in booking_dialogues) / len(booking_dialogues) if booking_dialogues else None

    # Policy violations rate
    violation_rate = total_violations / total_turns if total_turns > 0 else 0.0

    total_cost = sum(t.get("cost", 0.0) for d in dialogue_results for t in d.get("turn_results", []))
    avg_latency = sum(t.get("response_time", 0.0) for d in dialogue_results for t in d.get("turn_results", [])) / total_turns if total_turns > 0 else 0.0

    # Per-domain results
    per_domain = {}
    for domain in ["hotel", "restaurant"]:
        dom_turns = [
            t for d in dialogue_results
            for t in d.get("turn_results", [])
            if t.get("predicted_domain") == domain
        ]
        if not dom_turns:
            continue

        hall_t = [t for t in dom_turns if t["entity_mentioned"]]
        booking_t = [t for t in dom_turns if t.get("predicted_intent") == f"book_{domain}"]
        domain_t = [t for t in dom_turns if t["domain_p"] is not None]
        intent_t = [t for t in dom_turns if t["intent_p"] is not None]
        action_t = [t for t in dom_turns if t["action_correct"] is not None]

        # Per-domain booking rate: turns where booking was attempted, slots complete, and booking confirmed (what fraction of booking attempts were confirmed? -> turn level)
        confirmed_bookings = sum(1 for t in booking_t if not t["violations"] and "[ref]" in t.get("lex_response", "").lower())

        per_domain[domain] = {
            "num_turns": len(dom_turns),
            # Custom
            "avg_domain_f1": sum(t["domain_f1"] for t in domain_t) / len(domain_t) if domain_t else None,
            "avg_domain_p": sum(t["domain_p"] for t in domain_t) / len(domain_t) if domain_t else None,
            "avg_intent_f1": sum(t["intent_f1"] for t in intent_t) / len(intent_t) if intent_t else None,
            "avg_intent_p": sum(t["intent_p"] for t in intent_t) / len(intent_t) if intent_t else None,
            "avg_action": sum(t["action_correct"] for t in action_t) / len(action_t) if action_t else None,
            "avg_hall": sum(t["hallucinated"] for t in hall_t) / len(hall_t) if hall_t else 0.0,
            "avg_policy": 1 - sum(t["policy_compliant"] for t in dom_turns) / len(dom_turns),  # violation rate = 1 - compliance rate
            "avg_system": sum(t["system_correct"]   for t in dom_turns) / len(dom_turns),
            "booking_rate": confirmed_bookings / len(booking_t) if booking_t else None,
            "total_cost": sum(t.get("cost", 0.0) for t in dom_turns),
            "avg_latency": sum(t.get("response_time", 0.0) for t in dom_turns) / len(dom_turns),
            # Official
            "avg_jga": sum(t["jga"] for t in dom_turns) / len(dom_turns),
            "avg_slot_p": sum(t["slot_p"] for t in dom_turns) / len(dom_turns),
            "avg_slot_r": sum(t["slot_r"] for t in dom_turns) / len(dom_turns),
            "avg_slot_f1": sum(t["slot_f1"] for t in dom_turns) / len(dom_turns),
        }

    return {
        "num_dialogues": num_dialogues,
        "total_turns": total_turns,
        # Custom
        "avg_domain_f1": avg_domain_f1,
        "avg_domain_p": avg_domain_p,
        "avg_intent_f1": avg_intent_f1,
        "avg_intent_p": avg_intent_p,
        "avg_action": avg_action,
        "avg_hall": avg_hall,
        "avg_policy": avg_policy,
        "avg_system": avg_system,
        "booking_rate": booking_rate,
        "violation_rate": violation_rate,
        "total_violations": total_violations,
        # Official
        "avg_jga": avg_jga,
        "avg_slot_f1": avg_slot_f1,
        "avg_slot_p": avg_slot_p,
        "avg_slot_r": avg_slot_r,
        # Cost and Latency
        "total_cost": total_cost,
        "avg_latency": avg_latency,
        # Per domain results
        "per_domain": per_domain,
    }


def evaluate_experiment(custom_results: dict, split: str) -> dict:
    """
    Evaluate full experiment against ground truth.

    Args:
        custom_results: dict keyed by dialogue_id → list of custom_turn dicts (one per USER turn),
        e.g., {"MUL1271.json": [{"domain": "restaurant", "intent": "find_restaurant", "slots": {...}, ...}, ...]}
        split: dataset split, one of 'train', 'dev', 'test'
    Returns:
        dataset-level metrics dict with nested dialogue_results
    """
    dialogues = load_split(split)

    # [{"dialogue_id": "PMUL4398.json", "turns": [...], "services": [...]}, ...}, ...] → {"PMUL4398.json": {dialogue dict}, ...}
    dialogue_map = {d["dialogue_id"]: d for d in dialogues}

    dialogue_results = []

    # Loop over predicted dialogues (custom_results keyed by dialogue_id)
    for dialogue_id, custom_turns in custom_results.items():

        # Match predicted turns (custom_results) with GT dialogue data (dialogue_map) by dialogue_id
        dialogue = dialogue_map.get(dialogue_id)
        if dialogue is None:
            continue

        turn_results = []
        custom_idx = 0
        gt_accumulated: dict[str, str] = {}  # Accumulate GT slots across turns

        # Loop over GT turns (USER turns trigger evaluation, SYSTEM turns skipped)
        for i, turn in enumerate(dialogue["turns"]):

            if turn["speaker"] != "USER":
                continue

            if custom_idx >= len(custom_turns):
                break

            custom_turn = custom_turns[custom_idx]
            custom_idx += 1

            # Extract GT from all active frames in TARGET_DOMAINS
            gt_domains = set()
            gt_intents = set()
            gt_slots_this_turn = {}
            for frame in turn["frames"]:
                if frame["service"] in TARGET_DOMAINS and frame["state"]["active_intent"] != "NONE":
                    gt_domains.add(frame["service"])
                    gt_intents.add(frame["state"]["active_intent"])
                    for k, v in frame["state"]["slot_values"].items():
                        gt_slots_this_turn[k] = v[0]

            if not gt_domains:
                continue

            # Accumulate GT slots across turns, mirrors how pipeline accumulates predicted slots
            # e.g., {"area": "north"} + {"food": "chinese"} → {"area": "north", "food": "chinese"} + {"area": "center"} → {"area": "center", "food": "chinese"}
            gt_accumulated.update(gt_slots_this_turn)

            # GT dialog_act from next SYSTEM turn
            gt_dialog_act = {}
            if i + 1 < len(dialogue["turns"]):
                next_turn = dialogue["turns"][i + 1]
                if next_turn["speaker"] == "SYSTEM":
                    gt_dialog_act = next_turn.get("dialog_act", {})

            turn_result = evaluate_turn(custom_turn, gt_domains, gt_intents, gt_accumulated.copy(), gt_dialog_act)
            turn_results.append(turn_result)

            # print(f"\n  Turn {turn['turn_id']} User utterance: {turn['utterance']}")
            # print(f"  GT domains: {gt_domains} | Pred domain: {custom_turn['domain']} | Domain Precision: {turn_result['domain_p']}")
            # print(f"  GT intents: {gt_intents} | Pred intent: {custom_turn['intent']} | Intent Precision: {turn_result['intent_p']}")
            # print(f"  GT action: {turn_result['gt_action']} | Pred action: {turn_result['predicted_action']} | Action: {turn_result['action_correct']}")
            # print(f"  GT slots: {gt_accumulated} | Pred slots: {custom_turn['slots']} ")
            # print(f"  JGA: {turn_result['jga']} | Slot F1: {turn_result['slot_f1']:.2f} ")
            # print(f"  Hall: {turn_result['hallucinated']}")
            # print(f"  Policy: {turn_result['policy_compliant']}")
            # print(f"  System: {turn_result['system_correct']}")

        dialogue_result = evaluate_dialogue(turn_results, custom_turns)
        dialogue_result["dialogue_id"] = dialogue_id
        dialogue_result["services"] = dialogue["services"]
        dialogue_results.append(dialogue_result)

    dataset_metrics = evaluate_dataset(dialogue_results)
    dataset_metrics["dialogue_results"] = dialogue_results

    # dataset_metrics structure:
    # {
    #   "num_dialogues": int, "total_turns": int, "total_cost": float,
    #   "avg_domain_p": float|None, "avg_intent_p": float|None, "avg_action": float|None,
    #   "avg_jga": float, "avg_slot_f1": float, "avg_slot_r": float,
    #   "avg_hall": float, "avg_policy": float, "avg_system": float,
    #   "booking_rate": float|None, "violation_rate": float, "total_violations": int,
    #   "dialogue_results": [
    #     { "dialogue_id": str, "services": list, "num_turns": int,
    #       "avg_jga": float, "avg_slot_f1": float, "booking_success": bool|None,
    #       "turn_results": [{ "user_utterance": str, "jga": bool, "slot_f1": float,
    #                          "hallucinated": bool, "policy_compliant": bool, ... }]
    #     }, ...
    #   ]
    # }

    return dataset_metrics
