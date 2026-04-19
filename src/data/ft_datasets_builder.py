"""Builds datasets for fine-tuning for Exp3."""
import json
from pathlib import Path
from src.config import TARGET_DOMAINS, DST_INSTRUCTION, RESPGEN_INSTRUCTION
from src.db import find_entity
from src.utils import format_history, format_slots
from src.data.loader import load_split
from src.pipeline.lexicalizer import delexicalize
from src.pipeline.dst import build_dst_prompt
from src.pipeline.response_generator import build_respgen_prompt


def build_dst_sample(turn: dict, history: list[dict], dialogue_id: str) -> dict | None:
    """
    Build one DST fine-tuning sample from a USER turn.

    Args:
        turn: USER turn dict with frames, utterance, turn_id
        history: list of previous turns in this dialogue
        dialogue_id: dialogue identifier for traceability
    Returns:
        sample dict with keys: dialogue_id, turn_id, instruction, input, output
        None if no active intent found in target domains
    """
    # Find the first active frame in our target domains
    label = None
    for frame in turn.get("frames", []):
        intent = frame.get("state", {}).get("active_intent", "NONE")
        service = frame.get("service", "")
        if intent != "NONE" and service in TARGET_DOMAINS:
            slot_values = frame["state"].get("slot_values", {})
            label = {"domain": service, "intent": intent, "slots": slot_values}
            break

    if not label:
        return None

    _, input_str = build_dst_prompt(turn["utterance"], history)
    output_str = (f"DOMAIN: {label['domain']}\n"
                  f"INTENT: {label['intent']}\n"
                  f"SLOTS: {format_slots(label['slots'])}")

    return {
        "dialogue_id": dialogue_id,
        "turn_id": turn["turn_id"],
        "instruction": DST_INSTRUCTION,
        "input": input_str,
        "output": output_str,
    }


def build_dst_dataset(split: str = "train") -> list[dict]:
    """
    Build DST fine-tuning dataset from a dataset split.

    Args:
        split: one of 'train', 'dev', 'test'
    Returns:
        list of DST sample dicts
    """
    dialogues = load_split(split)
    samples = []

    for dialogue in dialogues:
        history = []
        for turn in dialogue["turns"]:
            if turn["speaker"] == "USER":
                sample = build_dst_sample(turn, history, dialogue["dialogue_id"])
                if sample:
                    samples.append(sample)
            history.append(turn)

    print(f"[{split}] DST samples: {len(samples)}")
    return samples


def extract_booking_ref(turn: dict) -> str:
    """
    Extract booking reference from a SYSTEM turn's dialog act if present.

    Args:
        turn: SYSTEM turn dict with dialog_act key
    Returns:
        booking reference string or empty string if not present

    Example:
        dialog_act = {"Booking-Book": [["ref", "FRGZWQL2"]]} → "FRGZWQL2"
    """
    booking_acts = turn.get("dialog_act", {}).get("Booking-Book", [])
    for slot, value in booking_acts:
        if slot == "ref":
            return value
    return ""


def extract_entity_name(turn: dict, domain: str) -> str:
    """
    Extract entity name mentioned in a SYSTEM turn's dialog act.

    Args:
        turn: SYSTEM turn dict with dialog_act key
        domain: target domain, e.g., 'hotel' or 'restaurant'
    Returns:
        entity name string or empty string if not present

    Example:
        dialog_act = {"Restaurant-Inform": [["name", "Bedouin"]]} → "bedouin"
    """
    domain_cap = domain.capitalize()
    for act_key, act_values in turn.get("dialog_act", {}).items():
        if act_key.startswith(domain_cap):
            for slot, value in act_values:
                if slot == "name":
                    return value.lower()
    return ""


def build_respgen_sample(turn: dict, history: list[dict], dialogue_id: str, last_dst_label: dict) -> dict | None:
    """
    Build one response generator fine-tuning sample from a SYSTEM turn.

    Args:
        turn: SYSTEM turn dict with utterance and turn_id
        history: list of previous turns in this dialogue
        dialogue_id: dialogue identifier for traceability
        last_dst_label: domain + intent + slots from the previous USER turn
    Returns:
        sample dict with keys: dialogue_id, turn_id, instruction, input, output
        None if no dst label available or no matching entity found
    """
    if not last_dst_label:
        return None

    domain = last_dst_label["domain"]
    slots = last_dst_label["slots"]
    intent = last_dst_label["intent"]

    # Get matching entity from DB for delexicalization
    act_name = extract_entity_name(turn, domain)  # GT dialog act name is more reliable than DB lookup as it always returns the first matching entity
    flat_slots = {k: v[0] if isinstance(v, list) else v for k, v in slots.items()}  # Flatten list values before DB query: {"hotel-area": ["north"]} → {"hotel-area": "north"}
    if act_name:
        flat_slots[f"{domain}-name"] = act_name
    entities = find_entity(domain, flat_slots)
    entity = entities[0] if entities else {}

    # Delexicalize the raw system utterance
    ref = extract_booking_ref(turn)
    output_str = delexicalize(turn["utterance"], domain, entity, ref)
    if not output_str:
        return None

    # input_str = format_history(history)
    # input_str = f"{input_str}\n" if input_str else ""
    # input_str += (f"DOMAIN: {domain}\n"
    #               f"INTENT: {intent}\n"
    #               f"SLOTS: {format_slots(slots)}")

    # history[-1] is always the USER turn preceding this SYSTEM turn (using the system utterance here would leak the target into the input)
    user_utt = history[-1]["utterance"] if history else ""

    _, input_str = build_respgen_prompt(
        history, user_utt, domain, intent,
        flat_slots, db_results=[entity] if entity else [],
        violations=[], zeroshot=True
    )

    return {
        "dialogue_id": dialogue_id,
        "turn_id": turn["turn_id"],
        "instruction": RESPGEN_INSTRUCTION,
        "input": input_str,
        "output": output_str,
    }


def build_respgen_dataset(split: str = "train") -> list[dict]:
    """
    Build response generator fine-tuning dataset from a dataset split.

    Args:
        split: one of 'train', 'dev', 'test'
    Returns:
        list of response generator sample dicts
    """
    dialogues = load_split(split, verbose=False)
    samples = []

    for dialogue in dialogues:
        history = []
        last_dst_label = {}

        for turn in dialogue["turns"]:
            if turn["speaker"] == "USER":
                for frame in turn.get("frames", []):
                    intent = frame.get("state", {}).get("active_intent", "NONE")
                    service = frame.get("service", "")
                    if intent != "NONE" and service in TARGET_DOMAINS:
                        last_dst_label = {
                            "domain": service,
                            "intent": intent,
                            "slots": frame["state"].get("slot_values", {})
                        }
                        break
            else:
                sample = build_respgen_sample(turn, history, dialogue["dialogue_id"], last_dst_label)
                if sample:
                    samples.append(sample)

            history.append(turn)

    print(f"[{split}] Response generator samples: {len(samples)}")
    return samples


def save_dataset(samples: list[dict], output_path: Path) -> None:
    """
    Save fine-tuning samples to a JSON file.

    Args:
        samples: list of sample dicts
        output_path: path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples → {output_path}")
