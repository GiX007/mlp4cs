"""Data loading utilities for MLP4CS."""
import json
from src.config import MULTIWOZ_DATA_DIR, DIALOG_ACTS_FILE, TARGET_DOMAINS


def load_dialogues(split: str) -> list[dict]:
    """
    Load all dialogues for a given dataset split.

    Args:
        split: one of 'train', 'dev', 'test'
    Returns:
        list of dialogue dicts, each with a unique 'dialogue_id'
    """
    split_dir = MULTIWOZ_DATA_DIR / split
    files = sorted(split_dir.glob("dialogues_*.json"))

    dialogues = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            dialogues.extend(json.load(f))

    return dialogues


def load_dialog_acts() -> dict:
    """
    Load the dialog_acts.json file into memory.

    Returns:
        dict mapping dialogue_id to turn-level dialog act annotations
    """
    with open(DIALOG_ACTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def attach_dialog_acts(dialogues: list[dict], dialog_acts: dict) -> list[dict]:
    """
    Attach dialog act annotations to each turn of each dialogue.

    Args:
        dialogues: list of dialogue dicts from load_dialogues()
        dialog_acts: dict from load_dialog_acts()
    Returns:
        same dialogues list with 'dialog act' key added to each turn
    """
    for dialogue in dialogues:
        acts = dialog_acts.get(dialogue["dialogue_id"], {})
        for turn in dialogue["turns"]:
            tid = str(turn["turn_id"])
            turn["dialog_act"] = acts.get(tid, {}).get("dialog_act", {})
    return dialogues


def filter_by_domains(dialogues: list[dict], domains: set) -> list[dict]:
    """
    Keep dialogues whose services are all within target domains, e.g., ['hotel', 'restaurant'] passes, ['restaurant', 'taxi'] does not.

    Args:
        dialogues: list of dialogue dicts
        domains: set of target domain strings
    Returns:
        filtered list of dialogues
    """
    return [d for d in dialogues if all(s in domains for s in d["services"])]


def load_split(split: str, verbose: bool = False) -> list[dict]:
    """
    Load a dataset split filtered by target domains with dialog acts attached.

    Args:
        split: one of 'train', 'dev', 'test'
        verbose: if True, prints loading summary
    Returns:
        filtered list of dialogue dicts with dialog acts attached to each turn
    """
    dialogues = load_dialogues(split)
    acts = load_dialog_acts()
    dialogues = attach_dialog_acts(dialogues, acts)
    filtered = filter_by_domains(dialogues, TARGET_DOMAINS)

    if verbose:
        print(f"[{split}] {len(dialogues)} total → {len(filtered)} kept "
              f"{sorted(TARGET_DOMAINS)} only ({len(filtered)/len(dialogues)*100:.1f}%)")

    return filtered
