"""Exploration utilities for the MultiWOZ 2.2 dataset."""
import json
from pathlib import Path
from src.utils import print_separator
from src.data.loader import load_split
from src.config import DB_DIR, TARGET_DOMAINS, OPEN_SOURCE_MODELS, FINETUNE_DST_TRAIN_FILE, FINETUNE_RESPGEN_TRAIN_FILE, FINETUNE_DST_DEV_FILE, FINETUNE_RESPGEN_DEV_FILE
from src.utils import capture_and_save


def explore_basic_structure(split: str = "train") -> None:
    """
    Show basic structure of one MultiWOZ 2.2 dialogue.

    Args:
        split: dataset split to load, one of 'train', 'dev', 'test'
    """
    dialogues = load_split(split)

    print_separator(f"BASIC STRUCTURE of MultiWOZ 2.2 + Dialog Acts of [{split}]")

    first = dialogues[0]
    print(f"\n{split} split is a {type(dialogues)} with dialogues as elements of {type(first)}")
    print(f"Number of dialogues: {len(dialogues)}")
    print(f"\nTop-level keys: {list(first.keys())}")
    print(f"dialogue_id: {type(first['dialogue_id']).__name__} | {first['dialogue_id']}")
    print(f"services: {type(first['services']).__name__} | length={len(first['services'])} | {first['services']}")
    print(f"turns: {type(first['turns']).__name__} of {type(first['turns'][0]).__name__} | length={len(first['turns'])}")

    first_turn = first["turns"][0]
    print(f"\nFirst turn keys: {list(first_turn.keys())}")
    print(f"speaker: {type(first_turn['speaker']).__name__} | {first_turn['speaker']}")
    print(f"turn_id: {type(first_turn['turn_id']).__name__} | {first_turn['turn_id']}")
    print(f"utterance: {type(first_turn['utterance']).__name__} | {first_turn['utterance']}")
    print(f"frames: list of {type(first_turn['frames'][0]).__name__} | length={len(first_turn['frames'])}")

    first_frame = first_turn["frames"][0]
    print(f"\nFirst frame keys: {list(first_frame.keys())}")
    print(f"actions: list | length={len(first_frame['actions'])}")
    print(f"service: {type(first_frame['service']).__name__} | {first_frame['service']}")
    print(f"slots: list | length={len(first_frame['slots'])}")
    print(f"state: {type(first_frame['state']).__name__} | keys={list(first_frame['state'].keys())}")

    state = first_frame["state"]
    print(f"\nUnpacking state:")
    print(f"active_intent: {type(state['active_intent']).__name__} | {state['active_intent']}")
    print(f"requested_slots: list | length={len(state['requested_slots'])} | {state['requested_slots']}")
    print(f"slot_values: {type(state['slot_values']).__name__} | length={len(state['slot_values'])} | {state['slot_values']}")

    print(f"\nDialog act (attached):")
    print(f"dialog_act: {type(first_turn['dialog_act']).__name__} | {first_turn['dialog_act']}")

    print_separator("END OF BASIC STRUCTURE")


def explore_single_dialogue(dialogue_idx: int = 0, split: str = "train") -> None:
    """
    Print a full dialogue turn by turn with intents, slot values and dialog acts.

    Args:
        dialogue_idx: index of dialogue to inspect
        split: dataset split to load
    """
    dialogues = load_split(split, verbose=False)
    dialogue = dialogues[dialogue_idx]

    print_separator(f"SINGLE DIALOGUE (Index {dialogue_idx}) of [{split}]")
    print(f"\ndialogue_id: {dialogue['dialogue_id']}")
    print(f"services: {dialogue['services']}")
    print(f"num turns: {len(dialogue['turns'])}")

    for turn in dialogue["turns"]:
        speaker = turn["speaker"]
        print(f"\n  Turn {turn['turn_id']} [{speaker}]: {turn['utterance']}")

        if speaker == "USER":
            for frame in turn["frames"]:
                intent = frame["state"]["active_intent"]
                slots = frame["state"]["slot_values"]
                requested = frame["state"]["requested_slots"]

                if intent == "NONE":
                    continue

                print(f"         service: {frame['service']}")
                print(f"         intent: {intent}")
                print(f"         slots: {slots}")
                print(f"         requested: {requested}")

        print(f"         dialog_act: {turn['dialog_act']}")

    print_separator("END OF SINGLE DIALOGUE")


def explore_turn_details(dialogue_idx: int = 0, turn_idx: int = 4, split: str = "train") -> None:
    """
    Deep dive into one specific turn with all fields with types.

    Args:
        dialogue_idx: index of dialogue in the split
        turn_idx: index of turn within the dialogue
        split: dataset split to load
    """
    dialogues = load_split(split, verbose=False)
    dialogue = dialogues[dialogue_idx]
    turn = dialogue["turns"][turn_idx]

    print_separator(f"TURN DETAILS (Dialogue {dialogue_idx}, Turn {turn_idx}) of [{split}]")
    print(f"\nturn_id: {type(turn['turn_id']).__name__} | {turn['turn_id']}")
    print(f"speaker: {type(turn['speaker']).__name__} | {turn['speaker']}")
    print(f"utterance: {type(turn['utterance']).__name__} | {turn['utterance']}")
    print(f"frames: list of dict | length={len(turn['frames'])}")
    print(f"dialog_act: {type(turn['dialog_act']).__name__} | {turn['dialog_act']}")

    print("\nActive frames (active_intent != NONE):")
    for frame in turn["frames"]:
        intent = frame["state"]["active_intent"] if frame.get("state") else "N/A"
        if intent in ("NONE", "N/A"):
            continue
        print(f"\n  service: {type(frame['service']).__name__} | {frame['service']}")
        print(f"  active_intent: {type(intent).__name__} | {intent}")
        print(f"  slot_values: {type(frame['state']['slot_values']).__name__} | {frame['state']['slot_values']}")
        print(f"  requested_slots: list | length={len(frame['state']['requested_slots'])} | {frame['state']['requested_slots']}")
        print(f"  slots (spans): list | length={len(frame['slots'])}")

    print("\nNote on USER turns:")
    print("  frames = 8 dicts → one frame per service (hotel, restaurant, taxi, ...)")
    print("  intent = NONE → service not active in this turn (skipped)")
    print("  intent = find_X → user is searching for an entity")
    print("  intent = book_X → user wants to make a booking")
    print("  slot_values → constraints user has specified so far (accumulate across turns)")
    print("  requested_slots → information user is asking the system to provide")

    print("\nNote on SYSTEM turns:")
    print("  frames = [] → system utterance with no span annotations")
    print("  dialog_act → attached from dialog_acts.json")

    print_separator("END OF TURN DETAILS")


def explore_db_structure() -> None:
    """Explore hotel and restaurant DB files with fields, types, and differences."""
    hotel_path = DB_DIR / "hotel_db.json"
    restaurant_path = DB_DIR / "restaurant_db.json"

    with open(hotel_path, "r", encoding="utf-8") as f:
        hotel_db = json.load(f)
    with open(restaurant_path, "r", encoding="utf-8") as f:
        restaurant_db = json.load(f)

    print_separator("DB STRUCTURE of HOTEL & RESTAURANT")

    print(f"\nHotel DB type: {type(hotel_db)} | is a list of: {type(hotel_db[0])}")
    print(f"Hotel DB length: {len(hotel_db)} entries\n")

    h = hotel_db[0]
    for key, val in h.items():
        print(f"  {key}: {type(val).__name__} | {val}")

    print("\nFull first hotel entry:")
    print(json.dumps(h, indent=4))

    print(f"\nRestaurant DB type: {type(restaurant_db)} | is a list of: {type(restaurant_db[0])}")
    print(f"Restaurant DB length: {len(restaurant_db)} entries\n")

    r = restaurant_db[0]
    for key, val in r.items():
        print(f"  {key}: {type(val).__name__} | {val}")

    print("\nFull first restaurant entry:")
    print(json.dumps(r, indent=4))

    print("\nKey differences hotel vs restaurant:")
    print("  Hotel only: stars, parking, internet, takesbookings, type, price (dict)")
    print("  Restaurant only: food, introduction")
    print("  Shared: name, area, pricerange, phone, address, postcode, location, id")

    print_separator("END OF DB STRUCTURE")


def explore_conversation_examples(n_examples: int = 2, split: str = "train") -> None:
    """
    Print readable conversation examples from target domain dialogues.

    Args:
        n_examples: number of dialogues to print
        split: dataset split to load
    """
    dialogues = load_split(split, verbose=False)

    print_separator(f"CONVERSATION EXAMPLES of {sorted(TARGET_DOMAINS)} [{split}]")
    print(f"\nTotal filtered dialogues: {len(dialogues)}")
    print(f"Showing first {n_examples}:")

    for dialogue in dialogues[:n_examples]:
        print("\n" + "-" * 60)
        print(f"Dialogue ID: {dialogue['dialogue_id']}")
        print(f"Services: {dialogue['services']}")
        print(f"Num turns: {len(dialogue['turns'])}")

        for turn in dialogue["turns"]:
            speaker = turn["speaker"]
            print(f"\n  Turn {turn['turn_id']} [{speaker}]: {turn['utterance']}")

            if speaker == "USER":
                for frame in turn["frames"]:
                    intent = frame["state"]["active_intent"]
                    slots = frame["state"]["slot_values"]
                    requested = frame["state"]["requested_slots"]

                    if intent == "NONE":
                        continue

                    print(f"         service: {frame['service']}")
                    print(f"         intent: {intent}")
                    print(f"         slots: {slots}")
                    print(f"         requested: {requested}")

            print(f"         dialog_act: {turn['dialog_act']}")

    print_separator("END OF CONVERSATION EXAMPLES")


def count_dialogues_per_split() -> None:
    """Count target domain dialogues across all splits."""
    print_separator(f"DIALOGUE COUNTS of {sorted(TARGET_DOMAINS)}")
    print()

    for split in ("train", "dev", "test"):
        load_split(split, verbose=True)

    print_separator("END OF DIALOGUE COUNTS")


def explore_slot_values(split: str = "train") -> None:
    """
    Print all unique slot values per slot across target domain dialogues.

    Args:
        split: dataset split to load
    """
    from collections import Counter

    dialogues = load_split(split, verbose=False)
    slot_value_map = {}

    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            for frame in turn.get("frames", []):
                if frame.get("service") not in TARGET_DOMAINS:
                    continue
                for slot, values in frame.get("state", {}).get("slot_values", {}).items():
                    if slot not in slot_value_map:
                        slot_value_map[slot] = Counter()
                    for v in values:
                        slot_value_map[slot][v.lower()] += 1

    print_separator(f"SLOT VALUE DISTRIBUTION of {sorted(TARGET_DOMAINS)} of [{split}]")
    for slot, counter in sorted(slot_value_map.items()):
        print(f"\n{slot}:")
        for value, count in counter.most_common():
            print(f"  {value!r}: {count}")
    print_separator("END OF SLOT VALUE DISTRIBUTION")


def inspect_json_file(filepath: str) -> None:
    """
    Load and inspect a JSON fine-tuning dataset file.

    Args: filepath: path to the JSON file to inspect
    """
    path = Path(filepath)

    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nFile: {path.name}")
    print(f"Type: {type(data).__name__}")
    print(f"Count: {len(data)} examples")
    print(f"Keys in each example: {list(data[0].keys())}")

    # Show first example in full
    print("\nFirst Example:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))


def explore_finetune_token_lengths(model_alias: str = "qwen3_8b") -> None:
    """
    Print token-length distribution of DST and RespGen fine-tuning datasets.

    Args:
        model_alias: key from OPEN_SOURCE_MODELS used to pick a tokenizer
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(OPEN_SOURCE_MODELS[model_alias])

    print_separator(f"TOKEN LENGTHS of FINETUNE TRAIN DATASETS (tokenizer={model_alias})")

    for role, path in [("dst", FINETUNE_DST_TRAIN_FILE), ("respgen", FINETUNE_RESPGEN_TRAIN_FILE)]:
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        lengths = []
        for ex in examples:
            # Reconstruct the exact training text (Alpaca + EOS) to get a true count
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}{tokenizer.eos_token}"
            )
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))

        lengths.sort()
        n = len(lengths)
        print(f"\n{role} ({n} examples):")
        print(f"  min: {lengths[0]}")
        print(f"  max: {lengths[-1]}")
        print(f"  mean: {sum(lengths) / n:.0f}")
        print(f"  median: {lengths[n // 2]}")
        print(f"  p95: {lengths[int(n * 0.95)]}") # 95% of examples are shorter than this
        print(f"  p99: {lengths[int(n * 0.99)]}") # 99% of examples are shorter than this

    print_separator("END OF TOKEN LENGTHS")


def inspect_finetune_datasets() -> None:
    """Inspect fine-tuning dataset files with counts, keys, and first example."""
    print_separator("INSPECTING FINE-TUNING DATASETS")
    inspect_json_file(str(FINETUNE_DST_TRAIN_FILE))
    inspect_json_file(str(FINETUNE_RESPGEN_TRAIN_FILE))
    inspect_json_file(str(FINETUNE_DST_DEV_FILE))
    inspect_json_file(str(FINETUNE_RESPGEN_DEV_FILE))
    print_separator("END OF INSPECTING FINE-TUNING DATASETS")


def main() -> None:
    """Exploration workflow. It runs all explore functions in order."""
    explore_basic_structure()
    explore_single_dialogue()
    explore_turn_details()
    explore_db_structure()
    explore_conversation_examples()
    count_dialogues_per_split()
    explore_slot_values()
    inspect_finetune_datasets()


# Run with: python -m src.data.dataset_explorer
if __name__ == "__main__":
    capture_and_save(func=main, output_path="docs/mwz22_exploration.txt")
