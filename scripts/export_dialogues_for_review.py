"""Export MultiWOZ dialogues to plain text files for expert review.

Run with: python -m scripts.export_dialogues_for_review
"""
from pathlib import Path
from collections import Counter

from src.data.loader import load_split


def count_dialogues_by_services(split: str) -> Counter:
    """Count dialogues grouped by their `services` list.nPurpose: See how many dialogues are single-domain vs multi-domain.

    Args:
        split: Dataset split, one of 'train', 'dev', 'test'

    Return:
        Counter mapping a tuple of sorted service names to dialogue count
    """
    dialogues = load_split(split)
    return Counter(tuple(sorted(d["services"])) for d in dialogues)


def print_multi_domain_counts() -> None:
    """Print multi-domain dialogue counts per split."""
    for split_name in ["train", "dev", "test"]:
        counts = count_dialogues_by_services(split_name)
        total = sum(counts.values())
        print(f"\n[{split_name}] total = {total}")
        for services, n in counts.most_common():
            tag = "multi" if len(services) > 1 else "single"
            print(f"  {tag} {services}: {n} ({n / total * 100:.1f}%)")


def categorize_dialogue(dialogue: dict) -> str:
    """Return the bucket label for one dialogue. Purpose: Decide which output file a dialogue belongs to.

    Args:
        dialogue: One MultiWOZ dialogue dict with a "services" list

    Return:
        One of "hotel", "restaurant", "multi", "skip"
    """
    services = sorted(dialogue["services"])
    if len(services) == 0:
        return "skip"
    if len(services) == 1:
        return services[0]
    return "multi"


def format_dialogue(dialogue: dict, number: int) -> str:
    """Format one dialogue as a printable text block. Purpose: Build the "Dialogue N / user: ... / system: ..." layout for one dialogue.

    Args:
        dialogue: One MultiWOZ dialogue dict with a "turns" list
        number: Sequence number shown in the header (e.g., 1 for "Dialogue 1")

    Return:
        A string with the header and all turns, blank lines between user-system pairs
    """
    lines: list[str] = [f"Dialogue {number}\n"]

    for turn in dialogue["turns"]:
        speaker = turn["speaker"].lower()  # "USER" -> "user", "SYSTEM" -> "system"
        lines.append(f"{speaker}: {turn['utterance']}")
        # Blank line after each system turn to separate user-system pairs
        if speaker == "system":
            lines.append("")

    return "\n".join(lines)


def group_dialogues_by_bucket(split: str) -> dict[str, list[dict]]:
    """Group dialogues of one split into buckets by category. Purpose: Split one dataset split into "hotel", "restaurant", "multi" lists.

    Args:
        split: Dataset split, one of 'train', 'dev', 'test'

    Return:
        Dict mapping bucket label to list of dialogues. "skip" dialogues are dropped.
    """
    buckets: dict[str, list[dict]] = {}

    for dialogue in load_split(split):
        label = categorize_dialogue(dialogue)
        if label == "skip":
            continue
        buckets.setdefault(label, []).append(dialogue)

    return buckets


def build_text_for_bucket(bucket_label: str) -> str:
    """Build the full text body for one bucket across all splits. Purpose: Assemble the content of one output .txt file (e.g., "hotel").

    Args:
        bucket_label: Bucket name, one of "hotel", "restaurant", "multi"

    Return:
        A string with split headers and numbered dialogues, ready to be written to disk
    """
    sections: list[str] = []

    for split in ["train", "dev", "test"]:
        buckets = group_dialogues_by_bucket(split)
        dialogues = buckets.get(bucket_label, [])
        if not dialogues:
            continue  # Skip splits where this bucket has no dialogues

        section_lines: list[str] = [f"=== {split.upper()} ==="]
        for i, dialogue in enumerate(dialogues, start=1):
            section_lines.append(format_dialogue(dialogue, number=i))

        sections.append("\n\n".join(section_lines))

    return "\n\n".join(sections)


def write_bucket_file(bucket_label: str, output_dir: Path) -> None:
    """Write one bucket's text to disk as a .txt file. Purpose: Build the text for a bucket and save it to <output_dir>/dialogues_<bucket_label>.txt.

    Args:
        bucket_label: Bucket name, one of "hotel", "restaurant", "multi"
        output_dir: Folder where the .txt file will be written (created if missing)

    Return:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    text = build_text_for_bucket(bucket_label)
    file_path = output_dir / f"dialogues_{bucket_label}.txt"
    file_path.write_text(text, encoding="utf-8")

    print(f"Wrote {file_path} ({len(text)} chars)")


def main() -> None:
    """Writes hotel, restaurant, and multi .txt files."""
    output_dir = Path("data/expert_review")

    for bucket_label in ["hotel", "restaurant", "multi"]:
        write_bucket_file(bucket_label, output_dir)


if __name__ == "__main__":
    # print_multi_domain_counts()
    main()
