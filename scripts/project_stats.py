"""Project statistics: file counts, line counts, size, and directory breakdown."""
import json
from pathlib import Path

EXCLUDE = {"venv"}


def _is_excluded(path: Path) -> bool:
    """Check if any part of the path is in the exclusion set."""
    return bool(EXCLUDE & set(path.parts))


def count_py_files(root: str = ".") -> int:
    """Count .py files in the project."""
    return sum(1 for f in Path(root).rglob("*.py") if not _is_excluded(f))


def count_lines(root: str = ".") -> tuple[int, int]:
    """Count total and non-blank lines across all .py files. Returns: (total_lines, non_blank_lines)."""
    total = 0
    non_blank = 0
    for f in Path(root).rglob("*.py"):
        if not _is_excluded(f):
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            total += len(lines)
            non_blank += sum(1 for line in lines if line.strip())
    return total, non_blank


def lines_per_file(root: str = ".") -> dict[str, int]:
    """Line count per .py file, sorted largest first."""
    result = {}
    for f in Path(root).rglob("*.py"):
        if not _is_excluded(f):
            result[str(f)] = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def dir_summary(root: str = ".") -> dict[str, int]:
    """Count all files per directory, sorted by count descending. Returns: dict of {directory_path: file_count}."""
    counts: dict[str, int] = {}
    for f in Path(root).rglob("*"):
        if f.is_file() and not _is_excluded(f):
            folder = str(f.parent)
            counts[folder] = counts.get(folder, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def project_size_mb(root: str = ".") -> float:
    """Total project size in MB."""
    total_bytes = sum(
        f.stat().st_size
        for f in Path(root).rglob("*")
        if f.is_file() and not _is_excluded(f)
    )
    return round(total_bytes / (1024 * 1024), 2)


def find_retried_turns(turns_path: str) -> None:
    """
    Print all turns where the supervisor triggered a retry (attempts > 1).

    Args:
        turns_path: path to a *_turns.json file
    Return:
        None (prints results to console)
    """
    # turns.json is a dict: {dialogue_id: [turn_dict, turn_dict, ...]}
    data = json.loads(Path(turns_path).read_text(encoding="utf-8"))

    count = 0
    for dialogue_id, turns in data.items():
        for i, turn in enumerate(turns):
            if turn.get("attempts", 1) > 1:
                count += 1
                print(f"{dialogue_id} | turn {i} | attempts={turn['attempts']} | valid={turn.get('valid')}")
                print(f"   response: {turn.get('delex_response')}")

    print(f"\nTotal retried turns: {count}")


# Run from project root: python scripts/project_stats.py
if __name__ == "__main__":
    total, non_blank = count_lines()
    print(f"Python files: {count_py_files()}")
    print(f"Total lines: {total}")
    print(f"Non-blank: {non_blank}")
    size_mb = project_size_mb()
    print(f"Project size : {size_mb} MB ({round(size_mb / 1024, 2)} GB)")

    # print("\nLines per .py file:")
    # for path, lines in lines_per_file().items():
    #     print(f"  {lines:>5}  {path}")

    # print("\nFiles per directory:")
    # for folder, count in dir_summary().items():
    #     print(f"  {count:>5}  {folder}")

    # Check retries in turns.json
    # path = r"C:\...\mlp4cs\archived_results\open_source_test_20260418\exp1_gpt\overall\exp1_gpt_test_20260421_083748_turns.json"
    # find_retried_turns(path)