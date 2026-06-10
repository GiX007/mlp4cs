"""Project statistics: file counts, line counts, size, and directory breakdown."""
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