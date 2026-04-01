"""Results saving and reporting for MLP4CS pipeline."""
import json
import time
from pathlib import Path
from src.config import RESULTS_DIR


def save_results(experiment_name: str, dataset_metrics: dict, split: str) -> None:
    """
    Save experiment results to 3 JSON files: dataset, dialogues, turns.

    Args:
        experiment_name: e.g., 'exp2_homo_gpt'
        dataset_metrics: full metrics dict from evaluate_experiment() + tomiinek
        split: dataset split used
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    exp_dir = RESULTS_DIR / experiment_name
    overall_dir = exp_dir / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    base = f"{experiment_name}_{split}_{timestamp}"

    # Extract nested results before saving
    dialogue_results = dataset_metrics.pop("dialogue_results", [])

    # shared header
    header = {
        "experiment": experiment_name,
        "split": split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # File 1: dataset-level
    _save_json(overall_dir / f"{base}_dataset.json", {**header, "evaluation_level": "dataset", **dataset_metrics})

    # File 2: dialogue-level, remove turn_results from each dialogue
    dialogues = []
    for d in dialogue_results:
        turn_results = d.pop("turn_results", [])

        dialogues.append({
            "dialogue_id": d.get("dialogue_id"),
            "services": d.get("services"),
            **{k: v for k, v in d.items() if k not in ("dialogue_id", "services")}
        })

        d["turn_results"] = turn_results  # restore
    _save_json(overall_dir / f"{base}_dialogues.json", {**header, "evaluation_level": "dialogue", "dialogues": dialogues})

    # File 3: turn-level
    turns = []
    for d in dialogue_results:
        for t in d.get("turn_results", []):
            turns.append({"dialogue_id": d["dialogue_id"], **t})
    _save_json(overall_dir / f"{base}_turns.json", {**header, "evaluation_level": "turn", "turns": turns})

    # Per-domain files, same 3 files as overall but filtered by domain
    per_domain_dir = exp_dir / "per_domain"
    per_domain_dir.mkdir(parents=True, exist_ok=True)
    per_domain = dataset_metrics.get("per_domain", {})

    for domain, dm in per_domain.items():
        domain_base = f"{experiment_name}_{split}_{timestamp}_{domain}"

        # File 1: domain dataset-level
        _save_json(per_domain_dir / f"{domain_base}_dataset.json", {**header, "evaluation_level": "domain", "domain": domain, **dm})

        # File 2: domain dialogue-level, filter dialogues by service
        domain_dialogues = [d for d in dialogues if domain in d.get("services", [])]
        _save_json(per_domain_dir / f"{domain_base}_dialogues.json", {**header, "evaluation_level": "domain_dialogue", "domain": domain, "dialogues": domain_dialogues})

        # File 3: domain turn-level, filter turns by predicted domain
        domain_turns = [t for t in turns if t.get("predicted_domain") == domain]
        _save_json(per_domain_dir / f"{domain_base}_turns.json", {**header, "evaluation_level": "domain_turn", "domain": domain, "turns": domain_turns})

    print(f"\nResults saved to {RESULTS_DIR}/")



def _save_json(path: Path, data: dict) -> None:
    """Save dict to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_table(all_results: dict, experiment_name: str, exp_id: int = 2) -> None:
    """
    Print comparison table to screen and append to leaderboard.txt.

    Args:
        all_results: dict mapping config_name → metrics dict
        experiment_name: e.g., 'Modular Zero-Shot Pipeline'
        exp_id: experiment number
    """
    from tabulate import tabulate

    rows = []
    for name, r in all_results.items():
        if not r:
            continue
        short_name = name.split("/")[-1]  # "data/models/llama31_8b" → "llama31_8b", "homo_gpt" → "homo_gpt"
        rows.append([
            short_name,
            f"{r.get('avg_domain_p', 0)*100:.1f}",
            # f"{r.get('avg_domain_f1', 0)*100:.1f}",  # domain F1
            f"{r.get('avg_intent_p', 0)*100:.1f}",
            # f"{r.get('avg_intent_f1', 0)*100:.1f}",  # intent F1
            f"{r.get('avg_action', 0)*100:.1f}" if r.get('avg_action') is not None else "N/A",
            f"{r.get('avg_jga', 0)*100:.1f}",
            f"{r.get('avg_slot_r', 0)*100:.1f}",
            f"{r.get('avg_slot_f1', 0)*100:.1f}",
            f"{r.get('avg_hall', 0)*100:.1f}",
            f"{r.get('violation_rate', 0)*100:.1f}",
            f"{r.get('avg_system', 0)*100:.1f}",
            f"{r.get('booking_rate', 0)*100:.1f}" if r.get('booking_rate') is not None else "N/A",
            f"{r.get('inform', 0):.1f}",
            f"{r.get('success', 0):.1f}",
            f"{r.get('bleu', 0):.2f}",
            f"{r.get('combined', 0):.2f}",
            f"${r.get('total_cost', 0):.4f}",
            f"{r.get('avg_latency', 0):.2f}s",
        ])

    headers = ["Config", "DomainP%", "IntentP%", "Action%", "JGA%", "SlotR%", "SlotF1%", "Hall%", "PolViol%", "SysCorr%", "Book%", "Inform%", "Success%", "BLEU", "Combined", "Cost($)", "Latency(s)"]

    table = tabulate(rows, headers=headers, tablefmt="github")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    block = (
        f"\n{'-'*60}\n"
        f"Experiment {exp_id}: {experiment_name}\n"
        f"Updated: {timestamp}\n"
        f"{'-'*60}\n\n"
        f"{table}\n"
    )

    print(block)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = RESULTS_DIR / "leaderboard.txt"
    with open(leaderboard_path, "a", encoding="utf-8") as f:
        f.write(block)

    # Per-domain table
    _print_domain_table(all_results, exp_id, experiment_name, timestamp)


def _print_domain_table(all_results: dict, exp_id: int, experiment_name: str, timestamp: str) -> None:
    """
    Print per-domain breakdown table and append to leaderboard.txt.

    Args:
        all_results: dict mapping config_name → metrics dict
        exp_id: experiment number
        experiment_name: e.g., 'Modular Zero-Shot Pipeline'
        timestamp: formatted timestamp string e.g., '2026-03-23 16:12:08'
    """
    from tabulate import tabulate

    rows = []
    for name, r in all_results.items():
        if not r:
            continue
        short_name = name.split("/")[-1]  # "data/models/llama31_8b" → "llama31_8b", "homo_gpt" → "homo_gpt"
        per_domain = r.get("per_domain", {})
        for domain, dm in per_domain.items():
            rows.append([
                short_name,
                domain,
                f"{dm.get('avg_domain_p', 0)*100:.1f}" if dm.get('avg_domain_p') is not None else "N/A",
                # f"{dm.get('avg_domain_f1', 0)*100:.1f}",  # domain F1
                f"{dm.get('avg_intent_p', 0)*100:.1f}" if dm.get('avg_intent_p') is not None else "N/A",
                # f"{dm.get('avg_intent_f1', 0)*100:.1f}",  # intent F1
                f"{dm.get('avg_action', 0)*100:.1f}" if dm.get('avg_action') is not None else "N/A",
                f"{dm.get('avg_jga', 0)*100:.1f}",
                f"{dm.get('avg_slot_r', 0)*100:.1f}",
                f"{dm.get('avg_slot_f1', 0)*100:.1f}",
                f"{dm.get('avg_hall', 0)*100:.1f}",
                f"{dm.get('avg_policy', 0)*100:.1f}",
                f"{dm.get('avg_system', 0)*100:.1f}",
                f"{dm.get('booking_rate', 0)*100:.1f}" if dm.get('booking_rate') is not None else "N/A",
                f"{dm.get('total_cost', 0):.4f}" if dm.get('total_cost') is not None else "N/A",
                f"{dm.get('avg_latency', 0):.2f}s" if dm.get('avg_latency') is not None else "N/A",
            ])

    if not rows:
        return

    headers = ["Config", "Domain", "DomainP%", "IntentP%", "Action%", "JGA%", "SlotR%", "SlotF1%", "Hall%", "PolViol%", "SysCorr%", "Book%", "Cost($)", "Latency(s)"]

    table = tabulate(rows, headers=headers, tablefmt="github")

    block = (
        f"\n{'-'*60}\n"
        f"Experiment {exp_id}: {experiment_name} (Per-Domain)\n"
        f"Updated: {timestamp}\n"
        f"{'-'*60}\n\n"
        f"{table}\n"
    )

    print(block)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "leaderboard.txt", "a", encoding="utf-8") as f:
        f.write(block)
