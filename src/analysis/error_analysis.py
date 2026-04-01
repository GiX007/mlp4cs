"""Error analysis for MLP4CS experiments."""
import json
from tabulate import tabulate
from src.config import RESULTS_DIR
from src.utils import print_separator


def load_all_turns() -> dict[str, list[dict]]:
    """
    Load all turn results from results/ folder.

    Returns:
        dict mapping experiment_name → list of turn dicts
    """
    results: dict[str, list[dict]] = {}

    for turns_file in sorted(RESULTS_DIR.rglob("overall/*_turns.json")):
        data = json.loads(turns_file.read_text(encoding="utf-8"))
        experiment = data.get("experiment", turns_file.stem)
        turns = data.get("turns", [])
        if turns:
            results[experiment] = turns
            print(f"Loaded {len(turns)} turns from {turns_file.name}")

    return results


def print_summary(all_turns: dict[str, list[dict]]) -> None:
    """
    Print per-experiment summary table.

    Args:
        all_turns: dict mapping experiment_name → list of turn dicts
    """
    rows = []
    for experiment, turns in sorted(all_turns.items()):
        n = len(turns)
        domain_turns = [t for t in turns if t["domain_p"] is not None]
        intent_turns = [t for t in turns if t["intent_p"] is not None]
        action_turns = [t for t in turns if t["action_correct"] is not None]
        hall_turns = [t for t in turns if t["entity_mentioned"]]

        rows.append([
            experiment,
            n,
            f"{sum(t['domain_p'] for t in domain_turns)/len(domain_turns)*100:.1f}" if domain_turns else "N/A",
            f"{sum(t['intent_p'] for t in intent_turns)/len(intent_turns)*100:.1f}" if intent_turns else "N/A",
            f"{sum(t['action_correct'] for t in action_turns)/len(action_turns)*100:.1f}" if action_turns else "N/A",
            f"{sum(t['jga'] for t in turns)/n*100:.1f}",
            f"{sum(t['slot_f1'] for t in turns)/n*100:.1f}",
            f"{sum(t['hallucinated'] for t in hall_turns)/len(hall_turns)*100:.1f}" if hall_turns else "0.0",
            f"{sum(1 for t in turns if not t['policy_compliant'])/n*100:.1f}",
            f"{sum(t['system_correct'] for t in turns)/n*100:.1f}",
        ])

    headers = ["Experiment", "Turns", "DomainP%", "IntentP%", "Action%", "JGA%", "SlotF1%", "Hall%", "PolViol%", "SysCorr%"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def print_failures(all_turns: dict[str, list[dict]], n_examples: int = 3) -> None:
    """
    Print failure examples per type per experiment.

    Args:
        all_turns: dict mapping experiment_name → list of turn dicts
        n_examples: number of examples to print per failure type per experiment
    """
    failure_types = {
        "JGA failure": lambda t: not t["jga"],
        "Slot error": lambda t: t["slot_f1"] < 1.0,
        "Hallucination": lambda t: t["hallucinated"],
        "Policy violation": lambda t: not t["policy_compliant"],
        "Action error": lambda t: t["action_correct"] is False,
        "System incorrect": lambda t: not t["system_correct"],
        "Domain error": lambda t: t["domain_p"] is not None and t["domain_p"] < 1.0,
        "Intent error": lambda t: t["intent_p"] is not None and t["intent_p"] < 1.0,
    }

    for failure_name, condition in failure_types.items():
        print_separator(f"FAILURE: {failure_name}")

        for experiment, turns in sorted(all_turns.items()):
            failed = [t for t in turns if condition(t)]
            if not failed:
                continue

            print(f"\n  [{experiment}]: {len(failed)} failures")

            for i, t in enumerate(failed[:n_examples]):
                print(f"\n  Example {i+1}:")
                print(f"  Dialogue: {t['dialogue_id']}")
                print(f"  User utterance: {t['user_utterance']}")
                print(f"  LLM Response: {t['lex_response'][:150]}")
                print(f"  GT domains: {t['gt_domains']} | GT intents: {t['gt_intents']}")
                print(f"  Pred domain: {t['predicted_domain']} | Pred intent: {t['predicted_intent']}")
                print(f"  GT slots: {t.get('gt_slots', 'N/A')}")
                print(f"  Pred slots: {t.get('predicted_slots', 'N/A')}")
                print(f"  Violations: {t['violations']}")
                print(f"  JGA: {t['jga']} | SlotF1: {t['slot_f1']:.2f} | Hall: {t['hallucinated']} | Valid: {t['system_correct']}")

                # Failure-specific extra info
                if failure_name == "Hallucination":
                    print(f"  DB results: {t.get('db_results', [])[:1] if t.get('db_results') else 'empty'}")
                elif failure_name == "Action error":
                    print(f"  GT action: {t['gt_action']} | Pred action: {t['predicted_action']}")
                elif failure_name == "Domain error":
                    print(f"  DomainP: {t['domain_p']:.2f}")
                elif failure_name == "Intent error":
                    print(f"  IntentP: {t['intent_p']:.2f}")


def run_analysis() -> None:
    """Run full error analysis on all saved experiment results."""
    print_separator("MLP4CS ERROR ANALYSIS")

    all_turns = load_all_turns()

    if not all_turns:
        print("No turn results found in results/")
        return

    print_separator("SUMMARY TABLE")
    print_summary(all_turns)

    print_failures(all_turns)

    print_separator("END OF ERROR ANALYSIS")
