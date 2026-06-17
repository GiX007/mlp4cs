"""Evaluate Tomiinek metrics (Inform, Success, BLEU, Combined) from tomiinek_input.json files."""
import json
from mwzeval.metrics import Evaluator

# Open-source runs
PATH_EXP1_QWEN3_4B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen3_4b\exp1_qwen3_4b_tomiinek_input.json"
PATH_EXP1_QWEN3_8B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen3_8b\exp1_qwen3_8b_tomiinek_input.json"
PATH_EXP1_QWEN3_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen3_14b\exp1_qwen3_14b_tomiinek_input.json"
PATH_EXP1_QWEN25_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen25_14b\exp1_qwen25_14b_tomiinek_input.json"
PATH_EXP1_LLAMA31_8B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_llama31_8b\exp1_llama31_8b_tomiinek_input.json"
PATH_EXP1_PHI4_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_phi4_14b\exp1_phi4_14b_tomiinek_input.json"

PATH_EXP2_HOMO_QWEN3_4B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_homo_qwen3_4b\exp2_homo_qwen3_4b_tomiinek_input.json"
PATH_EXP2_HOMO_QWEN3_8B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_homo_qwen3_8b\exp2_homo_qwen3_8b_tomiinek_input.json"
PATH_EXP2_HOMO_QWEN3_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_homo_qwen3_14b\exp2_homo_qwen3_14b_tomiinek_input.json"
PATH_EXP2_HETERO_QWEN25_QWEN3_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_hetero_qwen25_qwen3_14b\exp2_hetero_qwen25_qwen3_14b_tomiinek_input.json"
PATH_EXP2_HOMO_LLAMA32_3B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_homo_llama32_3b\exp2_homo_llama32_3b_tomiinek_input.json"
PATH_EXP2_HOMO_LLAMA31_8B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_homo_llama31_8b\exp2_homo_llama31_8b_tomiinek_input.json"
PATH_EXP2_HOMO_PHI4_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_homo_phi4_14b\exp2_homo_phi4_14b_tomiinek_input.json"
PATH_EXP2_HETERO_QWEN3_14b_PHI4_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_hetero_qwen3_14b_phi4_14b\exp2_hetero_qwen3_14b_phi4_14b_tomiinek_input.json"

PATH_EXP3_FT_HOMO_QWEN3_4B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_homo_qwen3_4b\exp3_ft_homo_qwen3_4b_tomiinek_input.json"
PATH_EXP3_FT_HOMO_QWEN3_8B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_homo_qwen3_8b\exp3_ft_homo_qwen3_8b_tomiinek_input.json"
PATH_EXP3_FT_HOMO_QWEN3_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_homo_qwen3_14b\exp3_ft_homo_qwen3_14b_tomiinek_input.json"
PATH_EXP3_FT_HOMO_LLAMA32_3B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_homo_llama32_3b\exp3_ft_homo_llama32_3b_tomiinek_input.json"
PATH_EXP3_FT_HOMO_LLAMA31_8B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_homo_llama31_8b\exp3_ft_homo_llama31_8b_tomiinek_input.json"
PATH_EXP3_FT_HOMO_PHI4_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_homo_phi4_14b\exp3_ft_homo_phi4_14b_tomiinek_input.json"
PATH_EXP3_FT_HETERO_QWEN3_14b_PHI4_14B = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp3_ft_hetero_qwen3_14b_phi4_14b\exp3_ft_hetero_qwen3_14b_phi4_14b_tomiinek_input.json"

# API runs
PATH_EXP1_GPT = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp1_gpt\exp1_gpt_tomiinek_input.json"
PATH_EXP1_GPT_NANO = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp1_gpt-nano\exp1_gpt-nano_tomiinek_input.json"
PATH_EXP1_HAIKU = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp1_haiku\exp1_haiku_tomiinek_input.json"

PATH_EXP2_HOMO_GPT = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp2_homo_gpt\exp2_homo_gpt_tomiinek_input.json"
PATH_EXP2_HOMO_HAIKU = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp2_homo_haiku\exp2_homo_haiku_tomiinek_input.json"
PATH_EXP2_HETERO_GPT_HAIKU = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp2_hetero_gpt_haiku\exp2_hetero_gpt_haiku_tomiinek_input.json"
PATH_EXP2_HETERO_HAIKU_GPT = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\api_baselines_cot_test_20260420\exp2_hetero_haiku_gpt\exp2_hetero_haiku_gpt_tomiinek_input.json"

# Bug 1 fix: undo the inverted TOMIINEK_SLOT_MAP that the runner applied
SLOT_UNMAP: dict[str, str] = {
    "booking-day": "bookday",
    "booking-people": "bookpeople",
    "booking-stay": "bookstay",
    "booking-time": "booktime",
}

RUNS: dict[str, str] = {
    # "exp1_gpt": PATH_EXP1_GPT,
    # "exp1_gpt_nano": PATH_EXP1_GPT_NANO,
    # "exp1_haiku": PATH_EXP1_HAIKU,
    # "exp1_qwen3_4b": PATH_EXP1_QWEN3_4B,
    # "exp1_qwen3_8b": PATH_EXP1_QWEN3_8B,
    # "exp1_qwen3_14b": PATH_EXP1_QWEN3_14B,
    # "exp1_qwen25_14b": PATH_EXP1_QWEN25_14B,
    # "exp1_llama31_8b": PATH_EXP1_LLAMA31_8B,
    # "exp1_phi4_14b": PATH_EXP1_PHI4_14B,
    # "exp2_homo_gpt": PATH_EXP2_HOMO_GPT,
    # "exp2_homo_haiku": PATH_EXP2_HOMO_HAIKU,
    # "exp2_hetero_gpt_haiku": PATH_EXP2_HETERO_GPT_HAIKU,
    # "exp2_hetero_haiku_gpt": PATH_EXP2_HETERO_HAIKU_GPT,
    # "exp2_homo_llama32_3b": PATH_EXP2_HOMO_LLAMA32_3B,
    # "exp2_homo_llama31_8b": PATH_EXP2_HOMO_LLAMA31_8B,
    # "exp2_homo_qwen3_4b": PATH_EXP2_HOMO_QWEN3_4B,
    # "exp2_homo_qwen3_8b": PATH_EXP2_HOMO_QWEN3_8B,
    # "exp2_homo_qwen3_14b": PATH_EXP2_HOMO_QWEN3_14B,
    # "exp2_hetero_qwen25_qwen3_14b": PATH_EXP2_HETERO_QWEN25_QWEN3_14B,
    # "exp2_hetero_qwen3_14b_phi4_14b": PATH_EXP2_HETERO_QWEN3_14b_PHI4_14B,
    # "exp2_homo_phi4_14b": PATH_EXP2_HOMO_PHI4_14B,
    # "exp3_ft_homo_llama32_3b": PATH_EXP3_FT_HOMO_LLAMA32_3B,
    # "exp3_ft_homo_llama31_8b": PATH_EXP3_FT_HOMO_LLAMA31_8B,
    # "exp3_ft_homo_qwen3_8b": PATH_EXP3_FT_HOMO_QWEN3_8B,
    # "exp3_ft_homo_qwen3_14b": PATH_EXP3_FT_HOMO_QWEN3_14B,
    # "exp3_ft_homo_qwen3_1b": PATH_EXP3_FT_HOMO_QWEN3_4B,
    # "exp3_ft_homo_phi4_14b": PATH_EXP3_FT_HOMO_PHI4_14B,
    "exp3_ft_hetero_qwen3_14b_phi4_14b": PATH_EXP3_FT_HETERO_QWEN3_14b_PHI4_14B,
}


def fix_slot_keys(data: dict) -> dict:
    """
    Rewrite booking slot names back to canonical Tomiinek form.

    Args:
        data: dict mapping dialogue_id to list of turn dicts
    Returns:
        same dict with state slot keys corrected in place
    """
    for turns in data.values():
        for turn in turns:
            for domain, slots in turn.get("state", {}).items():
                if isinstance(slots, dict):
                    turn["state"][domain] = {SLOT_UNMAP.get(k, k): v for k, v in slots.items()}
    return data


def score_run(data_path: str) -> dict:
    """
    Load one Tomiinek input JSON, evaluate, return metrics.

    Args:
        data_path: absolute path to a *_tomiinek_input.json file
    Returns:
        dict with keys: inform, success, bleu, combined
    """
    with open(data_path) as f:
        data = json.load(f)

    # BEFORE fix
    before_keys = set()
    for turns in data.values():
        for turn in turns:
            for slots in turn.get("state", {}).values():
                if isinstance(slots, dict):
                    before_keys.update(slots.keys())



    data = fix_slot_keys(data)

    # AFTER fix
    after_keys = set()
    for turns in data.values():
        for turn in turns:
            for slots in turn.get("state", {}).values():
                if isinstance(slots, dict):
                    after_keys.update(slots.keys())

    # print(f"\n  {data_path.split(chr(92))[-1]}")
    # print(f"  BEFORE: {sorted(k for k in before_keys if 'book' in k or 'price' in k)}")
    # print(f"  AFTER:  {sorted(k for k in after_keys if 'book' in k or 'price' in k)}")

    e = Evaluator(bleu=True, success=True, richness=False)
    r = e.evaluate(data)
    inform = r["success"]["inform"]["total"]
    success = r["success"]["success"]["total"]
    bleu = r["bleu"]["mwz22"]
    combined = 0.5 * (inform + success) + bleu
    return {"inform": inform, "success": success, "bleu": bleu, "combined": combined}



# Run with: python src/evaluation/score_tomiinek.py
print(f"\n{'Run':<35} {'Inform':>8} {'Success':>8} {'BLEU':>6} {'Combined':>9}")
print("-" * 70)
for name, path in RUNS.items():
    m = score_run(path)
    print(f"{name:<35} {m['inform']:>8.2f} {m['success']:>8.2f} {m['bleu']:>6.2f} {m['combined']:>9.2f}")






# with open() as f:
#     data = json.load(f)
#
# e = Evaluator(bleu=True, success=True, richness=False)
# r = e.evaluate(data)
#
# inform = r["success"]["inform"]["total"]
# success = r["success"]["success"]["total"]
# bleu = r["bleu"]["mwz22"]
# combined = 0.5 * (inform + success) + bleu
#
# print(f"Inform: {inform:.2f}")
# print(f"Success: {success:.2f}")
# print(f"BLEU: {bleu:.2f}")
# print(f"Combined: {combined:.2f}")
