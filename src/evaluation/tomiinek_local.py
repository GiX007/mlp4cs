"""
Evaluate Tomiinek metrics (Inform, Success, BLEU, Combined) from cluster-generated tomiinek_input.json files.

Run with: python src/evaluation/tomiinek_local.py
"""
import json
from mwzeval.metrics import Evaluator

PATH1 = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen3_8b\exp1_qwen3_8b_tomiinek_input.json"
PATH2 = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen3_14b\exp1_qwen3_14b_tomiinek_input.json"
PATH3 = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp1_qwen25_14b\exp1_qwen25_14b_tomiinek_input.json"
PATH4 = r"C:\Users\giorg\Projects\PycharmProjects\mlp4cs\archived_results\open_source_test_20260418\exp2_hetero_qwen25_qwen3_14b\exp2_hetero_qwen25_qwen3_14b_tomiinek_input.json"

with open(PATH4) as f:
    data = json.load(f)

e = Evaluator(bleu=True, success=True, richness=False)
r = e.evaluate(data)

inform = r["success"]["inform"]["total"]
success = r["success"]["success"]["total"]
bleu = r["bleu"]["mwz22"]
combined = 0.5 * (inform + success) + bleu

print(f"Inform: {inform:.2f}")
print(f"Success: {success:.2f}")
print(f"BLEU: {bleu:.2f}")
print(f"Combined: {combined:.2f}")
