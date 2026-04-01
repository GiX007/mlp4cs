"""Tomiinek evaluator wrapper for MLP4CS pipeline."""
from mwzeval.metrics import Evaluator


def run_tomiinek(tomiinek_results: dict) -> dict:
    """
    Run Tomiinek evaluator on pipeline predictions.

    Computes Inform%, Success%, BLEU and Combined scores.
    Uses GT belief state from MultiWOZ 2.2 (state={} per turn).

    Args:
        tomiinek_results: dict keyed by lowercase dialogue_id without .json, e.g., {"pmul4398": [{"response": "...", "state": {}, "active_domains": []}]}
    Returns:
        dict with keys: inform, success, bleu, combined
    """
    if not tomiinek_results:
        return {"inform": 0.0, "success": 0.0, "bleu": 0.0, "combined": 0.0}

    # print(f"State sent to Tomiinek: {list(tomiinek_results.values())[0][0]['state']}")
    # print(f"\nNum dialogues: {len(tomiinek_results)}")
    # first_id = list(tomiinek_results.keys())[0]
    # print(f"First dialogue id: '{first_id}' | turns: {len(tomiinek_results[first_id])}")
    # for i, t in enumerate(tomiinek_results[first_id]):
    #     print(f"  turn {i}: response='{t['response'][:80]}' | state={t['state']} | domains={t['active_domains']}")

    try:
        e = Evaluator(bleu=True, success=True, richness=False)
        results = e.evaluate(tomiinek_results)

        # Tomiinek output structure
        # results["success"]["inform"] = {"attraction": x, "hotel": x, "restaurant": x, "taxi": x, "train": x, "total": x}
        # results["success"]["success"] = {"attraction": x, "hotel": x, "restaurant": x, "taxi": x, "train": x, "total": x}
        # results["bleu"] = {"damd": x, "uniconv": x, "hdsa": x, "lava": x, "augpt": x, "mwz22": x}

        # "total" = aggregate across ALL domains present in the input dialogues and since we only feed hotel + restaurant, our "total" covers 2 of 5 leaderboard domains
        # Leaderboard "total" includes: attraction, hotel, restaurant, taxi, train so our scores are NOT directly comparable to leaderboard numbers

        # Overall metrics across all TARGET_DOMAINS
        inform = results["success"]["inform"]["total"]
        success = results["success"]["success"]["total"]
        bleu = results["bleu"]["mwz22"]
        combined = 0.5 * (inform + success) + bleu

        # Per-domain breakdown
        # inform_hotel = results["success"]["inform"].get("hotel", 0.0)
        # inform_restaurant = results["success"]["inform"].get("restaurant", 0.0)
        # success_hotel = results["success"]["success"].get("hotel", 0.0)
        # success_restaurant = results["success"]["success"].get("restaurant", 0.0)

        return {
            "inform": round(inform, 2),
            "success": round(success, 2),
            "bleu": round(bleu, 2),
            "combined": round(combined, 2),
            # "inform_hotel": round(inform_hotel, 2),
            # "inform_restaurant": round(inform_restaurant, 2),
            # "success_hotel": round(success_hotel, 2),
            # "success_restaurant": round(success_restaurant, 2),
        }

    except Exception as ex:
        print(f"\nTomiinek evaluation failed: {ex}")
        return {"inform": 0.0, "success": 0.0, "bleu": 0.0, "combined": 0.0}
