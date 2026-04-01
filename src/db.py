"""Database query functions for target domains."""
import json
import random
import string
from src.config import DB_DIR, MAX_DB_RESULTS
from src.utils import normalize_slot_value

# Loaded once per domain, reused across all turns and dialogues
_DB_CACHE: dict[str, list[dict]] = {}


def load_db(domain: str) -> list[dict]:
    """
    Load hotel or restaurant DB into memory. Cached after first load.

    Args:
        domain: target domain, e.g., 'hotel' or 'restaurant'
    Returns:
        list of entity dicts from the DB
    """
    if domain not in _DB_CACHE:
        db_path = DB_DIR / f"{domain}_db.json"
        with open(db_path, "r", encoding="utf-8") as f:
            _DB_CACHE[domain] = json.load(f)
    return _DB_CACHE[domain]


def _match_entity(entity: dict, constraints: dict) -> bool:
    """
    Check if a DB entity satisfies all slot constraints.

    Args:
        entity:  one entity dict from the DB
        constraints: normalized belief state dict {slot_name: value}
    Returns:
        True if entity matches all constraints, False otherwise
    """
    for slot, value in constraints.items():
        if value in ("dontcare", "none", "", "not mentioned", "any"):
            continue

        field = slot.split("-")[-1]  # "hotel-area" → "area"

        if field not in entity:
            continue

        # Name: fuzzy containment match, all other fields: exact match
        entity_val = normalize_slot_value(str(entity[field]))

        if field == "name":
            # fuzzy match: "home from home" matches "home from home guest house" and vice versa
            if value not in entity_val and entity_val not in value:
                return False
        else:
            if entity_val != value:
                return False

    return True


def find_entity(domain: str, belief_state: dict) -> list[dict]:
    """
    Search the DB for entities matching the belief state constraints.

    Args:
        domain: target domain, e.g., 'hotel' or 'restaurant'
        belief_state: prefixed slot dict e.g. {"hotel-area": "north", "hotel-pricerange": "cheap"}
    Returns:
        list of matching entity dicts, up to MAX_DB_RESULTS
    """
    if not domain:
        return []

    db = load_db(domain)
    normalized = {k: normalize_slot_value(v) for k, v in belief_state.items()}
    matches = [e for e in db if _match_entity(e, normalized)]
    return matches[:MAX_DB_RESULTS]


def _generate_ref() -> str:
    """
    Generate a random 8-character alphanumeric booking reference.

    Returns:
        e.g., 'AB3X9K2M'
    """
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=8))


def book_entity(domain: str, belief_state: dict) -> dict:
    """
    Verify a matching entity exists and return a booking confirmation.

    Args:
        domain: 'hotel' or 'restaurant'
        belief_state: prefixed slot dict including booking slots, e.g., {"hotel-name": "acorn guest house", "hotel-bookday": "monday"}

    Returns:
        dict with keys: success, ref, entity, reason
    """
    matches = find_entity(domain, belief_state)

    if not matches:
        return {"success": False, "ref": None, "entity": None, "reason": f"No {domain} found matching the given constraints."}

    entity = matches[0]  # always book the first matching entity (standard MultiWOZ convention)
    return {"success": True, "ref": _generate_ref(), "entity": entity, "reason": None}
