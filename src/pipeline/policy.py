"""Policy checker for MLP4CS pipeline."""
from src.config import BOOKING_REQUIRED_SLOTS


def policy(intent: str | None, slots: dict[str, str]) -> list[str]:
    """
    Check that all required slots are present for the current booking intent.

    Args:
        intent: current intent from DST, e.g., 'book_hotel'
        slots: accumulated belief state, e.g., {"hotel-bookday": "monday"}
    Returns:
        list of missing required slot names, empty if no violations
    """
    if intent not in BOOKING_REQUIRED_SLOTS:
        return []

    required = BOOKING_REQUIRED_SLOTS[intent]
    return [slot for slot in required if slot not in slots]
