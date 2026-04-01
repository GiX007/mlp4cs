"""Lexicalization and delexicalization utilities for MLP4CS pipeline."""


def lexicalize(delex_response: str, db_results: list[dict], domain: str, ref: str = "") -> str:
    """
    Replace delexicalized placeholders with real entity values.

    Args:
        delex_response: delexicalized response with placeholders
        db_results: list of matching entities from DB, uses first match
        domain: active domain, e.g., 'hotel' or 'restaurant'
        ref: booking reference string if available
    Returns:
        lexicalized response with real values substituted
    """
    if not db_results:
        return delex_response

    entity = db_results[0]  # Standard MultiWOZ convention: always use first match

    slots = ["name", "phone", "address", "postcode"]  # Slots as they defined in db (without prefix)
    result = delex_response

    for slot in slots:
        placeholder = f"[{domain}_{slot}]"
        value = entity.get(slot, "")
        if value and placeholder in result:
            result = result.replace(placeholder, str(value))

    if ref and "[ref]" in result:
        result = result.replace("[ref]", ref)

    return result


def delexicalize(response: str, domain: str, entity: dict, ref: str = "") -> str:
    """
    Replace entity values in a response with domain-prefixed placeholders.

    Args:
        response: raw system utterance
        domain: 'hotel' or 'restaurant'
        entity: DB entity dict with name, phone, address, postcode
        ref: booking reference string if available
    Returns:
        delexicalized response string

    Example:
        "Bedouin's phone is 01223367660" → "[restaurant_name]'s phone is [restaurant_phone]"
    """
    slots = ["name", "phone", "address", "postcode"]
    result = response

    for slot in slots:
        value = entity.get(slot, "")
        if value and value.lower() in result.lower():
            # Case-insensitive replace preserving original casing position
            idx = result.lower().find(value.lower())
            result = result[:idx] + f"[{domain}_{slot}]" + result[idx + len(value):]

    if ref and ref in result:
        result = result.replace(ref, "[ref]")

    return result
