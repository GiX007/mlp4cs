"""Conversation history manager for MLP4CS pipeline."""


def memory(history: list[dict], user_utterance: str, lex_response: str) -> list[dict]:
    """
    Append current turn's user utterance and system response to history so every subsequent turn's components have full dialogue context.

    Args:
        history: list of previous turns as dicts with 'speaker' and 'utterance'
        user_utterance: current user message
        lex_response: lexicalized system response for this turn
    Returns:
        updated history list with two new turns appended

    Each turn is a separate function call so history does not persist automatically. memory() carries it forward manually between turns in runner.py.
    Lexicalized response (and not delex) used as input. Storing delex responses with placeholders like [restaurant_name] causes DST to extract placeholder strings
    as real slot values in subsequent turns, corrupting the belief state. Real entity names must appear in history.
    """
    history.append({"speaker": "USER", "utterance": user_utterance})
    history.append({"speaker": "SYSTEM", "utterance": lex_response})
    return history
