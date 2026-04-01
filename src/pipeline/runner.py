"""Pipeline orchestrator for MLP4CS."""
import re
import json
import time
from tqdm import tqdm
from src.config import MAX_RETRIES, MAX_DIALOGUES, TARGET_DOMAINS, BOOKING_REQUIRED_SLOTS
from src.pipeline.dst import dst, DST_SLOTS
from src.pipeline.policy import policy
from src.pipeline.response_generator import response_generator
from src.pipeline.supervisor import supervisor
from src.pipeline.lexicalizer import lexicalize
from src.pipeline.memory import memory
from src.utils import build_tomiinek_turn, build_custom_turn, format_history, format_slots, normalize_slot_value
from src.db import load_db, find_entity, book_entity
from src.models.llm import call_model

# Load DB once at module level to avoid reloading every turn
_HOTEL_DB = load_db("hotel")
_RESTAURANT_DB = load_db("restaurant")


def run_turn_single(user_utterance: str, history: list[dict], accumulated_slots: dict[str, str], model_config: dict[str, str]) -> tuple[str, str, dict[str, str], list[dict], dict, dict]:
    """
    Run one full pipeline turn for Exp1 as a single LLM call.

    Passes full hotel and restaurant databases directly in the system prompt.
    LLM selects best matching entity, extracts slots, and generates response in one call. No separate DST or DB lookup step.

    Args:
        user_utterance: current user message
        history: list of previous turns
        accumulated_slots: full belief state from previous turns
        model_config: dict with key 'single' pointing to model name
    Returns:
        tuple of (delex_response, lex_response, accumulated_slots, history, tomiinek_turn, custom_turn)
    """
    system_prompt = (
        f"You are an end-to-end customer service assistant for {', '.join(TARGET_DOMAINS)} bookings.\n\n"
        f"HOTEL DATABASE:\n{json.dumps(_HOTEL_DB, indent=2)}\n\n"
        f"RESTAURANT DATABASE:\n{json.dumps(_RESTAURANT_DB, indent=2)}"
    )

    history_str = format_history(history)
    slots_str = format_slots(accumulated_slots)
    slot_list = ", ".join(DST_SLOTS["hotel"] + DST_SLOTS["restaurant"])

    user_prompt = (
        f"{history_str}\n"
        f"USER: {user_utterance}\n\n"
        f"Current belief state: {slots_str}\n\n"
        f"Valid domains: {', '.join(TARGET_DOMAINS)}\n"
        f"Valid intents: find_hotel, book_hotel, find_restaurant, book_restaurant\n"
        f"Valid slots: {slot_list}\n\n"
        f"Booking policy — required slots:\n"
        f"- book_hotel: {', '.join(BOOKING_REQUIRED_SLOTS['book_hotel'])}\n"
        f"- book_restaurant: {', '.join(BOOKING_REQUIRED_SLOTS['book_restaurant'])}\n\n"
        f"Extraction rules:\n"
        f"- ONLY extract slot values EXPLICITLY stated by the user.\n"
        f"- DO NOT infer or assume missing values.\n\n"
        f"Response rules:\n"
        f"1. Find the best matching entity from the DATABASE in the system prompt based on the current belief state constraints.\n"
        f"2. If no matching entity found, tell user nothing was found.\n"
        f"3. If booking intent and ALL required slots present, confirm booking with [ref] placeholder.\n"
        f"4. If booking intent but required slots MISSING, ask ONLY for those missing slots.\n"
        f"5. NEVER use real entity names or details — ALWAYS use placeholders:\n"
        f"   [hotel_name], [hotel_phone], [hotel_address], [hotel_postcode]\n"
        f"   [restaurant_name], [restaurant_phone], [restaurant_address], [restaurant_postcode]\n"
        f"   [ref] for booking reference.\n\n"
        f"6. Recommend ONE entity only and use each placeholder ONCE.\n"
        f"7. If user says goodbye or thanks, respond with a farewell message. Set intent to None.\n"
        f"Respond with valid JSON only:\n"
        f'{{"domain": "...", "intent": "...", "slots": {{}}, "response": "..."}}'
    )

    # Call LLM
    # response = call_model(model_config["single"], user_prompt, system_prompt)

    # Retry up to 3 times on empty response as Haiku occasionally returns empty due to rate limiting
    for attempt in range(3):
        response = call_model(model_config["single"], user_prompt, system_prompt)
        if response.text.strip():
            break
        time.sleep(2)

    total_cost = response.cost
    total_time = response.response_time

    # Parse JSON string from LLM into Python dict, e.g., '{"domain": "restaurant", ...}' → parsed["domain"] = "restaurant"
    domain, intent, new_slots, delex_response = None, None, {}, ""
    try:
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        # Haiku embeds newlines/tabs inside JSON string values (e.g. multi-line response field) which breaks json.loads()
        # The newlines (\n) inside the "response" value break json.loads() as it expects the string to be on one line, so we replace all control characters (including newlines, tabs) with spaces before parsing
        # This allows the LLM to output multi-line responses without breaking the JSON format
        raw = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)

        # If Haiku outputs plain text instead of JSON, try to extract JSON object from within the text (raw does NOT start with "{" → enters the if block and re.search finds the first { to last })
        if not raw.strip().startswith("{"):
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group()

        parsed = json.loads(raw)
        domain = parsed.get("domain")
        intent = parsed.get("intent")

        # new_slots = {k: normalize_slot_value(str(v).lower()) for k, v in parsed.get("slots", {}).items()}
        # Filter to valid slots only as LLM may hallucinate invalid slot names (e.g. total_cost)
        valid_slots = set(DST_SLOTS["hotel"] + DST_SLOTS["restaurant"])
        new_slots = {
            k.replace("_", "-"): normalize_slot_value(str(v).lower())
            for k, v in parsed.get("slots", {}).items()
            if k.replace("_", "-") in valid_slots
        }

        # Slot-specific normalization (same as parse_dst_output in dst.py)
        for k, v in new_slots.items():
            if k in ("hotel-parking", "restaurant-parking", "hotel-internet") and v in ("free", "free wifi", "free internet"):
                new_slots[k] = "yes"

        delex_response = parsed.get("response", "")
        # Merge new slots into accumulated belief state
        accumulated_slots = {**accumulated_slots, **new_slots}
    except (ValueError, KeyError, TypeError) as e:
        print(f"Failed to parse LLM JSON output: {e}")
        delex_response = "I'm sorry, I didn't understand. Could you please clarify?"

    # Policy check (violations based on parsed intent and merged slots, used in custom metrics)
    violations = policy(intent, accumulated_slots)

    # DB lookup for lexicalization to find matching entity based on parsed slots
    db_results = []
    if domain and domain in TARGET_DOMAINS:
        domain_slots = {k: v for k, v in accumulated_slots.items() if k.startswith(domain)}
        if intent in (f"book_{domain}",) and not violations:
            booking = book_entity(domain, domain_slots)
            if booking["success"]:
                db_results = [booking["entity"]]
        else:
            db_results = find_entity(domain, domain_slots)

    # Inject recommended entity name into accumulated_slots (standard MultiWOZ practice)
    # if db_results and domain:
    #     entity_name_key = f"{domain}-name"
    #     if entity_name_key not in accumulated_slots:
    #         accumulated_slots[entity_name_key] = db_results[0]["name"]

    # Supervisor check (no retry in Exp1, used only for custom metrics (valid flag))
    valid, _ = supervisor(delex_response, violations, db_results, intent, user_utterance, domain)

    # Lexicalize by replacing placeholders with real entity values
    lex_response = lexicalize(delex_response, db_results, domain or "")

    # Memory update by storing lex_response in history
    history = memory(history, user_utterance, lex_response)

    # Build output dicts
    tomiinek_turn = build_tomiinek_turn(delex_response, accumulated_slots, domain)
    custom_turn = build_custom_turn(
        domain, intent, accumulated_slots, violations, delex_response, lex_response,
        db_results, valid, 1, user_utterance, total_cost, total_time
    )

    # print(f"\nUser utterance: {user_utterance}")
    # print(f"Domain: {domain} | Intent: {intent}")
    # print(f"New slots: {new_slots} | Accumulated slots: {accumulated_slots}")
    # print(f"Violations: {violations}")
    # print(f"DB results: {db_results}")
    # print(f"Delex response: {delex_response} | Lex response: {lex_response}")
    # print(f"Valid: {valid}")

    return delex_response, lex_response, accumulated_slots, history, tomiinek_turn, custom_turn


def run_dialogue_single(dialogue: dict, model_config: dict[str, str]) -> tuple[str, list[dict], list[dict]]:
    """
    Run full dialogue for Exp1 using single LLM call per turn.

    Args:
        dialogue: dialogue dict from dataset
        model_config: dict with key 'single'
    Returns:
        tuple of (dialogue_id, tomiinek_turns, custom_turns)
    """
    dialogue_id = dialogue["dialogue_id"]
    history: list[dict] = []
    accumulated_slots: dict[str, str] = {}
    tomiinek_turns: list[dict] = []
    custom_turns: list[dict] = []

    for turn in dialogue["turns"]:
        if turn["speaker"] != "USER":
            continue
        _, _, accumulated_slots, history, tomiinek_turn, custom_turn = run_turn_single(turn["utterance"], history, accumulated_slots, model_config)
        tomiinek_turns.append(tomiinek_turn)
        custom_turns.append(custom_turn)

    return dialogue_id, tomiinek_turns, custom_turns


def run_turn(user_utterance: str, history: list[dict], accumulated_slots: dict[str, str], model_config: dict[str, str], zeroshot: bool = True) -> tuple[str, str, dict[str, str], list[dict], dict, dict]:
    """
    Run one full pipeline turn.

    Args:
        user_utterance: current user message
        history: list of previous turns as dicts with 'speaker' and 'utterance'
        accumulated_slots: full belief state from previous turns
        model_config: dict with keys 'dst' and 'response_generator'
        zeroshot: if True, use zero-shot prompt (Exp2), else fine-tuned (Exp3)
    Returns:
        tuple of (delex_response, lex_response, accumulated_slots, history, tomiinek_turn, custom_turn)
    """
    # Step 1: dialogue state tracking
    domain, intent, accumulated_slots, dst_cost, dst_time = dst(user_utterance, history, accumulated_slots, model_config)

    total_cost = dst_cost
    total_time = dst_time

    # If domain is None, infer from most recently added slot prefix,
    # e.g., user says "yes, book it" → DST returns None because no explicit domain mentioned → infer from last slot in accumulated_slots (e.g., "hotel-name" → "hotel")
    # if domain is None and accumulated_slots:
    #     last_key = list(accumulated_slots.keys())[-1]
    #     domain = last_key.split("-")[0]

    # Step 2: policy check
    violations = policy(intent, accumulated_slots)

    # Step 3: response generation with retry loop
    feedback = None
    valid = False
    attempts = 0
    delex_response = ""
    db_results = []

    for attempt in range(MAX_RETRIES):
        attempts = attempt + 1
        delex_response, db_results, rg_cost, rg_time = response_generator(
            history, user_utterance, domain, intent, accumulated_slots, violations, model_config, zeroshot=zeroshot, feedback=feedback
        )

        total_cost += rg_cost
        total_time += rg_time

        valid, feedback = supervisor(delex_response, violations, db_results, intent, user_utterance, domain)
        if valid:
            break

    # Inject recommended entity name into accumulated_slots (standard MultiWOZ practice) so belief state = user's constraints + recommended by the system entity
    # GT annotators always add entity name after system recommends it, we inject db_results[0]["name"]
    # if db_results and domain:
    #     entity_name_key = f"{domain}-name"
    #     if entity_name_key not in accumulated_slots:
    #         accumulated_slots[entity_name_key] = db_results[0]["name"]

    # Step 4: lexicalize
    lex_response = lexicalize(delex_response, db_results, domain or "")

    # Step 5: update history with lexicalized response
    history = memory(history, user_utterance, lex_response)

    # Step 6: build output dicts
    tomiinek_turn = build_tomiinek_turn(delex_response, accumulated_slots, domain)
    custom_turn = build_custom_turn(
        domain, intent, accumulated_slots, violations, delex_response, lex_response,
        db_results, valid, attempts, user_utterance, total_cost, total_time
    )

    return delex_response, lex_response, accumulated_slots, history, tomiinek_turn, custom_turn


def run_dialogue(dialogue: dict, model_config: dict[str, str], zeroshot: bool = True) -> tuple[str, list[dict], list[dict]]:
    """
    Run the full pipeline on a single dialogue.

    Args:
        dialogue: dialogue dict with 'dialogue_id', 'services', 'turns'
        model_config: dict with keys 'dst' and 'response_generator'
        zeroshot: if True, use zero-shot prompt (Exp2), else fine-tuned (Exp3)
    Returns:
        tuple of (dialogue_id, tomiinek_turns, custom_turns)
    """
    dialogue_id = dialogue["dialogue_id"]
    accumulated_slots: dict[str, str] = {}
    history: list[dict] = []
    tomiinek_turns: list[dict] = []
    custom_turns: list[dict] = []

    for turn in dialogue["turns"]:
        if turn["speaker"] != "USER":
            continue

        _, _, accumulated_slots, history, tomiinek_turn, custom_turn = run_turn(turn["utterance"], history, accumulated_slots, model_config, zeroshot=zeroshot)

        tomiinek_turns.append(tomiinek_turn)
        custom_turns.append(custom_turn)

    return dialogue_id, tomiinek_turns, custom_turns


def run_experiment(experiment_name: str, model_config: dict[str, str], split: str, zeroshot: bool = True, exp_id: int = 2, single: bool = False) -> tuple[dict, dict]:
    """
    Run the full pipeline on all dialogues in a split and save results.

    Args:
        experiment_name: name for output files e.g. 'exp2_homo_gpt'
        model_config: dict with keys 'dst' and 'response_generator'
        split: dataset split to evaluate on, one of 'train', 'dev', 'test'
        zeroshot: if True, use zero-shot prompt (Exp2), else fine-tuned (Exp3)
        exp_id: experiment number for display, one of 1, 2, 3
        single: if True, use run_dialogue_single() for Exp1, default False
    Returns:
        tuple of (tomiinek_results, custom_results) dicts
    """
    # Load dialogues
    from src.data.loader import load_split

    dialogues = load_split(split)
    if MAX_DIALOGUES is not None:
        dialogues = dialogues[:MAX_DIALOGUES]

    tomiinek_results: dict = {}
    custom_results: dict = {}

    # Print headers
    print(f"\n>>> Experiment {exp_id}: {experiment_name}")
    print(f"{'-'*60}")

    if exp_id == 1:
        model_name = model_config.get("single", "unknown")
        print(f"Exp{exp_id} | {model_name} | {len(dialogues)} dialogues")
    else:
        config_name = experiment_name.split("_", 1)[-1]  # "exp2_homo_gpt" → "homo_gpt"
        print(f"Exp{exp_id} | {config_name} | {len(dialogues)} dialogues")
        for role, model in model_config.items():
            print(f"  {role:}: {model}")

    # Loop over dialogues
    for i, dialogue in enumerate(tqdm(dialogues, desc=f"  {experiment_name}", unit="dlg", leave=True)):
        dialogue_id = dialogue["dialogue_id"]
        num_turns = len([t for t in dialogue["turns"] if t["speaker"] == "USER"])
        tomiinek_id = dialogue_id.replace(".json", "").lower()  # sng0073 instead of SNG0073.json

        print(f"  [{i+1}/{len(dialogues)}] {dialogue_id}... (turns={num_turns})")

        if single:
            dialogue_id, tomiinek_turns, custom_turns = run_dialogue_single(dialogue, model_config)
        else:
            dialogue_id, tomiinek_turns, custom_turns = run_dialogue(dialogue, model_config, zeroshot=zeroshot)

        tomiinek_results[tomiinek_id] = tomiinek_turns
        custom_results[dialogue_id] = custom_turns

    return tomiinek_results, custom_results
