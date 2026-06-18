# MLP4CS Development Notes

Implementation decisions, known limitations, and observations.

---

## 1. Dataset limitations

### Multi-value slot lists
MultiWOZ 2.2 stores slot values as lists to account for spelling variants, e.g., `{"hotel-name": ["rosas bed and breakfast", "rosa's"]}`.
All values refer to the same entity as they are annotation variants, not conflicting user preferences. We always take `v[0]` as the canonical form.

Affected slots (hotel + restaurant domains only):
- train: 1071 / 61366 (1.7%)
- dev: 94 / 5427 (1.7%)
- test: 70 / 5856 (1.2%)

Impact: negligible. Tomiinek uses fuzzy DB matching (unaffected). Custom metrics use the same `v[0]` convention on both GT and predicted sides (consistent).
In rare cases where the LLM predicts the second variant (e.g., "rosa's" instead of "rosas bed and breakfast"), JGA and Slot F1 are penalized, 
but this affects <2% of slots and is a known MultiWOZ annotation limitation accepted across published work.

### Annotation errors in GT responses
Some GT system responses contain annotation errors where slot values appear in unexpected positions. 
Example: in PMUL0540.json Turn 5, booking reference `[ref]` appears where the restaurant name should be: "Certainly! I have made a reservation at [ref] for the time specified..."
These are GT annotation errors, not pipeline bugs. Cases are rare and accepted as known MultiWOZ 2.2 noise. 

Impact: negligible across all evaluation paths. BLEU may be penalized on affected turns where GT reference itself is wrong. 
Fine-tuning learns from the majority of correct examples. Custom metrics are unaffected as they do not compare against GT responses.

---

## 2. DST: design decisions and known limitations
DST extracts domain, intent, and slot-value pairs per turn. Output is parsed JSON and downstream steps depend on the slot dict being correct, 
so robust parsing and explicit prompt rules matter.

### Domain/intent parsing edge cases
`parse_dst_output()` handles 5 LLM output combinations:

1. domain=valid, intent=valid, matching → use both as-is (normal case)
2. domain=valid, intent=invalid/multiple → use domain, intent=None
3. domain=invalid/multiple, intent=valid → infer domain from intent (e.g., "book_hotel" → "hotel")
4. domain=None, intent=None → both returned as None, pipeline continues gracefully
5. domain=valid, intent=valid but mismatched → trust domain, set intent=None

Design decision: slots accumulate correctly in ALL 5 cases regardless of domain/intent parsing outcome. 
JGA and Slot F1 are never affected by domain/intent failures.
domain=None only affects `active_domains` in Tomiinek output (optional, evaluator can infer).
intent=None only affects policy as no violations raised, pipeline continues safely.

Known limitation: multi-domain turns where the user addresses both hotel and restaurant simultaneously result in domain=None and intent=None (e.g., Turn 4 of PMUL4398.json). 
This is a tradeoff of our single-domain-per-turn pipeline design. Slots for both domains are still extracted and accumulated correctly.
See Section 5 for how these turns are handled in evaluation.

### Prompt: dontcare extraction rule
Added explicit rule to `build_dst_prompt()`: if user expresses no preference (e.g., "any food", "doesn't matter") → extract that slot with value `dontcare`.
Without this rule, LLMs omit the slot entirely. GT always has `slot=dontcare` in these cases so missing it causes JGA=False and lower Slot Recall.

Impact: improved JGA and Slot F1 in Exp2. Consistent across Experiment 2 and Experiment 3 as the same rule is used in fine-tuning data via `build_dst_prompt()`.
See Section 5 for observed failure cases in error analysis.

### Over-extraction patterns
Two known LLM extraction errors that affect JGA:

**City name as area value:** LLM extracts city names as area values (e.g., "visiting cambridge" → `restaurant-area=cambridge`). 
Valid area values are: north, south, east, west, centre. Cambridge is the city name, not an area. The DST prompt includes a rule ("DO NOT extract city names as area values") but LLMs occasionally ignore it.
Confirmed in error analysis: exp2_homo_haiku, MUL0126.json.

**Request confused with constraint:** User asks "what price range are they in?" → LLM extracts `restaurant-pricerange=dontcare`. 
GT has no pricerange slot because the user is requesting information, not providing a constraint. 
The DST prompt includes a rule ("Only extract when user is PROVIDING information, not REQUESTING it") but this distinction is difficult for zero-shot LLMs.

Impact: both cause JGA=False. One wrong extraction propagates through all subsequent turns. 
Expected to improve with fine-tuning (Experiment 3) as the model learns correct extraction patterns from training data.

---

## 3. Response Generation: design decisions and known limitations
Response Generation produces the system's reply per turn. Output is a delexicalized response with placeholders for entity values. 
Downstream steps (lexicalize, memory) depend on placeholder discipline.

### Response generator edge cases
`response_generator()` handles 6 cases per turn:

1. domain=None → hardcoded fallback response, empty db_results (multi-domain or goodbye turns)
2. Violations exist → skip DB, LLM asked to request missing slots from user
3. Book intent, no violations, entity found → `book_entity()`, response with `[ref]` placeholder
4. Book intent, no violations, no entity found → booking fails, LLM informs user
5. Find intent, results found → `find_entity()`, first entity passed to LLM prompt
6. Find intent, no results → LLM asks user to relax constraints

Design decisions:
- DB lookup happens BEFORE LLM call and LLM sees real entity data
- Violations checked BEFORE DB so no wasted DB call if slots missing
- Only `db_results[:1]` passed to prompt to reduce verbosity
- `domain_slots` filters `accumulated_slots` to active domain only before DB lookup
- `db_results` returned alongside `delex_response` for supervisor validation and lexicalization

### Response generator prompt modes
Two modes controlled by `zeroshot` flag:
- `zeroshot=True` (Exp2 and Exp3 inference): history + domain + intent + slots + DB results + violations + placeholder instructions
- `zeroshot=False` (fine-tuning data generation only): history + domain + intent + slots only. Mirrors fine-tuning input format

Only active domain placeholders included: `[{domain}_name]`, `[{domain}_phone]`, `[{domain}_address]`, `[{domain}_postcode]`, `[ref]`.

### Lexicalization: [ref] placeholder
`lexicalize()` accepts `ref=""` by default. The pipeline never passes a real booking reference as users see `[ref]` in lexicalized responses. 
For evaluation this works as `[ref]` presence signals booking confirmation. 

### Delexicalization on cross-domain turns
Delexicalization is applied per active domain only. 
On multi-domain turns where the response mentions a secondary domain entity, that entity remains lexicalized.

Example (active domain = restaurant): "[restaurant_name]'s phone is [restaurant_phone]. As far as hotels go,  I recommend the University Arms Hotel in the center of town."
"University Arms Hotel" is NOT delexicalized because hotel is not the active domain.

Impact: on Tomiinek is none, as Tomiinek evaluates only the active domain per turn. Regarding BLEU, minor penalty on multi-domain turns where GT references are fully delexicalized. 
Affects a minority of turns as it is an accepted tradeoff for simplicity.

---

## 4. Experiments Overview

### Research questions
Does architectural decomposition into specialized LLMs improve reliability over a single-LLM baseline, and does role-specific fine-tuning yield further improvement?
Additionally: can small fine-tuned open-source models compete with larger commercial API models (GPT-4o-mini, Claude 3 Haiku)?

### What we compare
**Experiment 1: Single LLM (everything in one call):**
```
User Turn → [Single LLM: DST + Entity Selection + Response] → policy() → DB lookup → supervisor() → lexicalize() → Evaluate
```
One LLM handles all tasks. Full DB in prompt. No retry.
Tested with: API models and open-source models.
Establishes the baseline: how well can a single model do without any architectural help?

**Experiment 2: Modular pipeline (two specialized LLMs):**
```
User Turn → [DST LLM] → policy() + DB lookup → [ResponseGen LLM] → supervisor() (with retry) → lexicalize() → Evaluate
```
Two separate LLM calls, each focused on one task. DB queried between them.
Tested with: homogeneous (same model for both) and heterogeneous (different models) configurations, using the same models as in Experiment 1.
Tests: does splitting responsibilities improve accuracy over Experiment 1?

**Experiment 3: Fine-tuned modular pipeline (LoRA adapters):**
```
User Turn → [DST LoRA] → policy() + DB lookup → [ResponseGen LoRA] → supervisor() (with retry) → lexicalize() → Evaluate
```
Same pipeline as Experiment 2, but DST and ResponseGen powered by LoRA fine-tuned open-source models.
Tested with: several fine-tuned configurations.

Tests the questions:
1. Do small open-source fine-tuned models outperform their zero-shot versions from Experiment 2?
2. Can small open-source fine-tuned models approach the performance of commercial API models from Experiment 1/Experiment 2?

### Shared components across all three experiments
- `policy()`: checks if required booking slots are present before allowing a booking
- `supervisor()`: validates the response, triggers retry if invalid (max 2 attempts)
- `lexicalize()`: replaces placeholders with real entity values from DB results
- `memory()`: stores the final lexicalized response in history for subsequent turns

---

## 5. Execution flow
This section traces what happens, function by function, when an experiment is run. Each experiment has its own entry point in `src/experiments/`, but all three converge
on the same evaluation and saving steps. The traces below cover the inference path only. Exp3's fine-tuning pipeline (dataset generation + LoRA 
training) is covered separately at the end.

### Experiment 1: Single-LLM baseline
Entry: `python -m src.main` with `run_experiment_1()` uncommented.

1. `src/main.py` calls `run_experiment_1(split="test")`
2. `src/experiments/exp1.py` → `run_experiment_1()` loops over each config in `EXP1_CONFIGS` (e.g., `exp1_gpt`, `exp1_haiku`, `exp1_qwen3_14b`)
3. `src/pipeline/runner.py` → `run_experiment(single=True)` loads dialogues via `load_split(split)` and dispatches to `run_dialogue_single()` because `single=True`
4. `src/pipeline/runner.py` → `run_dialogue_single()` iterates USER turns in the dialogue and calls `run_turn_single()` for each
5. `src/pipeline/runner.py` → `run_turn_single()` builds one prompt containing full hotel + restaurant DBs and booking rules, calls the single LLM once via `call_model()`, parses the JSON output
6. Same step, post-processing runs in order: `policy()` (re-checks the LLM's booking decision against the parsed slots; on violations, `book_entity()` is skipped and the violation is recorded for metrics) → `find_entity()` or `book_entity()` → `supervisor()` (no retry, valid flag only) → `lexicalize()` → `memory()`
7. Same step, packaging: `build_tomiinek_turn()` and `build_custom_turn()` produce the two per-turn output dicts
8. Back in `run_experiment()`, per-dialogue lists are merged into `tomiinek_results` (keyed by lowercased `dialogue_id` without `.json`) and `custom_results` (keyed by raw `dialogue_id`)
9. Back in `exp1.py`, `tomiinek_results` is saved to `results/<experiment_name>/<experiment_name>_tomiinek_input.json`
10. `src/evaluation/evaluator.py` → `evaluate_experiment()` computes custom metrics (JGA, Slot F1, hallucination, policy, etc.) from `custom_results`
11. `src/evaluation/tomiinek.py` → `run_tomiinek()` wraps `mwzeval.metrics.Evaluator(bleu=True, success=True)` and returns `{inform, success, bleu, combined}`
12. `src/evaluation/reporter.py` → `save_results()` writes three JSONs (dataset, dialogues, turns) under `results/<experiment_name>/`
13. Back in `exp1.py`, `print_table()` displays a comparison table across all configs in `EXP1_CONFIGS`

### Experiment 2: Modular pipeline (zero-shot)
Entry: `python -m src.main` with `run_experiment_2()` uncommented.

1. `src/main.py` calls `run_experiment_2(split="test")`
2. `src/experiments/exp2.py` → `run_experiment_2()` loops over each config in `EXP2_CONFIGS` (homogeneous and heterogeneous combos, e.g., `exp2_homo_gpt`, `exp2_hetero_qwen3_14b_phi4_14b`)
3. `src/pipeline/runner.py` → `run_experiment(single=False, zeroshot=True)` loads dialogues via `load_split(split)` and dispatches to `run_dialogue()` because `single=False`
4. `src/pipeline/runner.py` → `run_dialogue()` iterates USER turns in the dialogue and calls `run_turn()` for each
5. `src/pipeline/runner.py` → `run_turn()` runs the modular pipeline:
   - `dst()` (first LLM call) extracts domain, intent, and slots → updates `accumulated_slots`
   - `policy(intent, accumulated_slots)` produces `violations` (used to *steer* the next LLM call, not just instrument it)
   - DB lookup runs *before* ResponseGen: `find_entity()` for find intents, `book_entity()` for book intents when `not violations`
   - `response_generator()` (second LLM call) receives the slots, violations, and `db_results[:1]`, with `zeroshot=True` the prompt includes placeholder instructions
   - `supervisor()` validates the delex response, and if invalid, the retry loop calls `response_generator()` again with feedback (max 2 attempts)
6. Same step, post-processing: `lexicalize()` → `memory()`
7. Same step, packaging: `build_tomiinek_turn()` and `build_custom_turn()` produce the two per-turn output dicts
8. Back in `run_experiment()`, per-dialogue lists are merged into `tomiinek_results` (keyed by lowercased `dialogue_id` without `.json`) and `custom_results` (keyed by raw `dialogue_id`)
9. Back in `exp2.py`, `tomiinek_results` is saved to `results/<experiment_name>/<experiment_name>_tomiinek_input.json`
10. `src/evaluation/evaluator.py` → `evaluate_experiment()` computes custom metrics (JGA, Slot F1, hallucination, policy, etc.) from `custom_results`
11. `src/evaluation/tomiinek.py` → `run_tomiinek()` wraps `mwzeval.metrics.Evaluator(bleu=True, success=True)` and returns `{inform, success, bleu, combined}`
12. `src/evaluation/reporter.py` → `save_results()` writes three JSONs (dataset, dialogues, turns) under `results/<experiment_name>/`
13. Back in `exp2.py`, `print_table()` displays a comparison table across all configs in `EXP2_CONFIGS`

### Experiment 3: Modular pipeline (LoRA fine-tuned)
Entry: `python -m src.main` with `run_experiment_3()` uncommented.

The inference path is identical to Experiment 2 at the function level. Only the underlying LLMs change: the DST and ResponseGen calls now hit LoRA fine-tuned
open-source models instead of zero-shot ones. 

### Experiment 3: Fine-tuning pipeline
The LoRA adapters used at inference (step 5 above) are produced offline by a separate pipeline under `scripts/`. 
This runs once per (model, role) combination, not per run.

**Phase 1. Dataset generation (local machine):**
- `scripts/build_ft_data.py` loads the MultiWOZ 2.2 train split via `load_split("train")`
- For each dialogue, iterates USER turns and builds two training examples per turn:
   - **DST example:** input = `build_dst_prompt(history, user_utterance, accumulated_slots)`, target = GT domain + intent + slots as JSON
   - **ResponseGen example:** input = `build_respgen_prompt(..., zeroshot=False)` (stripped format: history + domain + intent + slots only, no DB results or placeholder rules), target = GT delexicalized response
- Examples are written to `data/finetune_data/dst_train.json` and `data/finetune_data/respgen_train.json`
- Same process on the dev split → `dst_dev.json` and `respgen_dev.json` for validation during training

**Phase 2. LoRA training (Leonardo cluster, A100):**
- `scripts/finetune.py` is launched per (base_model, role) pair
- Unsloth loads the base model in 4-bit (QLoRA), attaches LoRA adapters to attention layers (rank=16)
- Training loop runs for N epochs on the JSONL file, validating on the dev JSONL
- Best checkpoint (lowest dev loss) is saved as a LoRA adapter under `data/finetuned_models/<base_model>_<role>/`

**Phase 3. Inference (back to the main flow):**
- Exp3 configs in `src/config.py` point each role to the corresponding adapter path
- At inference, `call_model()` loads the base model + adapter and runs the modular pipeline described in the Exp3 inference trace above

Train/inference prompt consistency is guaranteed because both phases call the same `build_dst_prompt()` and `build_respgen_prompt()` 
functions, the only difference is that training uses `zeroshot=False` (stripped prompt) and inference uses `zeroshot=True` (full prompt). 
The FT model learns the task from examples and can handle either format.

---

## 6. Evaluation: Custom Metrics
Custom metrics measure pipeline-internal behavior (DST accuracy, hallucination, policy compliance) that Tomiinek does not cover. They are 
computed from `custom_results` produced per turn by `run_turn()` / `run_turn_single()`.

### What is counted per turn
The `skip` flag is set when BOTH predicted domain and intent are None (multi-domain or farewell turns). This affects metrics differently:

**Computed per turn (no skip):**
- JGA: accumulated slots are valid regardless of domain/intent
- Slot P / Slot R / Slot F1: same reason
- Hallucination: checked on lex_response against db_results
- Policy compliance: checked via violations list + booking confirmation signals
- System correctness: composite of hallucination + policy
- Cost and latency: always tracked

**Skipped when domain=None AND intent=None:**
- Domain accuracy (precision, recall, F1) → returns None, excluded from averages
- Intent accuracy (precision, recall, F1) → returns None, excluded from averages
- Action accuracy → returns None, excluded from averages

Rationale for exclusion: multi-domain turns produce None by design (single-domain pipeline), and conversational turns (greetings, goodbyes) have no task-oriented domain/intent. 
Including these as wrong predictions would unfairly penalize the pipeline for cases outside its design scope.

**Special averaging rules at dataset level:**
- Hallucination: averaged only over turns where `entity_mentioned=True`
- Policy: averaged over ALL turns (compliant turns count as 1.0)
- Booking rate: dialogue-level only, excludes dialogues with no booking intent
- All other metrics: micro-averaged over all turns (standard in MultiWOZ literature)

**Pre-evaluation exclusion in `evaluate_experiment()`:**
- SYSTEM turns: completely skipped (only USER turns trigger evaluation)
- Goodbye/farewell turns: skipped when `gt_domains` is empty
- These turns never reach `evaluate_turn()` at all

---

### DST metrics (JGA, Slot F1) 

#### How JGA is computed
JGA uses real slot values (e.g., `{"restaurant-name": "golden wok"}`), not placeholders. JGA = True only if ALL slot-value pairs match exactly across ALL domains.
MultiWOZ 2.2 stores accumulated belief state per frame per domain. Our evaluator merges across domain frames into a single flat dict via `gt_accumulated.update()` 
to enable cross-domain comparison against the pipeline's own flat accumulated slots.

#### How Slot F1 is computed
Computed on the same accumulated flat prefixed belief state as JGA. Unlike JGA, gives partial credit per correct slot. `dontcare` is included in both predicted 
and GT sides, consistent with GT annotations, to avoid artificially low recall.

#### dontcare vs none distinction
This convention affects both JGA and Slot F1.

Some leaderboard models (e.g., SimpleTOD) do not distinguish between `dontcare` (user explicitly expressed no preference) and `none` (slot never mentioned). 
This inflates their numbers because they get credit for unextracted slots. The [official MultiWOZ repository](https://github.com/budzianowski/multiwoz) 
acknowledges this issue and refers to [CheckDST](https://github.com/wise-east/checkdst) for corrected evaluation.

Our pipeline correctly distinguishes `dontcare` from `none`. Our JGA and Slot F1 numbers are lower than some published leaderboard results but methodologically correct.

#### Challenge: entity name mismatch
The pipeline always uses `db_results[0]`, the first matching entity. The GT annotator may have chosen a different entity from the same result set.

Example: user wants north + moderate restaurant → DB returns `[nirala, golden wok]`.
Pipeline predicts `restaurant-name=nirala`, GT has `restaurant-name=golden wok`.
JGA = False despite correct constraints.

This affects ALL published MultiWOZ systems and is a dataset limitation, not a pipeline bug. No mitigation possible as the pipeline cannot know which entity 
the GT annotator chose from the matching set.

#### Challenge: missing entity name in predicted slots
The pipeline does not inject the recommended entity name into `accumulated_slots`. The DST prompt instructs the model to extract only values explicitly stated 
by the user. Whether the LLM also picks up entity names from system recommendations in conversation history is model-dependent and inconsistent.

GT always contains the name once the system recommends it, causing JGA = False for subsequent turns even when all other slots match perfectly.

We tested injecting `db_results[0]["name"]` but it hurt JGA: our DB returns entities in a different order than the GT annotator chose, so the wrong name cascades 
as a mismatch through all subsequent turns. Without injection the name is simply absent (still JGA = False) but at least no wrong name propagates.

Leaderboard models avoid this because they train end-to-end on GT belief states and learn to predict the exact GT entity name. Zero-shot pipelines cannot 
replicate this. For Experiment 3 (fine-tuned), this limitation may be reduced as the model trains on GT belief states that include entity names and learns to 
extract them from history naturally.

Tradeoff: cleaner belief state, slightly lower JGA.

#### Challenge: cascading slot errors
A single wrong slot in an early turn propagates to all subsequent turns because `accumulated_slots` grows turn by turn and never resets.

Example:
- Turn 2: DST extracts `restaurant-area=cambridge` (wrong — city name, not an area)
- Turns 3–4: `accumulated_slots` still has `restaurant-area=cambridge`, but GT has `restaurant-area=centre`
- Result: JGA = False for turns 3 AND 4, caused by a single turn 2 error

This is by design accumulated belief state mirrors how a real dialogue system tracks constraints across turns.

---

### Policy compliance check
Evaluated per USER turn but checks the system's response to that turn. Policy compliance is NOT simply `len(violations) == 0`. 
A turn with violations is still compliant if the pipeline correctly asked the user for missing slots instead of confirming a booking. 
Non-compliant only when violations exist AND the response contains the `[ref]` placeholder (e.g., the system confirmed a booking despite missing required slots).

---

### Action accuracy with rule-based mapping
Predicted action is derived from intent + violations (not from LLM output):
- `find_*` intent → "inform"
- `book_*` intent + violations → "request" (asking for missing slots)
- `book_*` intent + no violations → "book"

GT action is derived from SYSTEM turn `dialog_act` keys. Mapping decision: `Booking-Inform` maps to "inform" not "book" and only `Booking-Book` confirms an actual booking. 
Turns where either action is undetermined (None) are excluded.

---

### Hallucination

#### Detection method
All known entity names are loaded from both hotel and restaurant DBs.
Any known name appearing in `lex_response` but NOT in this turn's `db_results` is counted as a hallucination. 
Turns where no entity is mentioned in the response are excluded from the hallucination average (not penalized, not rewarded).

#### Sub-cases across experiments
**True hallucination (primarily Experiment 1):**
LLM generates entity details from training knowledge instead of DB results.
Example: *"Cotto is at Regent Street."* The LLM knows Cotto from training, but the DB returned a different entity. Entity NOT in `db_results` → true hallucination.
Impact: even with the full DB in the prompt, single-call LLMs hallucinate and validates the modular architecture's value.

**Fine-tuning hallucination amplification (Experiment 3 only):**
Fine-tuned response generators sometimes emit real entity names where their zero-shot counterparts would have emitted placeholders. Cause: the fine-tuning training 
data was built by string-matching delexicalization on the gold MultiWOZ responses, which only replaced the active domain's entity names with placeholders. On 
multi-domain turns the non-active domain's names stayed lexicalized in the training response, so the model learned this noise as signal and emits real entity 
names at inference. 

---

### Booking success rate
Dialogue-level metric computed over booking attempts only. A booking attempt is a turn where `intent` is `book_hotel` or `book_restaurant`. 
A successful booking requires no violations AND `[ref]` placeholder present in `delex_response`. Booking rate = successful bookings / total booking attempts per dialogue.
Dialogues with no booking attempts contribute `None` and are excluded from the dataset average.

---

### Domain/intent accuracy
Domain and intent accuracy measure whether the pipeline's single predicted domain (or intent) matches the GT for that turn.
GT can have multiple active domains per turn (e.g., {"restaurant", "hotel"}). Our pipeline always predicts a single domain or None. Prediction is correct if it
matches ANY active GT domain. Precision, recall, and F1 are computed treating predicted as a set of size 1 and GT as a set of size 1 or 2.
On multi-domain turns: precision=1.0 if predicted is correct, recall=0.5 because we miss the second domain. This is a tradeoff of single-domain design.

---

### Observed failure patterns
The patterns below come from manual error analysis of the API-baseline runs (GPT-4o-mini, Claude 3 Haiku) on the dev set.

**Most common JGA failure: missing dontcare.**
User says "No particular type of food but moderate price" → model extracts pricerange=moderate but omits food=dontcare. 
CoT prompt and explicit dontcare examples partially address this. 
Remaining failures are cases where user combines preference + non-preference in one sentence and model focuses on explicit value and drops implicit dontcare.

**Multi-domain turn slot miss:**
GT has slots for both hotel and restaurant in same turn. Pipeline predicts single domain → misses the other domain's slots entirely.
JGA=False for that turn and all subsequent turns (cascading). Affects ~3-4% of turns. Accepted tradeoff of single-domain-per-turn design.

**Intent confusion find vs book:**
User says "book it" without explicit booking details → GT=book_hotel, pipeline predicts find_hotel. Booking slots not extracted.
Most common in implicit confirmation turns.

**Hallucination pattern:**
The pipeline mentions a real entity name in the response that was NOT returned by the DB for this turn. 
The entity exists in our knowledge (from LLM training or DB) but is factually wrong for the user's current constraints.

**Experiment 1 (~5% hallucination rate):** The single LLM sees the full DB in the system prompt AND generates the response. It sometimes ignores the DB results and uses memorized training knowledge instead.

Dummy example:
- User: "I need a cheap hotel in the east."
- DB query with {area=east, pricerange=cheap} → returns [] (no match)
- LLM response: "I found Home From Home at 124 Marathonos Road, cb12dp."
- LLM knows "Home From Home" from training data → generates real address
- "Home From Home" NOT in db_results → hallucination detected

**Experiment 2 (~2% hallucination rate):** ResponseGen LLM does NOT see the full DB. It only sees db_results from the current turn and is instructed to use placeholders only.

Same scenario:
- DB query → returns [] (no match)
- ResponseGen sees: "DB results: empty"
- ResponseGen prompted: use ONLY [hotel_name], [hotel_phone] etc.
- Response: "I'm sorry, I couldn't find a cheap hotel in the east. Would you like to try a different area?"
- No entity name mentioned → no hallucination

**Why the drop from 5% to 2%:**
Experiment 1 gives the LLM freedom to generate entity details from its own knowledge.
Experiment 2 forces ResponseGen to use only what the DB returned so hallucination becomes structurally much harder when the model outputs only placeholders.
Remaining 2% in Experiment 2 occurs when the LLM ignores placeholder instructions and outputs a real entity name directly.

---

## 7. Evaluation: Tomiinek

### How Tomiinek works
Source: [Tomiinek/MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation) (Nekvinda & Dušek, 2021, "Shades of BLEU, Flavours of Success")

Tomiinek is a standalone evaluator for the MultiWOZ context-to-response task. It computes Inform Rate, Success Rate, BLEU, and Combined Score.

**Input per turn:**
- `response`: delexicalized system response (e.g., "I found [restaurant_name] in the [restaurant_area]")
- `state` (optional): predicted belief state, nested by domain. If omitted, GT state from MultiWOZ 2.2 is used
- `active_domains` (optional): list of active domains. If omitted, inferred from state changes

**What we pass:**
- `state`: predicted informable (lexicalized) slots only (e.g., area, pricerange, food, name if mentioned), nested by domain.
  Requestables like phone, address, postcode are NOT in the state, they are tracked separately by Tomiinek from the GT goal
- `response`: delexicalized with placeholders (e.g., "I found [restaurant_name] in the [restaurant_area]").  Safer for BLEU and entity matching

---

### How Inform is computed
Inform is a **dialogue-level binary metric** (0 or 1 per dialogue), then averaged across the test set.

**Stage 1. Find the goal entity (uses the predicted state):**
For each domain in the dialogue, Tomiinek queries its internal MultiWOZ DB with our predicted constraints and gets back the set of matching entities (the "goal entities").

```
Example: 
Our state: {"restaurant": {"area": "north", "pricerange": "cheap"}} → DB query result: [nirala, royal_spice, ...]   ← "goal-entity set"
```

If the DB returns nothing (e.g., over-constrained or wrong values), the goal-entity set is **empty**.

**Stage 2. Check what we offered (uses the delex response):**
Tomiinek walks through **all turns** in the dialogue, accumulating any `[<domain>_name]` placeholders or literal entity names mentioned. For each domain, it keeps 
a single record: `venue_offered[domain]`.

```
Example:
Turn 2 (sys): "Sure, what food type?" → no placeholder yet
Turn 4 (sys): "I found [restaurant_name]." → venue_offered [restaurant] = True
Turn 6 (sys): "Phone is [restaurant_phone]." → already True
```

**Stage 3. Dialogue-level verdict:**
At the end of the dialogue, for each active domain:
- domain Inform = 1 if `venue_offered[domain]` is True AND the goal-entity set is non-empty
- domain Inform = 0 otherwise

Then the dialogue-level Inform combines all active domains with **AND**:
- dialogue Inform = 1 only if every active domain scored 1
- one domain failing → whole dialogue fails

**Stage 4. Dataset-level rate:**

```
Final Inform rate = (# dialogues with Inform = 1) / (total # dialogues) × 100
```

Number of turns per dialogue does not affect the math. A 6-turn and a 20-turn dialogue each contribute exactly one 0 or 1 to the final average.

---

### How Success is computed
Success is also a **dialogue-level binary metric**, computed the same way as Inform but with an extra requirement.

**Stage 1. Same as Inform:** Find the goal-entity set per domain via DB query.

**Stage 2. Two things to track per domain across all turns:**
1. `venue_offered[domain]`: was a `[<domain>_name]` placeholder ever mentioned? (same as Inform)
2. `provided_requestables[domain]`: for each attribute the user asked about (phone, address, postcode, ...), did the response ever 
   include the matching placeholder (e.g., `[restaurant_phone]`) or value?

The list of requested attributes per dialogue comes from **the GT goal in MultiWOZ 2.2**, not from our prediction. 
Tomiinek already knows what the user asked for.

**Stage 3. Dialogue-level verdict:**
For each active domain:
- domain Success = 1 if Inform = 1 for that domain AND every requested attribute was provided
- domain Success = 0 otherwise

Then dialogue Success combines all active domains with AND, just like Inform.

**Stage 4. Dataset-level rate:**

```
Final Success rate = (# dialogues with Success = 1) / (total # dialogues) × 100
```

Because Success requires Inform = 1, we always have **Success ≤ Inform**.
(Every dialogue that passes Success also passes Inform so Success is a stricter subset.)

---

**Dummy example:**
Setup for every case below:
```
Dialogue with 6 turns, single domain (restaurant)
User goal: find a cheap restaurant in the north
GT requested attributes: phone, address
```


| Case                                 | Predicted state                                                    | Response contents                                                 | Stage 1 DB result | venue_offered | requestables provided | Inform | Success |
|--------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------|-------------------|---------------|-----------------------|--------|---------|
| A: all good                          | `{area=north, pricerange=cheap}`                                   | `[restaurant_name]`, `[restaurant_phone]`, `[restaurant_address]` | non-empty         | True          | phone ✓, address ✓    | **1**  | **1**   |
| B: entity offered, attrs missing     | `{area=north, pricerange=cheap}`                                   | `[restaurant_name]` only                                          | non-empty         | True          | phone ✗, address ✗    | **1**  | **0**   |
| C: entity offered, one attr missing  | `{area=north, pricerange=cheap}`                                   | `[restaurant_name]`, `[restaurant_phone]`                         | non-empty         | True          | phone ✓, address ✗    | **1**  | **0**   |
| D: no entity offered                 | `{area=north, pricerange=cheap}`                                   | `[restaurant_phone]`, `[restaurant_address]` (no name)            | non-empty         | False         | phone ✓, address ✓    | **0**  | **0**   |
| E: empty DB result                   | `{area=north, pricerange=cheap, food=japanese}` (over-constrained) | `[restaurant_name]`, `[restaurant_phone]`, `[restaurant_address]` | **empty**         | True          | phone ✓, address ✓    | **0**  | **0**   |
| F: empty response                    | `{area=north, pricerange=cheap}`                                   | no placeholders at all                                            | non-empty         | False         | phone ✗, address ✗    | **0**  | **0**   |

Key takeaways:
- Inform = 1 requires BOTH a non-empty DB result AND a `[<domain>_name]` placeholder somewhere
- Success = 1 requires Inform = 1 AND every requested attribute provided
- Case D shows attributes alone don't save us. We must offer the entity itself
- Case E shows that even a perfect response can be killed by DST errors that empty the DB


---

### How BLEU is computed
- N-gram overlap between predicted delexicalized responses and GT delexicalized references
- Multiple reference sets available as we use `mwz22` (MultiWOZ 2.2 canonical references)

### Combined Score
```
Combined = 0.5 * (Inform + Success) + BLEU
```

---

**Critical implication for our pipeline:**
Because Stage 1 depends entirely on the predicted state, DST errors poison Inform/Success even when the response text is perfect. The dominant failure mode is wrong 
constraints producing an **empty DB result** → no valid goal entity exists → Inform = 0 regardless of how good the response is. This is why slot extraction accuracy is 
the root bottleneck for Inform/Success in our pipeline, not response generation quality.

### Why we use delexicalized responses
- Matches GT reference format → higher BLEU
- No hallucination risk (placeholders are always correct regardless of DB match)
- Same approach as all leaderboard models
- Tomiinek accepts both lexicalized and delexicalized via fuzzy matching, but delex is safer

### Slot format conversion
Tomiinek expects unprefixed keys nested by domain: `{"restaurant": {"area": "centre"}}`, not `{"restaurant-area": "centre"}`.
`build_tomiinek_turn()` converts our flat prefixed format to Tomiinek's nested format.

### Domain scope caveat
MultiWOZ has 7 raw domains (hotel, restaurant, attraction, train, taxi, hospital, police). The official Tomiinek leaderboard reports on **5**: attraction, hotel, 
restaurant, taxi, train. Hospital and police are excluded because they have negligible coverage in train and are absent from the test set.

Our Tomiinek "total" covers **2 of those 5** (hotel + restaurant), so direct comparison to leaderboard numbers requires caution.

---

## 8. Experiments

### Experiment 1: Single-LLM baseline
One LLM call per turn handles DST, entity selection, and response generation. 
Full hotel and restaurant databases passed in system prompt (~2000 extra tokens per turn).
LLM returns JSON: `{"domain", "intent", "slots", "response"}`, parsed with `json.loads()` with some processing.

Post-processing after LLM call (all rule-based, no additional LLM calls):
- `policy()` - check violations from parsed intent + slots
- `find_entity()` / `book_entity()` - DB lookup for lexicalization
- `supervisor()` - no retry, valid flag for metrics only
- `lexicalize()` - replace placeholders with real entity values
- `memory()` - store lex_response in history

Known limitations:
- No retry loop, single call, no self-correction
- JSON parsing failures handled with retry (max 3), control character stripping, and regex JSON extraction (needed for Haiku)
- Co-reference resolution ("same group", "book it") depends entirely on LLM reading history correctly
- Full DB in prompt feasible for API models only as small open-source models (3B) suffer context length truncation and quality degradation (see Section 9: Infrastructure)
- Entity name injection into accumulated_slots is commented out (see Section 5)

#### Context length problem with open-source models
Experiment 1 sends full dialogue history + full DB + all instructions in one prompt per turn.
By turn 3-4, prompts grow to 20,000+ tokens. With `LOCAL_MAX_SEQ_LENGTH=8192`, Unsloth truncates the prompt, cutting the system prompt which contains output format rules.
Result: model produces invalid JSON, slot extraction fails, metrics are unreliable.

This is a key finding motivating the modular architecture (Experiment 2/Experiment 3): each modular call is shorter and focused on one task (DST or ResponseGen). 
The modular approach is better suited for open-source models with limited context windows. 
Single-LLM baseline works reliably only with API models (GPT/Claude) that have 128K+ token context windows.


### Experiment 2 & 3: Modular pipeline
Two LLM calls per turn: DST and ResponseGen. Both Experiment 2 and Experiment 3 use `zeroshot=True` at inference with same pipeline, different models powering each module.

Pipeline steps per turn (`run_turn()`):
1. `dst()` → domain, intent, accumulated_slots
2. `policy()` → violations (missing required booking slots)
3. `response_generator()` → `supervisor()` → retry if invalid (max 2 attempts)
4. `lexicalize()` → replace placeholders with real entity values
5. `memory()` → store lex_response in history (NOT delex — prevents placeholder contamination)
6. Build `tomiinek_turn` (for official eval) and `custom_turn` (for custom metrics)

Key design decisions:
- `accumulated_slots` grows turn by turn, never reset within a dialogue
- Supervisor feedback passed to next retry attempt and enables self-correction
- Cost and response time accumulated across DST + all ResponseGen attempts per turn
- History stores lexicalized responses to avoid placeholder contamination in subsequent turns

`run_dialogue()` loops over USER turns only, passing `accumulated_slots` and `history` across turns. 
`run_experiment()` loops over all dialogues and returns two result dicts (`tomiinek_results`, `custom_results`) both keyed by dialogue_id.

#### Prompt consistency (Experiment 2 & Experiment 3)
Both zero-shot and fine-tuned models use identical input format at inference, generated by `build_dst_prompt()` and `build_respgen_prompt()`. 
Fine-tuning data preparation uses the same functions to generate training examples, guaranteeing train/inference prompt consistency.

### Evaluation pipeline (all experiments)
All three experiments produce the same two output dicts per turn and are evaluated identically:
- Custom metrics computed on **lexicalized** response (`lex_response`)
- Tomiinek metrics (Inform, Success, BLEU, Combined) computed on **delexicalized** response (`delex_response`)

### Intent confusion: find vs book
A recurring pattern across all experiments: GT intent is `book_hotel` but pipeline predicts `find_hotel`. 
User says "book it" or "yes please, for 8 people", GT considers this a booking intent, 
but the pipeline (especially Experiment 2 DST) sometimes interprets it as still searching. 
This cascades: wrong intent → booking slots not extracted → JGA failure. 
Most common in turns where the user implicitly confirms without using the word "book".

### Policy violation rate: Experiment 1 vs Experiment 2 tradeoff
Policy violation rate is consistently higher in Exp2 than Exp1.

Root cause: 
In Experiment 1, the single LLM sees the full conversation, full DB, AND booking policy rules in the prompt so it rarely predicts `book_*` intent unless confident all required slots are present. 
In Experiment 2, the DST model only sees the conversation and valid intents so it sometimes predicts `book_hotel` or `book_restaurant` too early before all required slots are provided.

Example: User says "I want to book a hotel."
- Experiment 1: sees booking rules → predicts `find_hotel` → no violation
- Experiment 2 DST: sees "book" → predicts `book_hotel` → policy catches missing slots → violation

The pipeline handles violations correctly as it asks the user for missing slots instead of executing a bad booking. 
The violation is an internal metric artifact, not a user-facing failure.

Impact: this is a real architectural tradeoff. The modular pipeline trades a slightly higher policy violation rate for better DST accuracy and overall task success.
Future work: adding booking policy rules to the DST prompt could reduce Experiment 2 violations.

---

## 9. Infrastructure

### What is LoRA and QLoRA
LoRA (Low-Rank Adaptation) freezes all base model weights and adds two small trainable matrices (A and B) to each attention layer. 
Instead of updating 9.4M parameters in a weight matrix, LoRA updates only 98K parameters (rank=16) that get multiplied together and added to the frozen weights at inference time.

QLoRA goes further: it quantizes the frozen base weights from 16-bit to 4-bit, reducing memory by ~4x while keeping the LoRA adapters in 16-bit for training precision.

Example for one attention layer (`q_proj`, shape 3072×3072):

|               | Base weights              | LoRA A          | LoRA B          | Total  |
|---------------|---------------------------|-----------------|-----------------|--------|
| LoRA (16-bit) | 9.4M × 2 bytes = 18.8 MB  | 3072×16 = 98 KB | 16×3072 = 98 KB | ~19 MB |
| QLoRA (4-bit) | 9.4M × 0.5 bytes = 4.7 MB | 98 KB (16-bit)  | 98 KB (16-bit)  | ~5 MB  |

Only LoRA A and B update during training. The base weights are frozen.

### Precision: 16-bit vs 4-bit
16-bit (float16): each number uses 16 bits → 65,000 possible values across the range. Values are packed closely together, so rounding error is tiny. 
Example: 0.3271 → nearest of 65,000 values → stored as 0.32714 (error: 0.00004)

4-bit (NF4): each number uses 4 bits → only 16 possible values across the range. Values are spread far apart, so rounding error is larger.
Example: 0.3271 → nearest of 16 values → stored as 0.3125 (error: 0.0146)

QLoRA accepts this precision loss in base weights because the LoRA adapters (16-bit) learn to correct for the quantization error during training.

### Why training cannot run locally
- GPU architecture: Unsloth uses Triton kernels requiring sm_70+ (Volta, 2017+). Local GPU is GTX 1050 Ti (Pascal, sm_61). No software fix.
- VRAM: a 3B model in 4-bit needs ~6 GB minimum. GTX 1050 Ti has 4 GB. No software fix.

---

## 10. Aggregation Notes

### Averaging strategy across evaluation levels
- **Turn level:** raw metrics (JGA=True/False, Slot F1=float)
- **Dialogue level:** macro average over turns within dialogue (stored in dialogues.json, internal use only)
- **Dataset level:** micro average over ALL turns directly (standard in MultiWOZ literature, used for reported results)
- **Per-domain:** micro average over domain-specific turns only (comparable to overall dataset metrics)
- **Booking rate:** dialogue-level fraction, not micro-averaged
- **Tomiinek metrics:** merged into dataset_metrics after `evaluate_experiment()` returns, from `run_tomiinek()`

### Latency vs cost: aggregation difference
Cost = TOTAL (sum of all turns). Per-domain costs add up to overall cost.
Latency = AVERAGE (mean per turn). Per-domain latencies do NOT add up to overall.

Example (1 dialogue, haiku, 5 turns: 2 hotel + 3 restaurant):
- Hotel avg latency: 9.67s, Restaurant avg latency: 3.09s
- Overall avg latency: (2 × 9.67 + 3 × 3.09) / 5 = 5.72s (weighted average, not sum)
- Hotel cost: 0.0119, Restaurant cost: 0.0176
- Overall cost: 0.0119 + 0.0176 = 0.0296, cost in dollars

---
