# MLP4CS Development Notes

Implementation decisions, known limitations, and observations.

---

## 1. Dataset

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

## 2. DST

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

### Prompt: dontcare extraction rule
Added explicit rule to `build_dst_prompt()`: if user expresses no preference (e.g., "any food", "doesn't matter") → extract that slot with value `dontcare`.
Without this rule, LLMs omit the slot entirely. GT always has `slot=dontcare` in these cases so missing it causes JGA=False and lower Slot Recall.

Impact: improved JGA and Slot F1 in Exp2. Consistent across Experiment 2 and Experiment 3 as the same rule is used in fine-tuning data via `build_dst_prompt()`.
See Section 5 for observed failure cases in error analysis.

---

## 3. Response Generation

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


### Delexicalization on cross-domain turns
Delexicalization is applied per active domain only. 
On multi-domain turns where the response mentions a secondary domain entity, that entity remains lexicalized.

Example (active domain = restaurant): "[restaurant_name]'s phone is [restaurant_phone]. As far as hotels go,  I recommend the University Arms Hotel in the center of town."
"University Arms Hotel" is NOT delexicalized because hotel is not the active domain.

Impact: on Tomiinek is none, as Tomiinek evaluates only the active domain per turn. Regarding BLEU, minor penalty on multi-domain turns where GT references are fully delexicalized. 
Affects a minority of turns as it is an accepted tradeoff for simplicity.

### Lexicalization: [ref] placeholder
`lexicalize()` accepts `ref=""` by default. The pipeline never passes a real booking reference as users see `[ref]` in lexicalized responses. 
For evaluation this works as `[ref]` presence signals booking confirmation. 

---

## 4. Pipeline Flow

### Research question
Does architectural decomposition into specialized LLMs improve reliability over a single-LLM baseline, and does role-specific fine-tuning yield further improvement?
Additionally: can small fine-tuned open-source models compete with larger commercial API models (GPT-4o-mini, Claude 3 Haiku)?

### What we compare
**Experiment 1: Single LLM (everything in one call):**
```
User Turn → [Single LLM: DST + Policy + Entity Selection + Response] → supervisor() → lexicalize() → Evaluate
```
One LLM handles all tasks. Full DB in prompt. No retry.
Tested with: API models (GPT-4o-mini, Claude 3 Haiku) and open-source models (8B, 12B).
Establishes the baseline: how well can a single model do without any architectural help?

**Experiment 2: Modular pipeline (two specialized LLMs):**
```
User Turn → [DST LLM] → policy() + Entity Selection → [ResponseGen LLM] → supervisor() → lexicalize() → Evaluate
```
Two separate LLM calls, each focused on one task. DB queried between them.
Tested with: homogeneous (same model for both) and heterogeneous (different models) configurations, using both API models and open-source models (3B, 8B, 12B).
Tests: does splitting responsibilities improve accuracy over Experiment 1?

**Experiment 3: Fine-tuned modular pipeline (LoRA adapters):**
```
User Turn → [DST LoRA] → policy() + Entity Selection → [ResponseGen LoRA] → supervisor() → lexicalize() → Evaluate
```
Same pipeline as Experiment 2, but DST and ResponseGen powered by LoRA fine-tuned open-source models.
Tested with: several fine-tuned configurations.

Tests the questions:
1. Do small open-source fine-tuned models outperform their zero-shot versions from Experiment 2?
2. Can small open-source fine-tuned models approach the performance of commercial API models from Experiment 1/Experiment 2?

### Shared components across Exp2 and Exp3
Between and around the two LLM calls, the pipeline runs rule-based components:
- `policy()` — checks if required booking slots are present before allowing a booking
- `supervisor()` — validates the response, triggers retry if invalid (max 2 attempts)
- `lexicalize()` — replaces placeholders with real entity values from DB results
- `memory()` — stores the final lexicalized response in history for subsequent turns

These components are identical across Experiment 2 and Experiment 3, only the LLM modules change.

---

## 5. Evaluation: Custom Metrics

### Metric computation: what is counted vs skipped per turn
The `skip` flag is set when BOTH predicted domain and intent are None (multi-domain or farewell turns). This affects metrics differently:

**Always computed (all turns):**
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


### JGA 

#### Computed on accumulated belief state
JGA uses real slot values (e.g., `{"restaurant-name": "golden wok"}`), not placeholders. JGA = True only if ALL slot-value pairs match exactly across ALL domains.
MultiWOZ 2.2 already stores accumulated belief state per frame per domain. Our evaluator merges across domain frames into a single flat dict via `gt_accumulated.update()` to enable cross-domain JGA comparison against the pipeline's own flat accumulated slots.


#### Challenge: entity name mismatch
Pipeline always uses `db_results[0]`, the first matching entity. GT annotator may have chosen a different entity from the same result set.
Example: user wants north + moderate restaurant → DB returns [nirala, golden wok]. Pipeline predicts `restaurant-name=nirala`, GT has `restaurant-name=golden wok`.
JGA=False despite correct constraints. This affects ALL published MultiWOZ systems and is not a pipeline bug, but a dataset limitation. 
No mitigation possible as the pipeline cannot know which entity the GT annotator chose from the matching results.

#### Challenge: missing entity name in predicted slots
Pipeline does not inject recommended entity name into `accumulated_slots`. The DST prompt instructs the model to extract only values explicitly stated by the user. 
Whether the LLM also picks up entity names from system recommendations in conversation history is model-dependent and inconsistent.
GT always has the name once the system recommends it, causing JGA=False for subsequent turns even when all other slots match perfectly.

We tested injecting `db_results[0]["name"]` but it hurt JGA because our DB returns entities in a different order than the GT annotator chose. 
Injecting the wrong name cascades as a mismatch through all subsequent turns. Without injection, the name is simply absent (still JGA=False) but at least no wrong name propagates.

Leaderboard models avoid this because they train end-to-end on GT belief states and learn to predict the exact GT entity name. 
Zero-shot pipelines cannot replicate this. For Experiment 3 (fine-tuned), this limitation may be reduced as the model trains on GT belief states that include entity names and learns to extract them from history naturally.

Tradeoff: cleaner belief state, slightly lower JGA.

#### Challenge: cascading slot errors
A single wrong slot in an early turn propagates to all subsequent turns because `accumulated_slots` grows turn by turn and never resets.

Example:
Turn 2: DST extracts `restaurant-area=cambridge` (wrong — city name, not area). Turn 3-4: `accumulated_slots` still has `restaurant-area=cambridge` and GT Turn 3-4: `restaurant-area=centre`.
JGA=False for turns 3 AND 4 as caused by single Turn 2 error.

This is by design as accumulated belief state mirrors how a real dialogue system tracks constraints across turns. 

#### Slot F1 as partial credit
Computed on the same accumulated flat prefixed belief state as JGA. Unlike JGA, gives partial credit for correct slots. 
`dontcare` is included in both predicted and GT sides, consistent with GT annotations and avoids artificially low recall.

#### dontcare vs none distinction
Some leaderboard models (e.g., SimpleTOD) do not distinguish between `dontcare` (user explicitly expressed no preference) and `none` (slot never mentioned).
This inflates their JGA because they get credit for unextracted slots.
The official MultiWOZ repository acknowledges this issue and refers to CheckDST (https://github.com/wise-east/checkdst) for corrected evaluation.

Our pipeline correctly distinguishes `dontcare` from `none`. Our JGA numbers are lower but methodologically correct. 

Source: https://github.com/budzianowski/multiwoz (DST leaderboard footnote)


### Policy compliance check
Evaluated per USER turn but checks the system's response to that turn. Policy compliance is NOT simply `len(violations) == 0`. 
A turn with violations is still compliant if the pipeline correctly asked the user for missing slots instead of confirming a booking. 
Non-compliant only when the response contains booking confirmation signal of `[ref]`, despite missing required slots.
See Section 7 for Experiment 1 vs Experiment 2 policy violation rate comparison.


### Action accuracy with rule-based mapping
Predicted action is derived from intent + violations (not from LLM output):
- `find_*` intent → "inform"
- `book_*` intent + violations → "request" (asking for missing slots)
- `book_*` intent + no violations → "book"

GT action is derived from SYSTEM turn `dialog_act` keys. Known issue: `Booking-Inform` maps to "inform" not "book" and only `Booking-Book` confirms an actual booking. 
Turns where either action is undetermined (None) are excluded.


### Hallucination

#### Detection method
All known entity names are loaded from both hotel and restaurant DBs.
Any known name appearing in `lex_response` but NOT in this turn's `db_results` is counted as a hallucination. 
Turns where no entity is mentioned in the response are excluded from the hallucination average (not penalized, not rewarded).

#### Sub-cases across experiments
**True hallucination (primarily Experiment 1):**
LLM generates entity details from training knowledge instead of DB results.
Example: "Cotto is at Regent Street". LLM knows Cotto (from training) but DB returns a different entity. Entity NOT in `db_results` → true hallucination.
Impact: even with full DB in prompt, single-call LLMs hallucinate and validates the modular architecture's value.

**Placeholder compliance failure (primarily Experiment 1, occasionally Experiment 2):**
LLM uses the correct entity but with real values instead of placeholders. Our metric flags this as hallucination because the real name matches against the full entity DB.
Example: "Home From Home is at 78-80 Milton Road...". Entity IS in `db_results` → not a true hallucination, but flagged because real name appears in response.
In Experiment 2/Experiment 3, placeholder instructions reduce this, but LLMs occasionally ignore them. Addressed by strengthening rules in `build_respgen_prompt()`.

**Wrong entity from DB lookup (all experiments):**
`find_entity()` returns wrong entity when user provides a name with slight variations.
Example: user says "Home From Home" → DB returns "a and b guest house" because exact
match fails. Fixed with fuzzy containment matching.

### Booking rate
Dialogue-level metric computed over booking attempts only. A booking attempt is a turn where `intent` is `book_hotel` or `book_restaurant`. 
A successful booking requires no violations AND `[ref]` placeholder present in `delex_response`. Booking rate = successful bookings / total booking attempts per dialogue.
Dialogues with no booking attempts return `None` and are excluded from the dataset average.


### Future work: request fulfillment
Not implemented. Would measure whether the system provided information the user explicitly requested (e.g., user asks for phone → did response include phone?).


### Domain/intent accuracy: GT set asymmetry
GT can have multiple active domains per turn (e.g., {"restaurant", "hotel"}).
Our pipeline always predicts a single domain or None. Prediction is correct if it
matches ANY active GT domain. Precision, recall, and F1 are computed treating
predicted as a set of size 1 and GT as a set of size 1 or 2.
On multi-domain turns: precision=1.0 if predicted is correct, recall=0.5 because
we miss the second domain. This is a tradeoff of single-domain design.


### Key failure patterns from error analysis (API baselines, dev set)

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

## 6. Evaluation: Tomiinek

### How Tomiinek works
Source: [Tomiinek/MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation) (Nekvinda & Dušek, 2021, "Shades of BLEU, Flavours of Success")

Tomiinek is a standalone evaluator for the MultiWOZ context-to-response task. It computes Inform Rate, Success Rate, BLEU, and Combined Score.

**Input per turn:**
- `response`: delexicalized system response (e.g., "I found [restaurant_name] in the [restaurant_area]")
- `state` (optional): predicted belief state, nested by domain. If omitted, GT state from MultiWOZ 2.2 is used
- `active_domains` (optional): list of active domains. If omitted, inferred from state changes

**What we pass:**
- `state`: real slot values (lexicalized), e.g., `{"restaurant": {"area": "north", "name": "nirala"}}`. Tomiinek needs real values to query its internal DB as placeholders would fail the lookup.
- `response`: delexicalized with placeholders, e.g., "I found [restaurant_name] in the [restaurant_area]". Safer for BLEU and entity matching.

**How Inform is computed:**
1. Takes the predicted belief state (or GT if not provided)
2. Queries its own internal copy of the MultiWOZ DB with those slots
3. Finds entities matching the constraints (the "goal entities")
4. Checks if our response mentions the correct entity by searching for entity name or its placeholder (e.g., "Bedouin" or "[restaurant_name]")
5. A dialogue is Informed if the system mentioned a matching entity for each domain

**How Success is computed:**
1. Same DB query as Inform to find the goal entity
2. Additionally checks if the response provides all **requested attributes** (e.g., if user asked for phone → did response include phone or [restaurant_phone]?)
3. A dialogue is Successful if Informed AND all requested attributes were provided

**How BLEU is computed:**
- N-gram overlap between predicted delexicalized responses and GT delexicalized references
- Multiple reference sets available as we use `mwz22` (MultiWOZ 2.2 canonical references)

**Combined Score:**
- `Combined = 0.5 * (Inform + Success) + BLEU`

**Dummy example:**
```
User goal: find a cheap restaurant in the north
GT requested attributes: phone, address

System response: "I found [restaurant_name] in the north. Phone: [restaurant_phone], address: [restaurant_address]"
Predicted state: {"restaurant": {"area": "north", "pricerange": "cheap"}}

Tomiinek DB query with {area=north, pricerange=cheap} → returns [nirala, golden wok, ...]
Response contains [restaurant_name] → matches an entity → Inform = True
Response contains [restaurant_phone] + [restaurant_address] → all requested attrs → Success = True
```

**Critical implication for our pipeline:**
Tomiinek uses the **predicted state** (not the response text) to query the DB internally.
If our DST extracts wrong slots, Tomiinek queries the DB with wrong constraints and finds wrong entities. 
Even if the response is perfect, Inform fails because the expected entity doesn't match. 
This means **slot extraction accuracy is the root bottleneck for Inform/Success, not response generation quality**.

Booking slots require additional remapping via TOMIINEK_SLOT_MAP in config.py:
bookday → booking-day, bookpeople → booking-people, bookstay → booking-stay, booktime → booking-time.
Without this mapping, Tomiinek silently ignores booking slots, artificially deflating Success% scores.

### Why we use delexicalized responses
- Matches GT reference format → higher BLEU
- No hallucination risk (placeholders are always correct regardless of DB match)
- Same approach as all leaderboard models
- Tomiinek accepts both lexicalized and delexicalized via fuzzy matching, but delex is safer

### Slot format conversion
Tomiinek expects unprefixed keys nested by domain:
`{"restaurant": {"area": "centre"}}`, not `{"restaurant-area": "centre"}`.
`build_tomiinek_turn()` converts our flat prefixed format to Tomiinek's nested format.

### Domain scope caveat
Our Tomiinek "total" covers hotel + restaurant only (2 of 5 leaderboard domains).
The official leaderboard aggregates across attraction, hotel, restaurant, taxi, and train.
Direct comparison requires caution. 

---

## 7. Experiments

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

## 8. Infrastructure

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

## 9. Aggregation Notes

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
- Hotel cost: $0.0119, Restaurant cost: $0.0176
- Overall cost: $0.0119 + $0.0176 = $0.0296 (sum)

---
