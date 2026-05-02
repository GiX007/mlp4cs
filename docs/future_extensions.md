# Future Extensions & Edge Cases

## 1. Multi-user dialogue
**Problem:** Real customers book in groups (couples, families). Single-user pipeline breaks on cross-talk and deliberation

**Flow:**
```
Multi-user chat → [Rewriter LLM] → Single-user query → [DST] → ...(rest of pipeline unchanged)
```

**Reference:** Jo et al. 2023, "Multi-User MultiWOZ" arXiv: 2310.20479 (Figure 1 shows the rewrite mechanism)

**What's needed:** 1 new front-end module, existing pipeline untouched

---

## 2. ReAct-style iterative DST
**Problem:** Current pipeline is single-shot per turn. The DB query is fixed by code, so the system can't recover from zero results or self-correct when its first guess is wrong.

**Flow:**
```
User → [DST loop: Thought → Action → Observation → ... → final slots] → ... (rest of pipeline unchanged)
```

**Reference:** Yao et al. 2022, "ReAct" arXiv: 2210.03629

**Example:**
- User: "I want a cheap Italian restaurant in the north."
- Current pipeline (single-shot):
  - DST extracts: pricerange=cheap, food=Italian, area=north
  - DB query → 0 results
  - ResponseGen: "Sorry, no matches found."
  - User leaves unsatisfied or frustrated
- ReAct iterative DST:
  - Thought: User wants cheap Italian in north
  - Action: search_db(area=north, pricerange=cheap, food=Italian)
  - Observation: 0 results
  - Thought: Zero results → relax area first, it's the most specific constraint
  - Action: search_db(pricerange=cheap, food=Italian)
  - Observation: 2 results found → one in center, one in east
  - Thought: Also check if north has any cheap restaurants of any cuisine
  - Action: search_db(area=north, pricerange=cheap)
  - Observation: 3 results → all non-Italian
  - Thought: Two angles to offer. Stop searching
  - Final slots passed to → ResponseGen
- ResponseGen: "I couldn't find a cheap Italian restaurant in the north, but I found two cheap Italian places in center and east, 
or three other cheap restaurants in the north if you're flexible on cuisine. Which would you prefer?"

**What's needed:** Replace single-shot DST + fixed DB query with a ReAct loop. 
Define a tool registry (search_db, relax_constraint, verify_availability, ask_user). 
Add hard step limit to avoid endless loops. New evaluation strategy needed (per-loop or end-to-end)

**Wins:** Better recovery on zero-result queries, possible replacement of rule-based supervisor

**Costs:** ~3x latency and tokens per turn, harder to debug and evaluate, more failure modes (infinite loops, hallucinated DB results)

**Model size guidance:** Use prompted ReAct for large models (GPT-4, Claude, 70B+). 
For smaller open-source models (≤8B), fine-tune tool calling into the model instead

---

## 3. Role-aware user clarification
**Problem:** Current pipeline assumes user input is complete and unambiguous. When the user is vague ("find me something nice"),
the DST guesses or extracts incomplete slots, leading to wrong results or wasted turns. 
There is no dedicated mechanism to proactively ask the user to clarify before acting

**Flow:** 
```
User → [Manager agent checks high-level clarity → if unclear: ask user] → [DST/Expert agents extracts slots → if unclear: ask user]
     → ... (rest of pipeline unchanged)
```
Add clarification at two levels:
- High-level (router/manager agent): resolve ambiguity about domain, intent, or vague constraints (e.g., "do you mean a hotel or guesthouse?", "for how many people?")
- Low-level (per-module expert agents): resolve slot-level ambiguity during DST or response generation (e.g., "which area? we have results in center and north")

**Reference:** Acikgoz et al. 2026, "MAC: A Multi-Agent Framework for Interactive User Clarification in Multi-turn Conversations" arXiv: 2512.13154

**What's needed:** Add a clarification taxonomy (what to ask at which level). Add a "should I clarify?" check before DST and before ResponseGen. 
Define a max-clarification budget per turn to avoid annoying the user

**Wins:** Higher task success (+7.8% in MAC paper, on MultiWOZ 2.4), fewer total dialogue turns by collecting all required info up front, fewer repeated/wrong responses

**Costs:** More LLM calls per turn, risk of over-clarifying (annoys users), needs careful budget tuning

---

## 4. Reflexion-style supervisor with self-reflection memory
**Problem:** Current supervisor returns short rule-based feedback ("missing [ref]", "use placeholder X") and ResponseGen retries with that feedback. 
The feedback is fixed by handwritten rules and exists only for one retry. Failures don't persist across turns in a dialogue

**Flow:**
```... (rest of pipeline unchanged) →
[ResponseGen (generate) → Supervisor
  (Option A: rule-based check)
  (Option B: LLM check that also explains the failure)
  → if invalid: ResponseGen (reflect on why) → store reflection
  → ResponseGen (retry with reflection in prompt)
  → loop until valid or step limit]
→ Lexicalizer → Memory
```
Replaces the existing supervisor + retry block. The same ResponseGen LLM is used in two modes: generation and reflection. 
Reflections persist in an episodic memory across turns of the same dialogue.

**Reference:** Shinn et al. 2023, "Reflexion" arXiv: 2303.11366

**What the paper does:**
- Step 1. ACTOR (LLM):
  - Input: "What is the capital of the country where the Eiffel Tower is?"
  - Output: "London"
- Step 2. EVALUATOR (rule-based, exact match):
  - Compares "London" vs ground truth "Paris"
  - Returns: WRONG (just a binary signal, no explanation)
- Step 3. SELF-REFLECTION (LLM, same Actor model with different prompt):
  - Prompted: "You answered 'London'. The answer was wrong. Reflect on why and how to do better next time."
  - Output: "I confused the location of the Eiffel Tower. It's in France, not England. Next time I should verify the country first before naming the capital."
- Step 4. RETRY:
  - Actor LLM is called again with the reflection added to prompt. This time it answers "Paris"

**Example:**
- User: "Book me a hotel in the center for 3 nights."
- ResponseGen (generate): "Your booking is confirmed!" (no [ref] placeholder, hallucinates a confirmation)
- Supervisor (rule-based check): INVALID as booking signaled but [ref] missing
- ResponseGen (reflect on why): "I assumed the booking went through, but the system never returned a reference. 
  I should only confirm bookings when [ref] is provided in the DB results." → reflection stored in episodic memory
- ResponseGen (retry with reflection in prompt): "I'd be happy to book that hotel. Could you confirm the day and number of guests so I can proceed?"

Later in the same dialogue:
- User: "Now also book me a restaurant for tonight."
- ResponseGen (generate, reflection from memory still in context): correctly asks for missing slots before confirming, avoiding the same mistake


**What's needed:** Replace handwritten rule feedback with LLM-generated reflections (no new LLM but same ResponseGen with different prompt). 
Add an episodic memory buffer that carries reflections across turns. Keep the hard step limit on retries to avoid infinite loops

**Wins:** No more maintaining rule code for new failure types, model carries lessons across turns within a session, no training needed

**Costs:** Extra LLM call per failure, only helps within one session (no learning carries across sessions as reflections are discarded at session end), 
reflections can be shallow or hallucinated

**Separate LLM judge as alternative:** Instead of reusing ResponseGen for reflection, a separate "judge" LLM could be added to evaluate and explain failures. 
Used in industry for evaluation, but adds cost (2 LLMs per turn), latency, and new risks: hallucinations. 
Most useful when domain-specific failure modes need expert judgment that rules can't capture, and a small cheap judge model can handle the cost

---

## 5. Schema-grounded prompting for individual modules
**Problem:** Current pipeline uses generic prompts that tell the LLM what to do in plain English, with no formal injection of business rules or expected behavior. 
The LLM is trusted to remember constraints turn after turn (e.g., "always confirm before booking", "never recommend without availability"). 
This works in prototypes but fails in production when business rules grow

**Flow:** 
No architectural change, just enrich each LLM module's prompt with explicit schema instructions: slot vocabulary + valid values for the DST module, 
and a handwritten decision flowchart for the ResponseGen / policy module. The same LLM is now schema-grounded inside each step

**Reference:** Zhang et al. 2023, "SGP-TOD: Building Task Bots Effortlessly via Schema-Guided LLM Prompting" arXiv: 2305.09067

**What's needed:** Write the slot vocabulary + valid values prompt block for DST. Write the policy flowchart prompt block for ResponseGen. Keep them in version control as plain text

**Wins:** More predictable bot behavior, business rules become text-editable, easier to audit and debug

**Costs:** Larger prompts (more tokens per call), the schema becomes a maintenance burden as the business grows

**Combination with Section 6:** This idea and the workflow-graph idea (Section 6) are complementary, not competing. A real product typically uses both at different layers: 
a workflow graph at the system level defining the flows, transitions, and guardrails (book / cancel / modify / escalate), 
with schema-grounded prompting injected into the LLM call inside each node to keep the bot's behavior at that step predictable. 
The graph enforces *which* flow runs and the schema enforces *what* the LLM does inside each step of that flow

**Dialog-act planning variant:** DiactTOD (Wu et al. 2023, current Combined SOTA on MWOZ 2.2 at 104.4) trains a single model to first generate a dialog act token (request, inform, confirm, 
recommend, book) and then continue with the response in the same autoregressive sequence. The same idea adapts naturally to MLP4CS: split this into a small 
"planner" step before ResponseGen that picks one dialog act per turn, and inject that act as a plain-text line in the ResponseGen prompt (e.g., "Dialog act: recommend")

---

## 6. Workflow graph orchestration
**Problem:** Current pipeline is a fixed linear sequence (DST → policy → ResponseGen → ...). 
Real customer service has many branching paths: cancellation, modification, complaint, escalation to human, retry on different domain. 
A linear pipeline can't represent these without messy if/else trees

**Flow:** Replace the linear runner with an explicit workflow graph where nodes = states (greeting, search, book, cancel, escalate), 
edges = transitions (driven by intent/conditions), and LLM modules (DST, ResponseGen) are called inside specific nodes. 
The graph enforces business rules and the LLM keeps natural-language flexibility within each node

**Earlier related idea:** Zhang et al. 2023, "SGP-TOD: Building Task Bots Effortlessly via Schema-Guided LLM Prompting" (arXiv: 2305.09067). 
The prototype version of the workflow-graph idea, where a handwritten task flowchart is injected into the LLM prompt at every turn. 
But, this approach generalizes this to a stateful graph executor suitable for production

**Reference:** Park et al. 2025, "A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs" arXiv: 2505.23006

**Example:**
- User: "Hi, I want to cancel my booking from last week."
- Top-level graph router (LLM or rules):
  - Detects intent = "cancel"
  - Routes to node: CANCEL_FLOW
- Inside CANCEL_FLOW node, sequential pipeline runs:
  -  DST → extracts booking_ref="ABC123"
  - Policy → checks if cancellation is allowed for this ref
  - DB query → fetches booking + cancellation rules
  - ResponseGen → "Your booking ABC123 is eligible for free cancellation. Confirm?"
- User changes intent mid-conversation:
  - User: "Actually, can I just change the date instead?"
- Graph router:
  - Detects intent change = "modify"
  - Routes to node: MODIFY_FLOW (different sequential pipeline)
- The graph handles branching between flows (cancel / modify / book / escalate) and the pipeline handles execution within a flow

**What's needed:** Create a graph executor (e.g., LangGraph). Define nodes for each business state. 
Define transition conditions. Keep DST/ResponseGen as node implementations

**Wins:** Branching support, easier to add new flows (cancel, modify), business rules guaranteed by the graph structure

**Costs:** Graph design upfront, less flexible than free-form chaining, requires defining all valid paths in advance

**Building the workflow from real data:** 
- For an existing business with past conversation logs (e.g., a restaurant chain with 10,000 past customer-agent chats), the workflow graph itself can be mined from those logs instead of designed by hand. 
  These logs can be fed to an offline extraction pipeline that outputs a draft graph (common intents, transitions, slots, exception paths), which is then reviewed and polished before deployment
- **Reference:** Choubey et al. 2025, "Turning Conversations into Workflows" arXiv: 2502.17321 (Salesforce QA-CoT prompting for workflow extraction)

**Evaluating policy compliance:** Once a workflow graph is in place, compliance with business rules can be measured directly. 
Balaji et al. 2026 propose JourneyBench (arXiv: 2601.00596) and the User Journey Coverage Score for this purpose, showing that structured orchestration
lets smaller models (GPT-4o-mini) outperform larger ones (GPT-4o) on policy-driven customer support tasks

---

## 7. Tool calling: schema-validated outputs for DST and DB lookup
**Problem:** Current pipeline asks the LLM to write a JSON string for the belief state ("output slot-value pairs in this format"). 
The LLM sometimes returns malformed JSON, extra text, or hallucinated keys, requiring retries. This is fragile in production

**Flow:**
```
User → [DST: LLM calls function_call(area=..., stars=..., ...)] → arguments parsed natively by the API client → DB query → [ResponseGen] → ... (rest of pipeline unchanged)
```
**Reference:** Li et al. 2024, "Large Language Models as Zero-shot Dialogue State Tracker through Function Calling" arXiv: 2402.10466

**Example:**
- User: "I want a 4-star hotel in the north."
- Current pipeline (JSON prompting):
  - LLM returns: `Sure! Here's the JSON: {"hotel-area": "north", "hotel-stars": 4}` ← extra text breaks parser
  - Or: `{"hotel-area": "north", "hotel-stars": "four"}` 
- Function-calling pipeline:
  - LLM is given: `book_hotel(area: str, stars: int, price: str, ...)` as a function definition
  - LLM emits: `function_call: book_hotel(area="north", stars=4)` ← guaranteed valid arguments
  - The OpenAI/Anthropic Python client returns the arguments as a typed dict directly, no parsing code needed

**What's needed:** Define each MultiWOZ domain schema as a function (book_hotel, find_restaurant, etc.). 
Switch DST prompts to function-calling format. Use a function-calling-capable LLM (GPT-4o-mini, Claude, Qwen-Instruct, Llama-3-Instruct all support it). 
The DB query can also become a tool the LLM directly calls instead of fixed code

**Wins:** Schema-validated outputs (no JSON parsing errors), unifies DST + DB lookup + tool actions in one mechanism, 
production-grade reliability, supports streaming structured output

**Costs:** Some smaller open-source models support function calling poorly (need fine-tuning to make it reliable), locks the implementation to function-calling-capable APIs

**Lighter alternative:** Xu et al. 2024, "AutoTOD" arXiv: 2407.13957

Same goal as native function calling, but implemented as plain-text tool descriptions in the prompt + a regex-based parser to detect when the LLM "calls" a tool. 
Useful when the chosen LLM does not support native function calling, or when we want one prompt format that works across all LLM providers

**Example:**
- Instead of registering `book_hotel(area, stars, price)` through an API, the prompt contains a plain-text description:
  ```
  Available tools:
   - query_hotel(area, stars, price) — returns matching hotels
   - book_hotel(name, day, people) — books a hotel and returns a reference number
    
  Use this format when calling a tool:
    Thought: <why the tool is needed>
    Tool: <tool name>
    Args: {"area": ..., "stars": ...}
  ```
- LLM produces:
  ```
  Thought: I need to find 4-star hotels in the north before booking.
  Tool: query_hotel
  Args: {"area": "north", "stars": 4}
  ```
- A regex pattern (`Tool:\s*(\w+)\s*Args:\s*(\{.*\})`) extracts the tool name and JSON args, runs the actual database call, and feeds the result back into 
  the conversation. No API client library needed as it works with any chat completion endpoint

---

## 8. Few-shot retrieval-augmented prompts
**Problem:** Current pipeline gives DST and ResponseGen the same fixed instructions for every dialogue, regardless of how similar the user's request is to seen training examples.
The model has no concrete examples of what good slot extraction or good response generation looks like for the kind of input it just received. Per-domain accuracy stays roughly the same whether the user input is typical or unusual

**Flow:**
```
┌──── (preprocessing: encode training set into a vector index) ────┐
                                                                                  
User input → [Retriever finds 2-5 most similar training dialogues] → injected  [DST prompt] → [DB lookup] → [Retriever again for ResponseGen prompt] → ... (rest of pipeline unchanged)
```

**Reference:** Hudecek & Dušek 2023, "Are LLMs All You Need for TOD?" SIGDIAL 2023, arXiv: 2304.06556 (Figure 2 shows the retrieval mechanism alongside the zero-shot pipeline)

**Example:**
- User: "I want a 5-star hotel in the north"
- Preprocessing (done once): encode all training dialogues into a vector index
- Runtime: retriever finds 3 most similar past dialogues, e.g.:
  - "I'd like a 4-star hotel in the center" → belief state: hotel-stars=4, hotel-area=center
  - "Find me a 5-star place in the south" → belief state: hotel-stars=5, hotel-area=south
  - "any 3-star option in the east?" → belief state: hotel-stars=3, hotel-area=east
- These 3 examples + the new user input are bundled into the DST prompt
- DST sees concrete patterns of the right output for similar inputs → cleaner, more accurate slot extraction
- Same retrieval idea repeats for the ResponseGen prompt
- Or, alternatively, the retrieved examples can be injected into the single prompt of Exp1 instead of the modular pipeline of Exp2, giving a retrieval-augmented version of the single-LLM architecture

**What's needed:** A vector index over the MultiWOZ training set (e.g. FAISS + a sentence embedding model). 
A retriever module before each LLM call (DST, ResponseGen) that pulls the top-k similar examples and injects them into the prompt. 
Tunable: k (number of examples), retriever model, similarity metric

**Wins:** Often gives substantial JGA gains over zero-shot on the same backbone (Hudecek & Dušek report this directly), no fine-tuning needed, easy to add domains by indexing more data

**Costs:** Slightly larger prompts (more tokens per call), need to maintain a vector index, retriever quality becomes a new failure mode (bad retrievals → misleading examples → wrong outputs)

**Note:** This is the closest "retrieval-augmented" cousin of Exp2, conceptually between zero-shot (Exp2) and full fine-tuning (Exp3) as it gets some training-data benefit without changing model weights

---

## 9. Data-side improvements: cleaning MultiWOZ 2.2 and collecting real dialogues
**Problem:** All previous extensions improve the *model* or the *pipeline*. None of them touch the *data*. But MultiWOZ 2.2, despite being the corrected 
version of MultiWOZ, still contains noisy, mislabeled, and ambiguous examples. The kind of issues DAIR tries to handle implicitly through reweighting. 
And even a clean MultiWOZ is a synthetic Wizard-of-Oz benchmark, not real customer-service conversations. A pipeline that scores well on MultiWOZ may behave 
very differently on the messy, half-formed, multi-intent messages real hotel and restaurant customers actually send

**Two complementary directions:**

### 9.1 Clean MultiWOZ 2.2
Run an LLM-assisted audit over the MultiWOZ 2.2 train and dev splits to flag suspicious belief-state annotations, missing slots, contradictory responses, 
and unrealistic turn flows. Each flagged turn is reviewed by a human annotator and either corrected or dropped. The cleaned subset becomes a higher-quality 
training and evaluation set, on top of which the existing experiments can be re-run.

**Wins:** Expected improvement in JGA simply from cleaner labels (no model change), per-turn error analysis becomes more meaningful, results are easier to defend in review

**Costs:** Annotator time and cost, the cleaned set is no longer directly comparable to published MultiWOZ 2.2 numbers (a reproducibility note becomes mandatory in any results table)

### 9.2 Collect real customer-service dialogues
Partner with one or more real hospitality businesses (a hotel chain, a restaurant group) to obtain anonymized customer-service transcripts. 
Get a small panel of domain experts (front-desk staff, restaurant managers) to annotate slot extractions, intents, and ideal responses on a sample of these dialogues. 
The result is a small but real evaluation set that complements MultiWOZ.

**Wins:** Far more realistic input distribution (multi-intent messages, hospitality-specific edge cases), exposes failure modes invisible on MultiWOZ, 
lets us measure whether MultiWOZ-trained pipelines actually transfer to production conversations

**Costs:** GDPR / data-protection compliance work upfront, business partnership negotiations, expert annotation time. 
The real-data set will be small (low hundreds of dialogues) and will need to be combined with MultiWOZ to remain statistically meaningful

### 9.3 Generate synthetic dialogues from a real seed
After collecting a small set of real hospitality dialogues (9.2), use a strong LLM (GPT-4o, Claude, etc.) as a *user simulator* and *agent simulator* to generate 
thousands of additional synthetic dialogues that follow the same patterns, slot distributions, and edge cases as the real seed. The result is a much larger 
training set: real dialogues anchor the realism, synthetic dialogues fill in coverage gaps the real seed is too small to cover

**Reference:** Wang et al. 2024, "LUAS" arXiv: 2407.20655

It uses GPT-4 to simulate user-agent dialogues on top of MultiWOZ and full-fine-tunes LLaMA-2-7B on real + synthetic data, reporting +4.3 JGA over real-only baseline. 
Same idea applies in production: bootstrap from a small real dataset and scale up cheaply with synthetic data

**Wins:** Cheap data scaling without re-engaging real customers or domain experts, easy to re-generate when business rules or product features change, 
exposes the model to rare scenarios that the real seed only contains a handful of times

**Costs:** Synthetic quality is bounded by the simulator LLM (a GPT-4 simulator can't generate situations GPT-4 itself doesn't model well), risk of distribution 
drift if the synthetic data overwhelms the real seed in volume, requires careful filtering to drop unrealistic synthetic dialogues

**Combined plan:** The three directions are complementary:
- 9.1 strengthens the synthetic benchmark we publish results on
- 9.2 sanity-checks whether those results carry over to real hospitality conversations
- 9.3 scales the real seed from 9.2 cheaply, so a production system has enough training data without months of expert annotation

A serious production system does all three: clean the public benchmark, collect a small real seed, and synthetically scale that seed to training-set size

---

## 10. Adding RL-style alignment on top of LoRA fine-tuning (as Experiment 4)
**Idea:** Modern instruction-tuned LLMs go through three training stages before release: pretraining on raw text, instruction tuning on (instruction, response) pairs, 
and preference alignment via RLHF or similar. We start from models that already completed all three, and Exp3 adds supervised LoRA fine-tuning specialized on 
MultiWOZ. The natural Exp4 adds the preference-alignment equivalent on top of Exp3, completing a parallel four-stage adaptation: 
starts from already-aligned model → SFT-LoRA on TOD data (Exp3) → preference-LoRA on TOD preferences (Exp4)

**Flow:**
```
Exp3 LoRA-tuned modular pipeline → generate response candidates → [GPT-4o-mini or Claude Haiku ranks them: preferred vs rejected] → [DPO trainer on preference pairs] → Exp4 LoRA + DPO modular pipeline
```

**Method choice of DPO over RLHF/RLAIF:**
DPO (Direct Preference Optimization) is the simpler, more stable alternative to RLHF. It needs only preference pairs (preferred output + rejected output) and a contrastive 
loss. No reward model, no PPO. DIMF (Feng et al. 2025, ref [15]) uses exactly this on TOD with reported improvements over standard fine-tuning. Available in Unsloth and 
HuggingFace TRL, so the engineering cost from Exp3 to Exp4 is small

**Synthetic preferences:** Real human preference rankings are expensive. A practical workaround is RLAIF-style synthesis. For each training dialogue, the Exp3 model generates 
a response (treated as "rejected") and a stronger LLM (GPT-4o-mini, Claude Haiku) generates a better response (treated as "preferred"). This is cheaper than human 
labeling but should be spot-checked on 50–100 pairs to confirm the preferences are meaningful

**What's needed:** Unsloth + TRL DPO trainer integration on top of the existing Exp3 LoRA setup. A preference-pair dataset generation script. Light spot-check of synthetic 
preferences. Validation that DPO doesn't degrade general response quality

**Costs:** Synthetic preferences are weaker than human ones (some "preferred" outputs aren't actually better). DPO can degrade general behavior if over-trained. 
Automatic metrics (Inform/Success/BLEU/Combined) may not move much as DPO often improves qualitative response style more than quantitative scores. 
A separate qualitative evaluation may be needed to surface the gain

---

## 11. Putting it all together: from MLP4CS to a commercial MVP
The 10 sections above are individual extensions. A real product picks the relevant ones, sequences them, and adds the production essentials this project does not cover. 
The recommended path from MLP4CS toward a domain-specific MVP for hospitality customer service:

**Foundation → choose the deployment shape:**
- One product per domain (hotel-only, restaurant-only), not multi-domain
- Workflow graph at the system level (Section 6) with LLM modules inside nodes
- Schema-grounded prompts inside each module (Section 5)
- Tool calling for DST and DB lookup (Section 7)

**Reliability → handle real-user input:**
- Multi-user rewriter at the entry of the pipeline (Section 1)
- User-clarification policy with budget (Section 3)
- ReAct-style retry on zero-result DB queries (Section 2)
- Reflexion-style supervisor for self-correction within a session (Section 4)

**Quality → close the gap with real conversations:**
- Few-shot retrieval-augmented prompts seeded from real customer logs (Section 8)
- Data-side improvements: clean evaluation set + small real-dialogue benchmark +  synthetic scaling (Section 9)
- Optional: DPO post-training for response style alignment (Section 10)

**Production essentials not covered by the project:**
- Authentication, GDPR-compliant data storage, audit trails
- Backend integration with the business' real booking/CRM systems (PMS for hotels, POS for restaurants)
- Human handoff path when the bot fails or the user asks for a person
- Monitoring: per-turn latency, cost per dialogue, success/escalation rates, hallucination flags
- A/B testing infrastructure to compare bot versions against each other and against staff
- Operator dashboard for staff to review flagged conversations and override decisions

**MVP scope to ship:**
1. Single domain (start with restaurant as it's simpler schema, faster booking flow)
2. Workflow graph with 4–5 flows: book, modify, cancel, inform, escalate (using public datasets or a small real seed collected from the business, see Section 9). 
   Two practical sources for the flow definitions:
   - **Hand-design the flows with the business owner** 
   - **Mine flows from existing customer-service logs** 
3. Tool calling on top of an instruction-tuned LLM (no fine-tuning yet)
4. Real-data benchmark of ~100 anonymized dialogues for offline evaluation
5. Operator dashboard + escalation path

**Adding fine-tuning (Sections after MVP):**
Bring in LoRA SFT (Exp3-style) and DPO (Section 10) only after the MVP has 1000+ real dialogues collected. Fine-tuning before that point overfits 
on synthetic MultiWOZ data and wastes the personalization opportunity from real customer conversations.

---

## Further Reading
Useful papers that don't tie to a specific extension above, but worth re-reading when designing the next iteration.

- Agentic AI (general overview)
  - Sapkota, Roumeliotis, Karkee, "AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges", Information Fusion 126, 2026.
    URL: https://doi.org/10.1016/j.inffus.2025.103599
  - Broad conceptual overview distinguishing single-entity tool-augmented LLM agents ("AI Agents") from multi-agent orchestrated systems ("Agentic AI").
    Useful as a vocabulary primer, not TOD or MultiWOZ-specific

- Agentic architectures (general overview)
  - Masterman et al. 2024, "The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey" arXiv: 2404.11584
  - Practitioner-friendly map of single/multi-agent patterns, leadership, reflection phases. Good entry point for revisiting the agentic literature

- Agent evaluation (general overview)
  - Mohammadi et al. 2025, "Evaluation and Benchmarking of LLM Agents: A Survey" arXiv: 2507.21504 
    
    Two-dimensional taxonomy: what to evaluate (behavior, capabilities, reliability, safety) and how to evaluate (interaction modes, datasets, metrics, tooling). 
    Useful starting point when designing evaluation for an agentic customer service system beyond standard TOD metrics
  - Liu et al. 2023, "G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment" arXiv: 2303.16634 
  
    Concrete LLM-as-judge implementation: GPT-4 with chain-of-thought scores generated text on coherence, fluency, consistency, relevance.
    Useful for production evaluation where ground-truth labels are unavailable (real customer logs, no reference responses)
  - Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" arXiv: 2306.05685 
    
    Foundational LLM-as-judge paper. Pairwise comparison + ELO rankings. The standard reference when comparing bot versions in production without ground-truth labels

- Customer service in related domains
  - Valentini et al. 2025, "Leveraging LLM-Powered Multi-Agent Systems to Enhance Customer Experience in Complex Product Domains"
    
    LLM multi-agent system for furniture retail. Manager agent orchestrates specialized sub-agents (product info, style advice, recommendation) with retrieval grounding. 
    Useful as a real industrial deployment example when designing customer service products in domains other than hotel/restaurant booking 

---

## Hardware Notes: How to Run These Experiments 

All the above extensions are worth trying as ablation experiments to compare architectures for a real customer service product, on both API-based and open-source models.

API models (GPT-4o-mini, Claude Haiku, etc.) need only API keys and budget, no GPU.

Open-source models (LLaMA, Qwen, etc.) of different sizes (3B–14B) need a GPU with at least ≥40 GB VRAM for inference and fine-tuning.

**GPU access options:**
- **EuroHPC:** apply to other EuroHPC sites. Free for academic research
  - **Independent practitioners** can still apply for access, which is explicitly open to individuals with smaller allocations, but free. Application takes weeks
- **Cloud GPU rental (pay per hour), no commitment:**
  - RunPod, Lambda Labs, Vast.ai: A100 40GB, A100 80GB, H100. Best price/performance for short experiments
  - Together.ai, Fireworks, Replicate: serverless inference on hosted open models. No GPU rental, pay per token like an API. Good for inference, not fine-tuning
- **Major cloud providers (AWS, GC Vertex, Azure):** more expensive ($3–6/hr for A100). Credits also require a registered company (Greek IKE works), if not academic affiliation
- **Anthropic / OpenAI credit programs:**
  - Researcher credits: require academic affiliation
  - Startup credits: require a registered company (Greek IKE works), no academic affiliation needed. 
    Anthropic, OpenAI, Google all offer $5–200K credit programs for early-stage startups

- **Buying a GPU:** 
  - Consumer top is RTX 5090 (32GB GDDR7) in Greece. Handles 8B fine-tuning and 14B inference, but not 14B fine-tuning
  - For ≥48GB (true 14B fine-tuning + larger models), only workstation cards apply: NVIDIA RTX 6000 Ada (48GB). Worth it only for years of heavy use, otherwise rental wins on flexibility

**Recommended path:** rent A100 hours. Use API providers for prototype-level testing of any extension before committing GPU time

---
