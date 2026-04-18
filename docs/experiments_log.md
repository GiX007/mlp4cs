# Experiments Log

Relative research behind each experiment: what we tried, why, and what we learned.

---

## Exp1: Single-LLM Baseline
**Setup:** One LLM handles the entire turn in a single prompt: domain, intent, slots, and response all at once. 

**Tested:** GPT-4o-mini, Claude Haiku, Qwen2.5-14B, Qwen3-8B, Qwen3-14B.

**Key findings:**
- Qwen3-14B beats all models on Combined, Slot extraction and hallucination rate
- All models do worse on hotel than restaurant in most cases (hotel has more booking slots and harder attributes like stars, parking, internet)
- Smaller models (Qwen3-8B) hallucinate more when forced to juggle DST + entity lookup + response generation in one prompt

**Next step:** We want to test whether splitting the task into focused sub-tasks (DST, ResponseGen) improves performance, 
especially for smaller models, which struggled most with the single-prompt complexity → **Exp2**.

---

## Exp2: Zero-Shot Modular Pipeline
**Setup:** Split the turn into two LLM calls: DST (extract slots) and ResponseGen (generate text given DB results). Both zero-shot. Tested homogeneous (same model for both) and heterogeneous (different models per role) configurations.

### Run 1: homo_qwen3_14b
- **Why:** Test whether decomposition helps the strongest open-source model from Exp1
- **Result:** Combined dropped from ~79 to ~69. Latency dropped from 12s to 2.6s (5× faster)
- **Finding:** Decomposition hurts strong models as they handle full-context prompts better than simplified sub-tasks
- **Next:** Test if a specialized pairing recovers the lost quality

### Run 2: hetero_qwen25_qwen3_14b
- **Why:** Qwen2.5-14B was optimized for structured/JSON output so we use it for DST. Qwen3-14B is stronger generally on natural language so we use it for ResponseGen (specialize each module by what the model does best)
- **Result:** Combined ~61, worse than homo_qwen3_14b, worse than the Qwen3-14B Exp1 baseline
- **Finding:** Qwen2.5's "JSON optimization" didn't transfer to DST here. Zero-shot DST needs reasoning over dialogue context, not just structured output
- **Next:** If decomposition hurts strong models, does it help weak ones?

### Run 3: homo_qwen3_8b
- **Why:** Same-family comparison, smaller model. Tests whether decomposition is a "boost for weaker models"
- **Result:** Combined rose from ~43 to ~53. Hallucinations also dropped
- **Finding:** Decomposition does help weak models. Model size/capability determines whether decomposition helps or hurts
- **Next:** Identify which module is the bottleneck

### Run 4: hetero_qwen3_14b_qwen3_8b
- **Why:** Strong DST + weaker ResponseGen. If DST is the bottleneck, this should approach homo_qwen3_14b quality
- **Result:** Combined ~49, worst of the four. Hallucinations also jumped
- **Finding:** ResponseGen quality matters more than expected. A weak response generator destroys the whole pipeline even with strong DST

### Exp2 takeaway
**Decomposition:**
- Helps weaker models (GPT-4o-mini, Qwen3-8B) as simpler sub-tasks fit their capacity
- Hurts stronger models (Qwen3-14B) as they lose context they could otherwise use
- Latency: Exp2 is 3-5x faster than Exp1, and smaller models are faster than larger ones, useful tradeoff for production even when quality drops
- **Why this happens:** Weaker models struggle to handle DST + DB reasoning + response generation in one prompt so splitting reduces the load per call. Stronger models use the full-context prompt to reason in one pass and splitting cuts them off from context they'd otherwise exploit

**Both modules matter:** Weak DST produces wrong slots and weak ResponseGen produces hallucinated or off-policy responses. Fixing just one doesn't close the gap.

**Next step:** As zero-shot seems to have a ceiling, fine-tuning both DST and ResponseGen on MultiWOZ training data probably should let smaller models compete with much larger ones. → **Exp3**.

---

## Exp3: LoRA Fine-Tuning
*In progress. Hypothesis: a fine-tuned small model (e.g., Qwen3-8B or LLaMA-3.1-8B) can match or beat the best Exp1 and Exp2 results at a fraction of the compute cost.*
