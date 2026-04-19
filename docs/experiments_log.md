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

**Next step:** We want to test whether splitting the task into focused sub-tasks (DST, ResponseGen) improves performance, especially for smaller models, 
which struggled most with the single-prompt complexity → **Exp2**.

---

## Exp2: Zero-Shot Modular Pipeline
**Setup:** Split the turn into two LLM calls: DST (extract slots) and ResponseGen (generate text given DB results). Both zero-shot. Tested homogeneous (same model for both) and heterogeneous (different models per role) configurations.

**Tested:** homo_qwen3_14b, hetero_qwen25_qwen3_14b, homo_qwen3_8b, hetero_qwen3_14b_qwen3_8b.

### Run 1: homo_qwen3_14b
- **Why:** Test whether decomposition helps the strongest open-source model from Exp1
- **Result:** ...
- **Finding:** ...
- **Next:** ...

### Run 2: hetero_qwen25_qwen3_14b
- **Why:** Qwen2.5-14B was optimized for structured/JSON output so we use it for DST. Qwen3-14B is stronger generally on natural language so we use it for ResponseGen (specialize each module by what the model does best)
- **Result:** ...
- **Finding:** ...
- **Next:** If ...

### Run 3: homo_qwen3_8b
- **Why:** Same-family comparison, smaller model. Tests whether decomposition is a "boost for weaker models"
- **Result:** ...
- **Finding:** ...
- **Next:** ...

### Run 4: hetero_qwen3_14b_qwen3_8b
- **Why:** Strong DST + weaker ResponseGen. If DST is the bottleneck, this should approach homo_qwen3_14b quality
- **Result:** ...
- **Finding:** ...

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
**Setup:** Same modular pipeline as Exp2 (DST + ResponseGen as separate calls), but each role gets its own QLoRA adapter fine-tuned on MultiWOZ training data. 
Adapters trained on `train` split, validated on `dev` split with early stopping.

**Tested:** ft_homo_llama32_3b, ft_homo_qwen3_8b, ft_homo_qwen3_14b.

### Run 1: ft_homo_llama32_3b
- **Why:** Smallest viable open-source base. Tests if fine-tuning can lift a small model that struggled in both Exp1 (single-LLM) and Exp2 (zero-shot modular)
- **Result:** ...
- **Finding:** ...

### Run 2: ft_homo_qwen3_8b
- **Why:** Mid-size model. Tests if fine-tuning can match the strongest Exp1 and Exp2 baselines using a smaller base than they used
- **Result:** ...
- **Finding:** ...

### Run 3: ft_homo_qwen3_14b
- **Why:** Strongest open-source base. Tests if fine-tuning still helps when the same model already performs well in both Exp1 and Exp2 zero-shot
- **Result:** ...
- **Finding:** ...

### Exp3 takeaway
...

---

**Notes:** 

- `LOCAL_MAX_SEQ_LENGTH`: A single global setting in `src/config.py` controls the max sequence length for both training and inference 
  - **Exp1** uses `32768` because the single-LLM prompt packs both DBs (~23K tokens) into one call. Anything smaller cuts the prompt and breaks the output
  - **Exp2 and Exp3** use `2048`. Modular prompts are ~700 tokens, so 2048 is more than enough. Exp3 also uses 2048 for fine-tuning, matching its training-time context
  - The fact that decomposed pipelines fit in a smaller fraction of the context window is an architectural benefit as each component runs on a small prompt, which helps both training cost and deployment

---
