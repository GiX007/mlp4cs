# Experiments Log

Relative research behind each experiment: what we tried, why, and what we learned.

---

## Exp1: Single-LLM Baseline
**Setup:** One LLM handles the entire turn in a single prompt: domain, intent, slots, and response all at once.

**Tested:** GPT-4o-mini, GPT-4.1-nano, Claude Haiku, Qwen2.5-14B, Qwen3-8B, Qwen3-14B, Qwen3-4B, LLaMA-3.2-3B, LLaMA-3.1-8B, Phi-4 14B.

**Key findings:**
- Qwen3-14B is the best open-source model and the most grounded overall while Haiku wins absolute scores but costs much more
- All models do worse on hotel than restaurant in most cases (hotel has more booking slots and harder attributes like stars, parking, internet)
- Smaller models (Qwen3-8B) hallucinate more when forced to handle DST + entity lookup + response generation in one prompt
- Cost vs latency tradeoff: APIs are fast (~3s) but expensive (Haiku $21, GPT $2.75), open-source is free but slower 

**Next step:** We want to test whether splitting the task into focused sub-tasks (DST, ResponseGen) improves performance, especially for smaller models, 
which struggled most with the single-prompt complexity → **Exp2**.

---

## Exp2: Zero-Shot Modular Pipeline
**Setup:** Split the turn into two LLM calls: DST (extract slots) and ResponseGen (generate text given DB results). Both zero-shot. Tested homogeneous (same model for both) and 
heterogeneous (different models per role) configurations.

**Configs tested:**
- `homo_gpt`, `homo_haiku`, `homo_qwen3_14b`, `homo_qwen3_8b`, `homo_qwen3_4b`, `homo_phi4_14b`, `homo_llama32_3b`, `homo_llama31_8b`: does decomposition help this model?
- `hetero_gpt_haiku`, `hetero_haiku_gpt`: which API role matters more, DST or ResponseGen?
- `hetero_qwen25_qwen3_14b`: Qwen2.5 (JSON-optimized) for DST + Qwen3 (stronger NLG) for ResponseGen
- `hetero_qwen3_14b_qwen3_8b`: strong DST + weak ResponseGen, isolates the DST bottleneck
- `hetero_qwen3_14b_phi4_14b`: tests if Phi-4's weak homo Exp2 is bottlenecked by its DST

**Key findings:**
- **Decomposition helps weak models, hurts the strongest open-source ones (vertical):**  
  LLaMA-3.1-8B +37 Combined (10→47), GPT-4o-mini +24 (54→79), Qwen3-4B +19 (48→67), Qwen3-8B +10 (42→52), Haiku slightly better but almost flat (~91, already near ceiling), 
  Qwen3-14B **−5** (81→76), Phi-4 14B **−12** (67→55). The weaker the base model, the more decomposition helps. LLaMA-3.1-8B's gain is the largest. Phi-4 14B regressed even more sharply 
  than Qwen3-14B, confirming that "decomposition hurts strong models" generalizes across families at the 14B scale

- **Hetero "specialize by strength" did NOT pay off (horizontal):**  
  - `hetero_qwen25_qwen3_14b` (71.8) < `homo_qwen3_14b` (76.0). Qwen2.5 produced worse slots (JGA 30 vs 33), so the JSON-optimization advantage didn't help  
  - `hetero_haiku_gpt` (87.7) < `homo_haiku` (91.4). Downgrading ResponseGen (gpt-4o-mini) alone cost 3.7 pts 

- **DST is the bottleneck:**  
  With one Haiku + one GPT, Haiku-as-DST scores 87.7 vs GPT-as-DST 79.2. An 8.4-point gap from role assignment alone. 
  Downgrading DST costs ~3× more than downgrading ResponseGen (`homo_haiku` → `hetero_gpt_haiku`: −12.1 pts vs `homo_haiku` → `hetero_haiku_gpt`: −3.7 pts). 
  The Qwen pair confirms it: `hetero_qwen3_14b_qwen3_8b` (54.1) ≈ `homo_qwen3_8b` (52.1). A strong DST barely rescues a weak ResponseGen, but a weak DST drags everything down.
  **Cross-family confirmation:** `hetero_qwen3_14b_phi4_14b` (Qwen3 DST + Phi-4 RG) jumps to 68.5, **+13 over `homo_phi4_14b` (55.2)**. Swapping Phi-4's weak DST for Qwen3's stronger one 
  is enough to lift the whole pipeline. The DST bottleneck rule holds across families

- **Hallucination drops for most models, but the pattern depends on Exp1 starting point:**  
  - Models with high Exp1 Hall% drop dramatically: LLaMA-3.1-8B 53.7→5.1, Qwen3-4B 13.2→3.2, LLaMA-3.2-3B (Exp1 failed)
  - API models drop sharply from already-low values: GPT 5.9→3.1, Haiku 4.4→0.8 (placeholder discipline works)
  - Already-disciplined Exp1 models stay flat or rise slightly: Qwen3-14B 2.7→3.0, Phi-4 14B 7.2→9.2, Qwen3-8B 9.7→11.2
  - Pattern: decomposition forces the model to focus only on the response with explicit DB results, removing the need to invent facts. Models that hallucinated heavily 
    in Exp1 benefit most. Models that were already grounded in Exp1 see little change

- **Latency is asymmetric:**  
  - API models get **slower** (network overhead × 2 calls). Local models get **3–5× faster** (shorter prompts beat the 2-call overhead)
  - Cost drops for API models as expected

**Both modules matter:** weak DST → wrong slots, weak ResponseGen → hallucinated or off-policy text. Fixing just one doesn't close the gap. 
Decomposition helps when the single-prompt task exceeds the model's capacity (GPT, LLaMA-3.1-8B, Qwen3-8B), and hurts when the model was already grounded and using 
the full context productively in one pass (Qwen3-14B, Phi-4 14B).

**Next step:** Fine-tuning DST and ResponseGen on MultiWOZ training data should let small models compete with large ones → **Exp3**.

---

## Exp3: LoRA Fine-Tuning
**Setup:** Same modular pipeline as Exp2 (DST + ResponseGen as separate calls), but each role gets its own QLoRA adapter fine-tuned on MultiWOZ training data. 
Adapters trained on `train` split, validated on `dev` split with early stopping.

**Tested:** ft_homo_llama32_3b, ft_homo_llama31_8b, ft_homo_qwen3_4b, ft_homo_qwen3_8b, ft_homo_qwen3_14b, ft_homo_phi4_14b, ft_hetero_qwen3_14b_phi4_14b.

**Configs at a glance:**
- `ft_homo_llama32_3b`, `ft_homo_qwen3_4b`: smallest viable bases (3-4B), does FT lift them above their Exp1 and Exp2 scores and close the gap with API giants like Haiku?
- `ft_homo_qwen3_8b`, `ft_homo_llama31_8b`: mid-size (8B), does FT lift them above their Exp1 and Exp2 scores and close the gap with API giants like Haiku, across two families? How much do they pull ahead of the smaller 3-4B bases?
- `ft_homo_qwen3_14b`, `ft_homo_phi4_14b`: strongest open-source bases (14B), does FT lift them above their Exp1 and Exp2 scores and close the gap with API giants like Haiku, across two families? How much do they pull ahead of the smaller bases (3-4B, 8B)?
- `ft_hetero_qwen3_14b_phi4_14b`: strong fine-tuned DST + strong fine-tuned ResponseGen across families, does mixing best-of-both lift performance above either homogeneous configuration?

**Key findings:**
- **Fine-tuning helps weak ("undisciplined") Exp2 bases and hurts already-strong ("disciplined") ones (vertical Exp2 → Exp3):**  
  LLaMA-3.2-3B 23 → 58 (**+35**), Phi-4 14B 55 → 71 (**+16**), Qwen3-8B 52 → 62 (**+10**), LLaMA-3.1-8B 47 → 55 (**+8**), Qwen3-14B 76 → 71 (**−5**), Qwen3-4B 67 → 54 (**−13**).
  Same pattern as decomposition in Exp2: gains come from filling capacity gaps, not from teaching strong models new tricks. The gain from the Exp2 score is not caused due to the model size 
  as Phi-4 14B (large but weak in Exp2) gains, Qwen3-4B (small but strong in Exp2) loses. More specifically, the predictor is **Exp2 hallucination level**: high-Hall% bases 
  (LLaMA-3.2-3B 14.2%, Qwen3-8B 11.2%, Phi-4 14B 9.2%) gain from FT teaching them placeholder discipline, while already-disciplined bases (Qwen3-14B 3.0%, Qwen3-4B 3.2%) have no room 
  to gain. Phi-4 14B confirms the rule at 14B scale (large + undisciplined → big gain), and Qwen3-4B is the clearest counter-example to a naive "smaller = bigger FT gain" assumption

- **DST closes the API gap. ResponseGen does not (with one exception):**  
  ft_homo_qwen3_14b vs `homo_haiku` (Exp2): JGA **47.7 vs 45.1**, SlotF1 **89.5 vs 87.3**, Combined **70.6 vs 91.4** (~21 pt gap). 
  ft_homo_qwen3_4b shows the same split: JGA jumps 26.7 → 44.8, SlotF1 77.2 → 88.3, but Inform crashes 68.3 → 54.8 and Success 59.7 → 36.0.
  Fine-tuning solves slot extraction but cannot match commercial-API response quality. When the base model was already disciplined zero-shot, FT can hurt end-to-end metrics despite improving DST, 
  under our setup, where multi-domain training turns leave the non-active domain lexicalized, the model learns the noise as signal and starts emitting real entity names instead of placeholders 
  at inference time, which Tomiinek's Inform/Success metrics explicitly penalize. Phi-4 14B is the exception: ft_homo_phi4_14b reaches Combined 70.8, the highest open-source homogeneous score in Exp3. 
  The fine-tuned hetero ft_hetero_qwen3_14b_phi4_14b lands at 70.6, essentially matching the two strongest homo configurations, confirming that after FT, a single Combined ceiling around 70-71 
  emerges across the strongest 14B configurations regardless of mixing strategy

- **Hallucination rises after fine-tuning, but family matters:**  
  46% (LLaMA-3.1-8B), 28% (LLaMA-3.2-3B), 22% (Qwen3-8B), 16% (Phi-4 14B), 16% (Qwen3-4B), 14% (Qwen3-14B) vs Exp2 placeholder discipline of 0.8% (Haiku) and 3% (Qwen3-14B zero-shot).  
  LLaMA models are hit hardest (28-46%), Qwen and Phi-4 land at moderate levels (14-22%). The LLaMA family is more sensitive to noisy delexicalized training data than Qwen or Phi-4.
  Likely cause: multi-domain turns leave the non-active domain lexicalized. Small models learn the noise as signal and produce real names at inference instead of placeholders

- **The bottleneck shifts after fine-tuning:**  
  In Exp2, DST was the dominant bottleneck (cross-family hetero `hetero_qwen3_14b_phi4_14b` jumped +13 over `homo_phi4_14b`).
  In Exp3, after both adapters are fine-tuned, the two DSTs end up roughly equal (Qwen3-14B FT JGA 47.7 ≈ Phi-4 14B FT JGA 47.0). The hetero `ft_hetero_qwen3_14b_phi4_14b` (Combined 70.56) 
  lands within 0.3 points of both homo configs (`ft_homo_qwen3_14b` 70.61, `ft_homo_phi4_14b` 70.84). Looking per component, the hetero's response-generation metrics (Inform 66.7, Success 57.0)
  track its ResponseGen model (Phi-4 FT: 66.7 / 57.5), while its slot-tracking metrics (JGA 46.9, SlotF1 89.2) track its DST model (Qwen3 FT: 47.7 / 89.5). After FT, both DSTs are good 
  enough, so the bottleneck moves from DST to ResponseGen and end-to-end scores are driven by ResponseGen quality

- **Latency matches APIs at zero cost:**  
  All three FT models run at 2–3.5s/turn, similar to GPT/Haiku. Free, fast, and competitive on DST, viable for production if the hallucination issue is addressed

- **Cross-family pattern confirmed across three families:** LLaMA (3B, 8B), Qwen (4B, 8B, 14B), and Phi-4 (14B) all show the same trajectory shape: DST gains in Exp3 via FT, with rising 
  hallucination producing end-to-end scores. The "FT helps undisciplined models" rule holds across families.
  Family-level differences: LLaMA is most vulnerable to hallucination explosion (28-46% post-FT). Qwen and Phi-4 stay at moderate levels (14-22%).
  Phi-4 14B is the only 14B model where FT clearly helps Combined (+16), confirming the rule depends on Exp2 discipline rather than model size alone.
  Cross-family heteros (Qwen3-14B paired with Phi-4 14B) further validate that the same patterns hold when components are mixed across families with DST bottleneck in Exp2, ResponseGen-dominated ceiling in Exp3

---

## Full Metric Tables
This section archives the complete custom-metric results for every Exp1, Exp2, and Exp3 run. The README's results section reports only the official MultiWOZ leaderboard metrics(JGA, SlotF1, Inform, 
Success, BLEU, Combined). The tables below contain the additional internal metrics used for error analysis and extra studies: domain/intent precision, action accuracy, slot recall, hallucination 
rate, policy violation rate, system correctness, booking rate, cost, and latency. All numbers are computed on the MultiWOZ 2.2 test split (hotel + restaurant domains) over 186 dialogues.

### Experiment 1: Single-LLM Baseline

| Config     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU | Combined | Cost($) | Latency(s) |
|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|------|----------|---------|------------|
| gpt        | 98.7     | 87.0     | 77.6    | 24.8 | 64.7   | 71.2    | 5.9   | 0.8      | 94.9     | 57.4  | 58.6    | 44.1     | 3.08 | 54.43    | $2.75   | 3.25s      |
| gpt-nano   | 96.9     | 84.1     | 75.3    | 31.4 | 74.6   | 78.7    | 12.4  | 0.3      | 97.2     | 33.8  | 33.9    | 19.4     | 3.96 | 30.61    | $1.80   | 2.10s      |
| haiku      | 98.8     | 90.1     | 82.8    | 44.8 | 84.3   | 88.3    | 4.4   | 3.6      | 92.6     | 76.7  | 90.9    | 83.3     | 3.67 | 90.77    | $21.03  | 3.04s      |
| qwen3_4b   | 98.0     | 83.9     | 65.9    | 14.8 | 51.4   | 54.7    | 13.2  | 16.6     | 75.2     | 34.7  | 50.50   | 39.80    | 2.59 | 47.74    | $0.00   | 7.82s      |
| qwen3_8b   | 99.2     | 88.5     | 74.1    | 32.3 | 78.6   | 82.1    | 9.7   | 2.2      | 94.5     | 64.3  | 43.0    | 31.2     | 4.48 | 41.58    | $0.00   | 7.72s      |
| qwen25_14b | 98.1     | 92.9     | 77.7    | 26.4 | 68.7   | 74.9    | 5.7   | 0.7      | 94.3     | 57.9  | 74.2    | 54.8     | 3.37 | 67.87    | $0.00   | 13.28s     |
| qwen3_14b  | 98.7     | 89.3     | 78.4    | 33.3 | 78.6   | 83.0    | 2.7   | 1.9      | 96.0     | 75.8  | 82.3    | 72.0     | 3.69 | 80.84    | $0.00   | 12.40s     |
| llama31_8b | 90.8     | 67.9     | 67.8    | 5.2  | 37.0   | 33.1    | 53.7  | 0.1      | 87.0     | 0.0   | 14.00   | 2.20     | 2.30 | 10.40    | $0.00   | 7.22s      |
| phi4_14b   | 97.0     | 83.7     | 78.8    | 25.8 | 74.0   | 74.3    | 7.2   | 5.1      | 90.5     | 74.8  | 66.10   | 59.70    | 4.46 | 67.36    | $0.00   | 13.81s     |

Per-Domain Breakdown

| Config     | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|------------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| gpt        | hotel      | 99.2     | 84.5     | 77.0    | 17.9 | 60.7   | 68.1    | 6.0   | 0.6      | 95.2     | 51.9  | $1.53   | 3.23s      |
| gpt        | restaurant | 98.2     | 90.4     | 78.4    | 33.5 | 69.9   | 75.3    | 5.7   | 1.0      | 94.7     | 60.3  | $1.21   | 3.27s      |
| gpt-nano   | hotel      | 96.7     | 81.3     | 74.3    | 21.7 | 67.7   | 73.1    | 22.5  | 0.6      | 96.3     | 31.7  | $1.05   | 2.10s      |
| gpt-nano   | restaurant | 97.1     | 87.7     | 76.7    | 44.4 | 83.7   | 86.1    | 5.6   | 0.0      | 98.4     | 27.0  | $0.79   | 2.09s      |
| haiku      | hotel      | 99.8     | 90.2     | 82.2    | 37.2 | 81.6   | 86.5    | 5.9   | 4.0      | 90.8     | 74.0  | $11.82  | 3.07s      |
| haiku      | restaurant | 99.5     | 91.9     | 83.7    | 55.5 | 88.2   | 91.0    | 1.5   | 3.1      | 95.5     | 76.7  | $8.99   | 3.00s      |
| qwen3_4b   | hotel      | 97.1     | 82.5     | 66.9    | 13.8 | 53.4   | 56.3    | 10.7  | 18.4     | 75.3     | 31.2  | $0.00   | 8.12s      |
| qwen3_4b   | restaurant | 99.2     | 85.8     | 64.3    | 16.4 | 48.8   | 52.5    | 16.4  | 14.2     | 74.8     | 42.9  | $0.00   | 7.38s      |
| qwen3_8b   | hotel      | 98.8     | 86.8     | 73.2    | 23.0 | 73.9   | 78.6    | 11.0  | 1.8      | 94.9     | 66.5  | $0.00   | 7.59s      |
| qwen3_8b   | restaurant | 99.7     | 90.8     | 75.4    | 45.1 | 85.1   | 86.9    | 8.4   | 2.9      | 93.9     | 51.0  | $0.00   | 7.79s      |
| qwen25_14b | hotel      | 99.8     | 92.3     | 76.4    | 20.5 | 64.5   | 71.4    | 2.9   | 0.6      | 96.8     | 41.1  | $0.00   | 13.32s     |
| qwen25_14b | restaurant | 97.9     | 94.9     | 79.8    | 34.4 | 74.3   | 79.5    | 8.9   | 0.8      | 91.5     | 58.1  | $0.00   | 13.22s     |
| qwen3_14b  | hotel      | 99.8     | 89.7     | 78.3    | 24.4 | 73.9   | 79.7    | 3.0   | 0.0      | 97.4     | 77.5  | $0.00   | 12.56s     |
| qwen3_14b  | restaurant | 98.7     | 90.2     | 78.7    | 45.2 | 85.2   | 87.8    | 2.2   | 4.4      | 94.1     | 68.7  | $0.00   | 12.27s     |
| llama31_8b | hotel      | 89       | 68.3     | 69      | 4.2  | 46.7   | 40.5    | 42.7  | 0        | 93.6     | 0     | $0.00   | 7.01s      |
| llama31_8b | restaurant | 93.7     | 67.2     | 65.8    | 6.6  | 23.8   | 23.5    | 59.3  | 0.3      | 75.2     | 0     | $0.00   | 6.58s      |
| phi4_14b   | hotel      | 99.6     | 79.6     | 77.8    | 17.8 | 68.9   | 70.0    | 8.2   | 6.6      | 89.3     | 64.6  | $0.00   | 13.42s     |
| phi4_14b   | restaurant | 94.4     | 88.4     | 80.0    | 36.0 | 80.6   | 80.1    | 6.4   | 3.6      | 91.3     | 78.6  | $0.00   | 12.54s     |



### Experiment 2: Modular Zero-Shot Pipeline

| Config                    | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU | Combined | Cost($) | Latency(s) |
|---------------------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|------|----------|---------|------------|
| homo_gpt                  | 99.6     | 90.9     | 80.2    | 35.7 | 80.1   | 81.7    | 3.1   | 3.4      | 94.6     | 65.9  | 79.0    | 72.0     | 3.14 | 78.64    | $0.20   | 4.00s      |
| homo_haiku                | 99.8     | 91.0     | 81.1    | 45.1 | 84.0   | 87.3    | 0.8   | 5.1      | 94.4     | 75.2  | 90.3    | 83.9     | 4.25 | 91.35    | $2.10   | 4.73s      |
| hetero_gpt_haiku          | 99.8     | 91.5     | 80.2    | 34.7 | 80.3   | 81.9    | 1.0   | 4.7      | 94.5     | 69.7  | 78.0    | 72.6     | 3.92 | 79.22    | $0.78   | 3.70s      |
| hetero_haiku_gpt          | 99.8     | 91.2     | 80.7    | 44.6 | 83.8   | 87.0    | 2.1   | 3.7      | 95.4     | 71.4  | 88.2    | 81.2     | 2.95 | 87.65    | $1.53   | 4.86s      |
| homo_qwen3_14b            | 99.8     | 89.1     | 77.9    | 33.4 | 79.9   | 80.4    | 3.0   | 1.1      | 97.1     | 70.0  | 73.7    | 68.3     | 5.00 | 76.00    | $0.00   | 2.50s      |
| hetero_qwen25_qwen3_14b   | 99.4     | 88.7     | 76.4    | 30.1 | 77.2   | 76.9    | 4.2   | 2.4      | 95.4     | 58.5  | 73.7    | 60.8     | 4.54 | 71.79    | $0.00   | 2.56s      |
| homo_qwen3_4b             | 99.3     | 85.4     | 73.1    | 26.7 | 76.9   | 77.2    | 3.2   | 5.2      | 93.0     | 63.3  | 68.30   | 59.70    | 3.11 | 67.11    | $0.00   | 2.26s      |
| homo_qwen3_8b             | 99.4     | 91.4     | 78.2    | 34.2 | 79.0   | 81.3    | 11.2  | 2.8      | 94.5     | 61.8  | 51.6    | 42.5     | 5.10 | 52.15    | $0.00   | 2.18s      |
| hetero_qwen3_14b_qwen3_8b | 99.6     | 91.1     | 81.6    | 33.7 | 80.2   | 81.6    | 10.9  | 2.2      | 94.8     | 68.0  | 52.7    | 45.2     | 5.20 | 54.15    | $0.00   | 2.46s      |
| homo_llama32_3b           | 86.4     | 69.9     | 65.5    | 5.3  | 40.5   | 42.5    | 14.2  | 4.9      | 89.7     | 22.1  | 28.50   | 14.50    | 1.26 | 22.76    | $0.00   | 3.17s      |
| homo_llama31_8b           | 96.9     | 85.8     | 76.0    | 18.1 | 60.3   | 63.4    | 5.1   | 1.8      | 95.2     | 40.7  | 55.40   | 35.50    | 1.53 | 46.98    | $0.00   | 3.26s      |
| homo_phi4_14b             | 99.3     | 91.8     | 76.8    | 23.8 | 73.8   | 69.7    | 9.2   | 9.1      | 88.9     | 46.1  | 61.30   | 43.50    | 2.83 | 55.23    | $0.00   | 2.75s      |
| hetero_qwen3_14b_phi4_14b | 99.6     | 91.6     | 80.4    | 33.7 | 80.8   | 81.1    | 7.9   | 3.0      | 93.9     | 72.8  | 68.30   | 62.90    | 2.91 | 68.51    | $0.00   | 2.88s      |

Per-Domain Breakdown

| Config                    | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|---------------------------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| homo_gpt                  | hotel      | 99.6     | 88.5     | 79.4    | 25.7 | 75.7   | 78.3    | 3.4   | 3.4      | 94.3     | 67.3  | $0.12   | 4.03s      |
| homo_gpt                  | restaurant | 99.5     | 94.0     | 81.4    | 48.8 | 86.0   | 86.2    | 2.7   | 3.4      | 95.1     | 72.3  | $0.08   | 3.96s      |
| homo_haiku                | hotel      | 99.8     | 89.9     | 79.9    | 35.3 | 80.2   | 84.2    | 1.2   | 5.4      | 93.5     | 72.4  | $1.21   | 4.88s      |
| homo_haiku                | restaurant | 99.7     | 92.3     | 82.7    | 58.7 | 89.6   | 91.9    | 0.3   | 4.8      | 95.2     | 77.8  | $0.86   | 4.63s      |
| hetero_gpt_haiku          | hotel      | 100.0    | 89.7     | 79.3    | 25.0 | 76.4   | 79.0    | 1.3   | 5.4      | 93.7     | 67.3  | $0.46   | 3.73s      |
| hetero_gpt_haiku          | restaurant | 99.5     | 93.8     | 81.4    | 47.3 | 85.4   | 85.6    | 0.7   | 3.9      | 95.6     | 72.5  | $0.32   | 3.67s      |
| hetero_haiku_gpt          | hotel      | 99.8     | 90.1     | 79.4    | 35.6 | 80.4   | 84.3    | 3.0   | 3.4      | 95.0     | 69.8  | $0.87   | 4.98s      |
| hetero_haiku_gpt          | restaurant | 99.7     | 92.6     | 82.4    | 58.0 | 89.1   | 91.3    | 1.0   | 4.3      | 95.7     | 75.2  | $0.63   | 4.82s      |
| homo_qwen3_4b             | hotel      | 100.0    | 84.3     | 72.4    | 22.1 | 74.8   | 75.8    | 3.4   | 6.0      | 92.0     | 60.3  | $0.00   | 2.25s      |
| homo_qwen3_4b             | restaurant | 98.5     | 86.8     | 74.0    | 32.6 | 79.5   | 79.1    | 2.8   | 4.1      | 94.4     | 64.7  | $0.00   | 2.28s      |
| homo_qwen3_14b            | hotel      | 99.8     | 87.4     | 78.4    | 25.3 | 76.0   | 77.2    | 3.2   | 0.6      | 97.2     | 66.3  | $0.00   | 2.51s      |
| homo_qwen3_14b            | restaurant | 99.7     | 91.3     | 77.1    | 44.6 | 85.4   | 84.9    | 2.7   | 1.8      | 96.9     | 70.6  | $0.00   | 2.51s      |
| hetero_qwen25_qwen3_14b   | hotel      | 100.0    | 88.0     | 75.5    | 18.6 | 69.1   | 70.3    | 5.7   | 1.0      | 96.0     | 48.0  | $0.00   | 2.53s      |
| hetero_qwen25_qwen3_14b   | restaurant | 98.7     | 89.5     | 77.6    | 44.6 | 87.5   | 85.2    | 2.3   | 4.1      | 94.6     | 69.7  | $0.00   | 2.60s      |
| homo_qwen3_8b             | hotel      | 99.4     | 90.3     | 78.1    | 25.7 | 74.4   | 78.0    | 14.4  | 1.6      | 95.6     | 61.2  | $0.00   | 2.19s      |
| homo_qwen3_8b             | restaurant | 99.5     | 92.7     | 78.3    | 45.3 | 85.0   | 85.7    | 8.1   | 4.4      | 93.0     | 64.9  | $0.00   | 2.18s      |
| hetero_qwen3_14b_qwen3_8b | hotel      | 99.8     | 90.1     | 82.1    | 25.2 | 76.2   | 78.5    | 13.5  | 1.0      | 95.6     | 68.2  | $0.00   | 2.48s      |
| hetero_qwen3_14b_qwen3_8b | restaurant | 99.2     | 92.5     | 81.0    | 44.9 | 85.7   | 86.1    | 8.1   | 3.9      | 93.8     | 63.3  | $0.00   | 2.44s      |
| homo_llama32_3b           | hotel      | 93.7     | 69.3     | 69.1    | 6.1  | 41.7   | 43.7    | 15.6  | 2.4      | 90.7     | 20.5  | $0.00   | 3.16s      |
| homo_llama32_3b           | restaurant | 80.1     | 70.4     | 62.4    | 4.7  | 39.8   | 41.7    | 12.6  | 7.3      | 88.4     | 18.8  | $0.00   | 3.25s      |
| homo_llama31_8b           | hotel      | 99.2     | 89.0     | 77.7    | 13.7 | 58.4   | 62.4    | 3.0   | 1.3      | 96.8     | 30.5  | $0.00   | 3.26s      |
| homo_llama31_8b           | restaurant | 94.4     | 82.2     | 74.1    | 23.5 | 62.3   | 64.6    | 7.8   | 2.4      | 93.2     | 43.4  | $0.00   | 3.05s      |
| homo_phi4_14b             | hotel      | 100.0    | 90.8     | 74.7    | 16.8 | 69.1   | 65.0    | 14.1  | 14.3     | 83.6     | 16.0  | $0.00   | 3.02s      |
| homo_phi4_14b             | restaurant | 98.4     | 93.2     | 79.5    | 33.6 | 80.0   | 75.9    | 5.8   | 3.5      | 94.3     | 77.4  | $0.00   | 2.65s      |
| hetero_qwen3_14b_phi4_14b | hotel      | 99.8     | 90.7     | 80.3    | 27.2 | 76.9   | 77.9    | 10.3  | 3.8      | 92.7     | 66.2  | $0.00   | 2.94s      |
| hetero_qwen3_14b_phi4_14b | restaurant | 99.2     | 92.7     | 80.7    | 42.3 | 86.3   | 85.5    | 5.6   | 2.1      | 95.6     | 74.3  | $0.00   | 2.81s      |

### Experiment 3: Modular Fine-Tuned Pipeline

| Config                       | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU  | Combined | Cost($) | Latency(s) |
|------------------------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|-------|----------|---------|------------|
| ft_homo_qwen3_4b             | 99.7     | 95.2     | 78.3    | 44.8 | 86.2   | 88.3    | 16.2  | 0.7      | 94.3     | 43.0  | 54.80   | 36.00    | 8.35  | 53.75    | $0.00   | 2.78s      |
| ft_homo_qwen3_8b             | 99.3     | 95.7     | 78.9    | 47.3 | 88.1   | 89.1    | 22.1  | 0.6      | 92.6     | 48.3  | 53.2    | 48.4     | 11.35 | 62.15    | $0.00   | 3.36s      |
| ft_homo_qwen3_14b            | 99.0     | 95.7     | 79.2    | 47.7 | 88.1   | 89.5    | 14.1  | 2.0      | 94.7     | 54.2  | 61.3    | 55.4     | 12.26 | 70.61    | $0.00   | 3.42s      |
| ft_homo_llama32_3b           | 98.2     | 94.3     | 77.1    | 43.5 | 87.5   | 87.8    | 27.9  | 0.3      | 90.6     | 44.6  | 57.0    | 39.8     | 9.39  | 57.79    | $0.00   | 2.12s      |
| ft_homo_llama31_8b           | 96.4     | 92.5     | 75.8    | 44.0 | 86.1   | 87.1    | 45.6  | 1.0      | 84.2     | 50.8  | 50.50   | 40.30    | 9.22  | 54.62    | $0.00   | 3.14s      |
| ft_homo_phi4_14b             | 99.1     | 94.5     | 78.6    | 47.0 | 88.7   | 88.9    | 15.5  | 0.8      | 94.9     | 55.0  | 66.70   | 57.50    | 8.74  | 70.84    | $0.00   | 3.44s      |
| ft_hetero_qwen3_14b_phi4_14b | 99.0     | 95.6     | 78.7    | 46.9 | 87.7   | 89.2    | 13.8  | 1.2      | 94.9     | 55.7  | 66.70   | 57.00    | 8.71  | 70.56    | $0.00   | 3.51s      |

Per-Domain Breakdown

| Config                       | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|------------------------------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| ft_homo_qwen3_4b             | hotel      | 100.0    | 94.0     | 77.1    | 35.7 | 83.6   | 86.1    | 13.9  | 0.0      | 95.4     | 82.9  | $0.00   | 2.76s      |
| ft_homo_qwen3_4b             | restaurant | 99.2     | 96.7     | 79.9    | 56.4 | 89.5   | 91.1    | 19.1  | 1.5      | 92.8     | 41.0  | $0.00   | 2.80s      |
| ft_homo_llama32_3b           | hotel      | 99.8     | 94.6     | 76.4    | 35.7 | 85.9   | 85.6    | 18.5  | 0.4      | 94.0     | 56.8  | $0.00   | 2.16s      |
| ft_homo_llama32_3b           | restaurant | 96.3     | 93.8     | 78.0    | 53.0 | 89.4   | 90.3    | 38.2  | 0.2      | 86.5     | 67.4  | $0.00   | 2.08s      |
| ft_homo_qwen3_8b             | hotel      | 99.8     | 95.2     | 77.7    | 37.5 | 86.0   | 87.2    | 19.5  | 0.4      | 94.0     | 74.2  | $0.00   | 3.39s      |
| ft_homo_qwen3_8b             | restaurant | 98.7     | 96.4     | 80.5    | 59.7 | 90.9   | 91.6    | 24.8  | 0.8      | 90.8     | 72.6  | $0.00   | 3.32s      |
| ft_homo_qwen3_14b            | hotel      | 100.0    | 95.3     | 78.7    | 40.4 | 86.7   | 88.3    | 13.3  | 1.2      | 95.9     | 82.6  | $0.00   | 3.43s      |
| ft_homo_qwen3_14b            | restaurant | 97.7     | 96.2     | 79.8    | 56.8 | 89.8   | 91.1    | 15.1  | 3.0      | 93.2     | 79.4  | $0.00   | 3.41s      |
| ft_homo_llama31_8b           | hotel      | 100.0    | 94.8     | 75.9    | 37.7 | 85.3   | 86.0    | 43.7  | 1.1      | 85.6     | 74.1  | $0.00   | 3.24s      |
| ft_homo_llama31_8b           | restaurant | 92.6     | 90.0     | 75.8    | 50.9 | 87.4   | 88.6    | 47.4  | 0.9      | 82.6     | 69.6  | $0.00   | 2.96s      |
| ft_homo_phi4_14b             | hotel      | 99.8     | 94.2     | 78.0    | 38.8 | 87.7   | 87.5    | 12.2  | 0.4      | 96.4     | 79.4  | $0.00   | 3.49s      |
| ft_homo_phi4_14b             | restaurant | 98.2     | 94.9     | 79.3    | 57.5 | 90.1   | 90.6    | 19.4  | 1.3      | 93.1     | 85.5  | $0.00   | 3.38s      |
| ft_hetero_qwen3_14b_phi4_14b | hotel      | 100.0    | 95.2     | 78.1    | 39.4 | 86.4   | 88.0    | 12.4  | 0.6      | 96.2     | 82.8  | $0.00   | 3.53s      |
| ft_hetero_qwen3_14b_phi4_14b | restaurant | 97.7     | 96.2     | 79.5    | 56.3 | 89.4   | 90.7    | 15.6  | 2.0      | 93.4     | 80.5  | $0.00   | 3.48s      |

---
