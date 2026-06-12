# MultiWOZ 2.2 Leaderboard Landscape

Compact comparison set covering every entry on the official MultiWOZ leaderboard with reported scores on MultiWOZ 2.2 (https://github.com/budzianowski/multiwoz).

---

## A) Pre-LLM neural DST methods

### SGD-baseline (Rastogi et al., 2019)
- **What:** A DST model that reads natural-language descriptions of slots at runtime instead of memorizing a fixed slot list during training, so it can handle 
    new slots/domains without retraining
- **Architecture:** BERT encoder of utterance + slot-description pairs + small classification heads
- **Approach:** pre-trained+fine-tuned · **Focus:** DST-only
- **Score:** JGA 42.0 on 2.2
- **Link:** https://arxiv.org/abs/1909.05855

### TRADE (Wu et al., 2019)
- **What:** A single DST model shared across all domains that writes each slot value one word at a time, copying unknown words (like a hotel name) directly from 
    the user's message instead of picking from a fixed list
- **Architecture:** GRU encoder + GRU decoder per slot with a pointer-generator copy mechanism
- **Approach:** fine-tuned · **Focus:** DST-only
- **Score:** JGA 45.4 on 2.2
- **Link:** https://arxiv.org/abs/1905.08743

### DS-DST (Zhang et al., 2019)
- **What:** A DST model that uses two strategies depending on the slot type. For free-form slots like a hotel name, it points at the value inside the conversation, 
    for categorical slots like price range or star rating it picks one option from a small fixed list
- **Architecture:** Two BERT-based heads (span head + picklist head) selected per slot
- **Approach:** pre-trained+fine-tuned · **Focus:** DST-only
- **Score:** JGA 51.7 on 2.2
- **Link:** https://arxiv.org/abs/1910.03544

### AG-DST (Tian et al., 2021)
- **What:** A DST model that does two passes: first writes a draft of the belief state, then re-reads the dialogue plus its own draft and writes a corrected version 
    that fixes earlier mistakes (missed slots, wrong values, outdated values)
- **Architecture:** Two-stage seq2seq generator (draft + amend)
- **Approach:** pre-trained+fine-tuned · **Focus:** DST-only
- **Score:** JGA 57.3 on 2.2
- **Link:** https://aclanthology.org/2021.nlp4convai-1.8/

### SDP-DST (Lee et al., 2021)
- **What:** A DST model where T5 sees two things at once for every slot: the dialogue history AND a plain-English description of what the slot means 
    (e.g. "hotel-stars: the star rating of the hotel"). Then writes the value as text. One trained T5 handles every slot by being called once per slot, each time with a different slot description in the prompt
    (e.g. for the user turn "I want a 4-star hotel in the north" T5 is called once with the hotel-area description and returns "north", 
    once with the hotel-stars description and returns "4", once with the hotel-pricerange description and returns "none")
- **Architecture:** T5 encoder–decoder + schema-driven prompts
- **Approach:** pre-trained+fine-tuned · **Focus:** DST-only
- **Score:** JGA 57.6 on 2.2
- **Link:** https://aclanthology.org/2021.emnlp-main.404/

### D3ST (Zhao et al., 2022)
- **What:** A DST model that uses T5 to extract every slot in a single pass. The prompt contains a numbered list of all slot descriptions ("1a: hotel star rating, 
    1b: parking availability...") together with the dialogue, and T5 writes out all the values at once. It works on any new schema as long as descriptions are provided.
    Effectively a faster, batched version of SDP-DST: same idea (T5 + schema descriptions in the prompt), but processes all slots in one call instead of one slot at a time
- **Architecture:** T5 encoder–decoder + natural-language schema prompts
- **Approach:** pre-trained+fine-tuned · **Focus:** DST-only
- **Score:** JGA 58.7 on 2.2
- **Link:** https://arxiv.org/abs/2201.08904

### DAIR (Huang et al., 2022)
- **What:** A DST training trick (not a new model) that down-weights noisy or mislabeled training examples so the model spends more capacity learning from clean ones. 
    It works on top of any existing DST architecture and improves accuracy by reducing the impact of annotation errors
- **Architecture:** Standard DST backbone + instance-reweighting loss
- **Approach:** pre-trained+fine-tuned · **Focus:** DST-only
- **Score:** JGA 60.0 on 2.2
- **Link:** https://arxiv.org/abs/2110.11205

---

## B) Pre-LLM end-to-end / policy fine-tuned models

### LABES (Zhang et al., 2020)
- **What:** End-to-end TOD where the belief state is a latent variable, trained semi-supervised
- **Architecture:** Seq2seq encoder–decoder with a latent belief variable
- **Approach:** fine-tuned · **Focus:** Both
- **Score:** BLEU 18.9 / Inform 68.5 / Success 58.1 / Combined 82.2
- **Link:** https://arxiv.org/abs/2009.08115

### DAMD (Zhang et al., 2019)
- **What:** End-to-end TOD with three separate decoders (belief / action / response), where training is augmented with multiple valid bot actions per turn 
    so the model learns flexibility instead of memorizing one fixed reply
- **Architecture:** GRU seq2seq with multiple decoders (belief, action, response)
- **Approach:** fine-tuned · **Focus:** Both
- **Score:** BLEU 16.4 / Inform 57.9 / Success 47.6 / Combined 84.8
- **Link:** https://arxiv.org/abs/1911.10484

### AuGPT (Kulhánek et al., 2021) 
- **What:** Fine-tunes GPT-2 end-to-end on MultiWOZ, with extra training tricks: paraphrased sentences (made by translating to another language and back) are added to 
    the training data, and the model is penalized if it gives different outputs for paraphrased versions of the same input
- **Architecture:** GPT-2 decoder generating the full belief→action→response sequence
- **Approach:** pre-trained+fine-tuned · **Focus:** Tomiinek-only
- **Score:** BLEU 16.8 / 76.6 / 60.5 / Combined 85.4
- **Link:** https://arxiv.org/abs/2102.05126

### MinTL (Lin et al., 2020) 
- **What:** End-to-end TOD by simply taking BART or T5 off the shelf and fine-tuning it on MultiWOZ. No custom architecture, no specialized modules. 
    The input is the dialogue history, the output is the bot's next response
- **Architecture:** BART/T5 encoder–decoder
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 19.4 / 73.7 / 65.4 / Combined 89.0
- **Link:** https://arxiv.org/abs/2009.12005

### SOLOIST (Peng et al., 2020) 
- **What:** End-to-end TOD via two-step training: first further-train GPT-2 on many TOD datasets (so it learns the general shape of task-oriented dialogue), 
    then fine-tune on MultiWOZ specifically to give the model a TOD head start before seeing MultiWOZ
- **Architecture:** GPT-2 decoder, single auto-regressive sequence (belief + DB + action + response)
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 13.6 / 82.3 / 72.4 / Combined 90.9
- **Link:** https://arxiv.org/abs/2005.05298

### DoTS (Jeon & Lee, 2021) 
- **What:** End-to-end TOD with a learnable gate that decides which slots matter for each reply so when writing a response, the model focuses on 
    the relevant slots (e.g. name + area for an address question) and ignores the unrelated ones
- **Architecture:** GPT-2 decoder + slot-gating module
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 16.8 / 80.4 / 68.7 / Combined 91.4
- **Link:** https://arxiv.org/abs/2103.06648

### UBAR (Yang et al., 2020) 
- **What:** Fine-tunes GPT-2 on the entire dialogue from start to finish (every turn together) instead of training on each user-bot turn separately, 
    so the model sees the full conversation context (what was said earlier, what the user already booked) when writing each reply
- **Architecture:** GPT-2 decoder, full-session sequences
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 17.6 / 83.4 / 70.3 / Combined 94.4
- **Link:** https://arxiv.org/abs/2012.03539

### PPTOD (Su et al., 2021) 
- **What:** A single T5 model trained on a wide mix of TOD datasets to handle all four pipeline sub-tasks at once: understanding the user (NLU), tracking slots (DST), 
    picking the next action (policy), and writing the reply (NLG), so one model can fill any role in a TOD system
- **Architecture:** T5 encoder–decoder with task-prefix tokens
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 18.2 / 83.1 / 72.7 / Combined 96.1
- **Link:** https://arxiv.org/abs/2109.14739

### RSTOD (Cholakov & Kolev, 2022) 
- **What:** A smaller, cheaper T5-based end-to-end TOD model (T5-small / T5-base instead of T5-large) trained with extra regularization 
    tricks (dropout, label smoothing, weight decay) to prevent overfitting. It proves you don't need a big model to get competitive TOD scores
- **Architecture:** T5-small/base encoder–decoder + regularization losses
- **Approach:** pre-trained+fine-tuned · **Focus:** Tomiinek-only
- **Score:** BLEU 18.0 / 83.5 / 75.0 / Combined 97.3
- **Link:** https://arxiv.org/abs/2208.07097

### BORT (Sun et al., 2022) 
- **What:** A BART-based end-to-end TOD model where pre-training combines two tricks: corrupting sentences and asking the model to reconstruct them (denoising), 
    plus translating responses to another language and back to teach the model that paraphrases mean the same thing, so it handles phrasing variation robustly
- **Architecture:** BART encoder–decoder + denoising pre-training objective
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 17.9 / 85.5 / 77.4 / Combined 99.4
- **Link:** https://arxiv.org/abs/2205.02471

### MTTOD (Lee et al., 2021) 
- **What:** A T5-based end-to-end TOD model with an extra training task: while generating the belief state and response, the model must also point at the exact words 
    in the user message where each slot value came from. This forces it to ground its outputs in what the user actually said and reduces hallucinated values
- **Architecture:** T5 encoder–decoder + auxiliary span head
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 19.0 / 85.9 / 76.5 / Combined 100.2
- **Link:** https://aclanthology.org/2021.findings-emnlp.112/

### GALAXY (He et al., 2021) 
- **What:** Pre-trains UniLM with a side task: predict the purpose of each bot sentence (request, inform, confirm, recommend...). 
    Across many TOD datasets, so the model learns "what bots typically say and why" before being fine-tuned on MultiWOZ
- **Architecture:** UniLM unified encoder–decoder + dialog-act head
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 19.6 / 85.4 / 75.7 / Combined 100.2
- **Link:** https://arxiv.org/abs/2111.14592

### SPACE-3 (He et al., 2022) 
- **What:** A successor to GALAXY by the same authors. It pre-trains a single shared backbone on multiple TOD tasks at once (belief state + dialog acts), 
    with separate small layers ("heads") on top specializing in each task, before fine-tuning on MultiWOZ
- **Architecture:** UniLM-style backbone + state head + policy head
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** JGA 57.5 on 2.2
- **Link:** https://arxiv.org/abs/2209.06664

### RewardNet (Feng et al., 2023) 
- **What:** Fine-tunes a TOD model with reinforcement learning, where a separate small "critic" network learns to score how good each generated response is 
    (correct, helpful, on-task) and that score is used as the reward signal, instead of relying on handwritten reward rules
- **Architecture:** Pre-trained TOD backbone (T5/UniLM) + reward-network critic
- **Approach:** pre-trained+fine-tuned (RL) · **Focus:** Tomiinek-only
- **Score:** BLEU 17.6 / 87.6 / 81.5 / Combined 102.2
- **Link:** https://arxiv.org/abs/2302.10342

### Mars (Sun et al., 2022) 
- **What:** A TOD model where each bot action is given a richer numeric "meaning representation," and training explicitly pushes correct and wrong responses apart, 
    so the model picks more accurate, on-task actions and drifts less
- **Architecture:** Pre-trained backbone + contrastive objective over actions
- **Approach:** pre-trained+fine-tuned · **Focus:** Tomiinek-only
- **Score:** BLEU 19.9 / 88.9 / 78.0 / Combined 103.4
- **Link:** https://arxiv.org/abs/2210.08917

### KRLS (Yu et al., 2022)
- **What:** A TOD response generator trained with reinforcement learning that identifies the load-bearing keywords (slot values, entity names, reference codes) and 
    gives the model word-by-word rewards while generating. Bigger rewards/penalties on keywords than filler, so the model prioritizes getting the critical content right
- **Architecture:** Pre-trained backbone + keyword-RL fine-tuning
- **Approach:** pre-trained+fine-tuned (RL) · **Focus:** Both
- **Score:** BLEU 19.0 / 89.2 / 80.3 / Combined 103.8
- **Link:** https://arxiv.org/abs/2211.16773

### DiactTOD (Wu et al., 2023)
- **What:** An end-to-end TOD model where the response decoder is explicitly told which dialog act to perform (request, inform, confirm, recommend) before writing 
    the reply, so the bot's action plan is a controllable input, leading to more focused on-task responses
- **Architecture:** Pre-trained TOD backbone + dialog-act controller
- **Approach:** pre-trained+fine-tuned · **Focus:** Both
- **Score:** BLEU 17.5 / 89.5 / 84.2 / Combined 104.4
- **Link:** https://arxiv.org/abs/2308.00878

See also TOATOD in section F.

---

## C) LLM in-context learning for DST

### ChatGPT-DST (Heck et al., 2023)
- **What:** Hand-crafted natural-language prompts ask GPT-3.5 to extract the dialogue state by making one separate call per slot per turn (~30 calls per turn). 
    Each prompt focuses on a single slot like "did the user mention hotel area? if yes, what?". Slow and expensive but more accurate per slot than asking for all slots at once
- **Relation to SDP-DST:** Same prompting recipe (slot description in the prompt, one call per slot), but different paradigm: SDP-DST fine-tunes a T5 encoder-decoder 
    on MultiWOZ training data, while ChatGPT-DST uses a frozen instruction-tuned decoder LLM with no training at all. 
    ChatGPT-DST shows the SDP-DST prompting idea transfers to the LLM era without retraining, at the cost of substantially lower JGA
- **Architecture:** Frozen GPT-3.5 + hand-engineered prompts
- **Approach:** zero-shot prompting · **Focus:** DST-only
- **Score:** per-domain Avg JGA 56.4 on 2.1/2.2 (Hotel 42.0, Restaurant 55.8), multi-domain JGA ≈31.5 (vs SDP-DST: JGA 57.6 on 2.2 multi-domain)
- **Link:** https://aclanthology.org/2023.acl-short.81/

---

## D) LLM zero-shot end-to-end TOD

### "Are LLMs All You Need for TOD?" (Hudeček & Dušek, 2023), Reference [9] in docs/references.md, *not on official leaderboard*
- **What:** Builds an end-to-end TOD pipeline from prompted LLM modules: domain detection → DST → DB lookup → response, with no fine-tuning. The canonical zero-shot baseline.
    Also tests a few-shot variant where a retriever pulls the most similar training dialogues and injects them as in-context examples into the DST and ResponseGen prompts
- **Architecture:** Modular prompted pipeline of frozen GPT-3.5 calls + DB API
- **Approach:** zero-shot, few-shot prompting · **Focus:** Both
- **Score:** zero-shot JGA ~30, Combined ~50 on 2.2. Few-shot retrieval improves both (DST JGA ~37, Combined ~63) but stays far below fine-tuned SOTA. 
    Conclusion of the paper: even with retrieved examples, zero-/few-shot LLMs cannot match fine-tuned TOD systems on MultiWOZ
- **Link:** https://aclanthology.org/2023.sigdial-1.21/ 

### SGP-TOD (Zhang et al., 2023), Reference [10] in docs/references.md
- **What:** One LLM and two prompted modules (DST Prompter + Policy Prompter) using a handwritten task schema
- **Architecture:** Frozen GPT-3.5 + schema-grounded DST and policy prompters
- **Approach:** zero-shot prompting · **Focus:** Both
- **Score:** Inform 82.0 / Success 72.5 / BLEU 9.2 / Combined 86.4 on 2.2 (zero-shot SOTA at the time)
- **Link:** https://aclanthology.org/2023.findings-emnlp.891/

---

## E) Multi-agent / agentic LLM frameworks

### DARD (Aggarwal et al., 2024), Reference [13] in docs/references.md, *not on official leaderboard*
- **What:** A central manager LLM dispatches each turn to one of 5 domain-specific sub-agents
- **Architecture:** 1 manager LLM + 5 domain agents (some fine-tuned, some prompted)
- **Approach:** multi-agent (mix of fine-tuned + prompted) · **Focus:** Both
- **Score:** Inform 97.2 / Success 91.7 (Claude) on 2.2; JGA 63.6 (Flan-T5 multi-agent vs 58.9 single)
- **Link:** https://arxiv.org/abs/2411.00427

### DIMF (Wang et al., 2025), Reference [15] in docs/references.md, *not on official leaderboard*
- **What:** A Domain-Independent Multi-agent Framework with separate Intent, Slot-Filling, Response agents trained via DPO
- **Architecture:** Three role-specialized LLM agents + DPO fine-tuning
- **Approach:** fine-tuned multi-agent (DPO) · **Focus:** Both
- **Score:** beats GALAXY, TOATOD, Mars, SGP-TOD, DARD on 2.2 average
- **Link:** https://arxiv.org/html/2505.14299

### clem:todd (Kranti et al., 2025), Reference [16] in docs/references.md, *not on official leaderboard*
- **What:** A self-play benchmark that compares Monolithic, Modular-Programmatic, Modular-LLM TOD architectures
- **Architecture:** Not a model itself but a benchmark suite that runs the same TOD task through three different architectures (monolithic single-LLM, 
    modular pipeline with handwritten code, modular pipeline with multiple LLM modules), each tested with several LLM backbones
- **Approach:** benchmark · **Focus:** Both
- **Score:** Modular-LLM (Qwen2.5-32B) Inform 0.68 vs Modular-Programmatic 0.41 on 2.2
- **Link:** https://arxiv.org/abs/2505.05445

---

## F) LoRA / PEFT fine-tuned LLMs for TOD

### LDST (Feng et al., 2023), *not on official leaderboard*
- **What:** Fine-tunes LLaMA-7B with LoRA using assembled domain-slot instruction tuning
- **Architecture:** LLaMA-7B + LoRA on q/k/v/o projections
- **Approach:** LoRA-tuned · **Focus:** DST-only
- **Score:** JGA 60.7 on 2.2 (multi-domain)
- **Link:** https://aclanthology.org/2023.emnlp-main.48/

### Spec-TOD (Nguyen et al., 2025), Reference [20] in docs/references.md, *not on official leaderboard*
- **What:** Fine-tunes LLaMA-3-8B with LoRA + explicit task instructions for end-to-end TOD
- **Architecture:** LLaMA-3-8B + LoRA on q_proj/v_proj only
- **Approach:** LoRA-tuned · **Focus:** Both
- **Score:** Inform 87.2 / Success 77.1 / Combined 92.6 on 2.2 (10% data)
- **Link:** https://arxiv.org/abs/2507.04841

### TOATOD (Bang et al., 2023) 
- **What:** Takes a pre-trained T5, freezes it, and attaches small "adapter" layers on top: one per sub-task (NLU, DST, NLG). Only the adapters are trained, 
    much cheaper than fine-tuning the whole T5. After supervised adapter training, applies reinforcement learning on top to push the scores further. 
    Same parameter-efficient recipe as Exp3 LoRA, but with adapters instead of LoRA, T5 instead of LLaMA/Qwen, and an extra RL polish on top
- **Architecture:** Frozen T5 + small adapter modules per sub-task (NLU/DST/NLG) + RL
- **Approach:** pre-trained adapter fine-tuned (PEFT) · **Focus:** Both
- **Score:** JGA 63.8 on 2.2, Combined 101.9 on 2.2 
- **Link:** https://aclanthology.org/2023.findings-acl.464/

### Confidence-LLM-DST (Sun et al., 2024), *not on official leaderboard*
- **What:** Takes Llama-3-8B and fine-tunes it with LoRA for DST. Adds one extra trick: a "confidence head", a small layer on top that outputs a probability score for 
    each slot prediction (e.g., "I'm 95% sure restaurant-area=north, but only 40% sure on hotel-stars"). Useful in production because the system can flag 
    low-confidence slots for follow-up questions instead of guessing wrong silently
- **Architecture:** Llama-3-8B + LoRA + small confidence-score layer per slot
- **Approach:** LoRA-tuned · **Focus:** DST-only
- **Score:** JGA 44.6 on 2.2 (multi-domain)
- **Link:** https://arxiv.org/abs/2409.09629

### LUAS (Wang et al., 2024), *not on official leaderboard*
- **What:** Tackles DST training from a different angle: instead of changing the model architecture, expand the training data. Uses GPT-4 as a fake "user simulator" 
    to generate thousands of synthetic user-agent conversations on top of MultiWOZ's real ones. Then full-fine-tunes LLaMA-2-7B (no LoRA, all weights updated) on the 
    combined real + synthetic dataset. The synthetic dialogues fill gaps in coverage that the real MultiWOZ training set has
- **Architecture:** LLaMA-2-7B (fully fine-tuned) + GPT-4 used offline as a user simulator to generate extra training dialogues
- **Approach:** full fine-tuning + LLM data augmentation · **Focus:** DST-only
- **Score:** +4.3 JGA on 2.2 over the baseline trained on real MultiWOZ data alone
- **Link:** https://aclanthology.org/2024.acl-long.473/

---

## MLP4CS: Modular LLM Pipeline for Customer Service
- **What:** Three controlled experiments comparing a single-LLM TOD agent vs a 2-module LLM pipeline (DST + ResponseGen), and measuring whether LoRA fine-tuning each 
    module on its specific role gives a further accuracy boost on top of the modular split
- **Architecture:**
  - **Exp1:** one frozen LLM handles everything (DST, DB lookup, response) in a single prompt, same philosophy as SimpleTOD (Reference [3] docs/references.md) but zero-shot, no fine-tuning
  - **Exp2:** two specialized prompted modules, a DST module that extracts slot-value pairs and a ResponseGen module that produces the user-facing reply, with a deterministic DB lookup in between
  - **Exp3:** the same 2-module pipeline as Exp2, but each module is now LoRA-fine-tuned (Unsloth / QLoRA / bnb-4bit)
- **Approach:**
  - Exp1: zero-shot single-LLM 
  - Exp2: zero-shot modular LLM pipeline
  - Exp3: LoRA-tuned modular pipeline
- **Focus:** Both (all metrics): JGA, Slot F1 and Inform, Success, BLEU, Combined (standardized Tomiinek evaluator)

---

## Summary of MLP4CS implementation
MLP4CS asks two questions: **for task-oriented dialogue with modern LLMs, does splitting a single LLM into two role-specialized modules (DST + ResponseGen) help? Also,
does LoRA fine-tuning on top of that split help further?** To answer, the same MultiWOZ 2.2 evaluation process (Tomiinek `mwzeval`, hotel + restaurant domains only) 
is run three times: once with a single LLM doing everything (Exp1), once with two prompted LLM modules + a deterministic DB lookup in between (Exp2), and
once with that same two-module pipeline LoRA-fine-tuned per role (Exp3). All three experiments report the full metric set of DST (JGA, Slot F1) and end-to-end (Inform, Success, BLEU, Combined). 
All experiments run on a single A100 (EuroHPC Leonardo) and uses Unsloth for QLoRA tuning.

MLP4CS sits at the intersection of three live trends visible in the leaderboard above: 
(1) the move from fine-tuned single models (GALAXY, DiactTOD) to **zero-shot LLM pipelines** (Hudeček & Dušek, SGP-TOD), 
(2) the move from monolithic agents to **modular / multi-agent LLM systems** (DARD, DIMF, clem:todd), and 
(3) the rise of **PEFT for TOD** (parameter-efficient fine-tuning for TOD), methods that train only a small fraction of a model's weights instead of all of them, 
like LoRA (LDST, Spec-TOD) and adapters (TOATOD).

The project is promising because it isolates the effect of each axis: Exp1 vs Exp2 isolates "single LLM vs modular LLM" with everything else held constant, 
and Exp2 vs Exp3 isolates "zero-shot vs LoRA-fine-tuned" with the architecture held constant. 
That is a cleaner controlled comparison than what most leaderboard papers report, which typically vary backbone, data, and architecture all at once. 
If Exp2 beats Exp1 on Combined Score, MLP4CS provides the first controlled evidence on MultiWOZ that role-specialization (not domain-specialization, like DARD) is 
enough to lift modern LLM TOD performance. If Exp3 beats Exp2, it also gives the first controlled estimate of the marginal value of LoRA on top of the pipeline. 
Either result places MLP4CS on the map between Hudeček & Dušek (the zero-shot baseline) and Spec-TOD / TOATOD (the PEFT upper bound).

---
