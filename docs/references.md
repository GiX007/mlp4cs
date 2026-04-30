# References cheatsheet of MLP4CS

A short memory aid for each entry in the formal References list.

---

[1] Budzianowski et al., MultiWOZ (2018) 
- Original MultiWOZ dataset. Introduces the Inform / Success / BLEU evaluation protocol still used today
- The foundational dataset this project is evaluated on

[2] Zang et al., MultiWOZ 2.2 (2020)
- Corrected version of MultiWOZ that fixed annotation errors and added schema-guided slot definitions
- The exact dataset version the project use (hotel + restaurant only)

[3] Hosseini-Asl et al., SimpleTOD (2020)
- A pretrained GPT-2 (stage-1 only) fine-tuned end-to-end on MultiWOZ. One model handles belief + action + response in a single autoregressive sequence
- Historical anchor for Exp1 design philosophy ("one LLM does everything"). The canonical pre-LLM baseline

[4] Brown et al., GPT-3 / "Language Models are Few-Shot Learners" (2020)
- 175B-parameter LLM that introduced and validated zero-shot and few-shot prompting as a paradigm, solving tasks from natural-language instructions without gradient updates
- Foundational citation for zero-shot experiments (Exp1, Exp2)

[5] Ouyang et al., InstructGPT (2022)
- GPT-3 fine-tuned with RLHF (Reinforcement Learning from Human Feedback) to follow user instructions. Established the methodology behind ChatGPT, Claude, and modern instruct models
- Bridges GPT-3 (base completion/stage-2 of pretraining) to actual models (instruction-tuned APIs). Explains why GPT-4o-mini and Claude Haiku reliably follow our prompts

[6] Touvron et al., LLaMA (2023)
- Meta's family of openly-released foundation LLMs, trained on public data, that kicked off the open-source LLM revolution
- Foundation paper for the LLaMA model family used as a fine-tuning backbone in Exp3

[7] Yang et al., Qwen2.5 Technical Report (2024)
- Alibaba's Qwen2.5 family, pretrained on 18T tokens with explicit optimization for structured outputs including JSON
- Justifies our choice of Qwen for TOD as its JSON-output reliability matters for DST belief-state generation

[8] Yang et al., Qwen3 Technical Report (2025)
- Alibaba's Qwen3 family with integrated "thinking" and "non-thinking" modes in a single model
- Covers Exp1 lead model (Qwen3-14B with thinking mode), which gives this project its novelty claim: first reported TOD evaluation of Qwen3 thinking mode

[9] Hudecek & Dušek, "Are LLMs All You Need for TOD?" (2023)
- Tests two zero-shot GPT-3.5 architectures on MultiWOZ: a single-prompt setup (one LLM does everything) and a modular pipeline (domain detection → DST → DB lookup → response). 
  Both setups are zero-shot. The paper also tests a few-shot variant where a retriever pulls the most similar training dialogues and injects them as in-context examples into the DST and ResponseGen prompts, no fine-tuning anywhere. The canonical zero-shot LLM baseline
- Direct counterpart for both Exp1 (single-LLM) and Exp2 (modular pipeline). Same architectural philosophies, but we replicate them across modern open + closed LLMs (2024–2026 instead of 2023 GPT-3.5)

[10] Zhang et al., SGP-TOD (2023)
- One frozen LLM (GPT-3.5) driven by two specialized prompts in sequence: a DST Prompter that includes the slot vocabulary and valid values, and a Policy Prompter 
  that includes a handwritten flowchart of how the bot should react to each belief state. With these two schemas as instructions, 
  the same LLM acts as DST and as policy/response generator. Best published zero-shot end-to-end result on MWOZ 2.2 (Combined 86.4)
- Strongest zero-shot baseline for Exp1/Exp2 to be measured against 

[11] Hu et al., IC-DST (2022)
- Treats DST as a SQL-writing task. The LLM (Codex, a code-trained GPT-3) is given three things in one prompt: a list of every possible slot (as a CREATE TABLE schema), 
  2-5 similar past dialogues with their correct SQL answers, and the new turn (which is the previous belief state plus the latest user/system exchange, not the full earlier 
  conversation). The LLM writes a SQL query capturing the slot changes in this turn, and a small parser converts that SQL back into slot-value pairs
- First paper that did zero-shot/few-shot DST on MultiWOZ with prompting alone, direct ancestor of Exp2 DST module

[12] Xu et al., AutoTOD (2024)
- A single autonomous LLM agent. Instead of pipeline modules, the LLM gets one fat instruction prompt with three blocks: the scenario ("you are a Cambridge travel assistant"), 
  the available APIs described in plain text (one per domain, with their parameters and what they return), and a strict output format requiring either "Thought + API call" or 
  "Thought + Response" at every turn. The "APIs" are not real services. They are just text descriptions in the prompt. 
  The LLM agrees to follow the format and writes one or the other at each round, while a small wrapper outside the LLM runs the actual database query whenever it sees an API call. 
  The LLM decides at every round whether to fetch data or reply to the user
- Same architecture philosophy as Exp1 (one LLM does everything, zero-shot)

[13] Gupta et al., DARD (2024)
- A central manager LLM dispatches each turn to one of 5 domain-specific sub-agents (one per MultiWOZ domain)
- Contrasting design to our 2-module pipeline. DARD splits by domain (hotel/restaurant/train/...) while MLP4CS splits by role (DST/ResponseGen). 
  Cited to position MLP4CS's role-specialization as a deliberate alternative to DARD's domain-specialization

[14] Baidya et al., The Behavior Gap (2025)
- Compares zero-shot LLM agents to human reference dialogues on MultiWOZ. Finds that LLMs reach the right answer through unnatural conversation patterns 
  (more turns, more clarifying questions, longer responses), patterns invisible to standard Inform/Success/BLEU metrics. Introduces behavior-level metrics that expose this gap
- Honest disclaimer for our zero-shot Exp1 and Exp2 numbers. A high Combined Score does not guarantee the bot reached it through human-like dialogue, and 
  this paper is the citation that lets us acknowledge this limitation

[15] Feng et al., DIMF / Empowering LLMs in TOD (2025)
- A Domain-Independent Multi-agent Framework with separate Intent, Slot-Filling and Response agents trained via DPO:
  Direct Preference Optimization: a training method where each example has two labels for the same input: a "preferred" output (the one the model should learn to 
  produce) and a "rejected" output (the one the model should learn to avoid). Functions as a simpler alternative to RLHF
- Closest contemporary to Exp3 in spirit: multi-agent + role-specialization + fine-tuning. Confirms the value of the role-specialization split that this project tests directly

[16] Kranti et al., clem:todd (2025)
- Defines a vocabulary for LLM-based TOD architectures: Monolithic (one LLM does everything in one prompt), Modular-Programmatic (an LLM is treated as a 
  text-generator that handwritten Python code calls in a fixed sequence, so code makes all the decisions), and Modular-LLM (multiple LLM modules, each responsible 
  for one role, making decisions in their own domain)
- Provides the exact vocabulary MLP4CS uses (Exp1 = Monolithic, Exp2/Exp3 = Modular-LLM) 

[17] Wei et al., Chain-of-Thought Prompting (2022)
- Showed that prompting LLMs to "think step by step" before answering substantially improves accuracy on reasoning-heavy tasks
- The prompting technique embedded in the zero-shot prompt of dst (Exp2, Exp3) and supported natively by Qwen3-14B-thinking

[18] Hu et al., LoRA (2021)
- Low-Rank Adaptation: adding small trainable low-rank matrices to a frozen pretrained model so only ~1% of parameters are tuned during fine-tuning
- Potential parameter-efficient fine-tuning method used in Exp3

[19] Dettmers et al., QLoRA (2023)
- LoRA combined with 4-bit quantization of the base model, making it possible to fine-tune large LLMs on a single consumer/academic GPU
- The exact PEFT method we run in Exp3 (via Unsloth's QLoRA / bnb-4bit)

[20] Nguyen et al., Spec-TOD (2025)
- LLaMA-3-8B fine-tuned with LoRA on q_proj/v_proj for end-to-end TOD on MWOZ 2.2. Reaches Combined 92.6 with only 10% training data.
  The key contribution is hand-crafted, task-specific instructions added to the prompts during fine-tuning
- The closest published cousin of Exp3 (same dataset, same metrics, similar PEFT method, similar backbone size, similarly detailed task-specific prompts), 
  but with two key differences that make it an upper-bound reference rather than a direct competitor: it covers all 7 MWOZ domains (we cover only hotel + restaurant), 
  and it trains one end-to-end model whereas we train two separate LoRA adapters per role

[21] Deriu et al., Survey on Evaluation Methods for Dialogue Systems (2021)
- Standard reference survey covering evaluation methodologies for both task-oriented and open-domain dialogue systems
- Background reference for project's Evaluation as it grounds all metric choices in the broader evaluation literature

[22] Nekvinda & Dušek, "Shades of BLEU, Flavors of Success", (2021)
- Identified inconsistencies in how prior MultiWOZ work computed Inform / Success / BLEU. Released the standardized `mwzeval` evaluator everyone now uses
- We use their `mwzeval` package directly to compute all our metrics

---
