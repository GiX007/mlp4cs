# Future Extensions

At a high level, the proposed extensions follow three directions: improving the data, evolving the architecture, and further training.

## 1. Improving the data

MultiWOZ 2.2 contains annotation noise such as wrong or missing slot labels.

Three complementary directions:
- **Clean the existing benchmark.** Audit the hotel and restaurant dialogues of the dev and test splits for annotation errors and fix them, following the approach of 
  MultiWOZ 2.4 (Ye et al. 2022), which corrected the dev and test annotations of 2.1. The train split stays untouched, as in 2.4. Used for evaluation: all experiments are 
  re-run on the cleaned dev and test splits.
- **Collect a small real-dialogue seed.** Gather and anonymize a few hundred real customer service dialogues from a hospitality business, annotate them with the same schema 
  (domain, intent, slots). Use them as a second test set only so the same pipeline is evaluated on both benchmarks to measure how well benchmark scores transfer to real dialogues.
- **Scale the real seed with synthetic data.** Use a strong LLM as a user simulator and agent simulator to generate thousands of synthetic dialogues that  follow the patterns of 
  the real seed (Wang et al. 2024, LUAS, arXiv: 2407.20655). The real dialogues anchor realism, the synthetic ones fill coverage gaps, and the combined set becomes large enough 
  for training. Used as training data only: it replaces the MultiWOZ-derived fine-tuning sets, while evaluation stays on the two benchmarks above.

The cleaned benchmark is for evaluation, the real seed checks if results hold on real dialogues, and the synthetic set is for training.

## 2. Evolving the architecture

The current pipeline is fixed: code decides the steps, and each LLM call is single-shot. The field, in both research and industry, has converged on agentic designs where 
the LLM decides the steps itself.

Two architecture shapes to explore:
- **Single agent with tools.** One LLM gets a set of tools (search_db, book, ask_user, escalate) and the business policy as a plain-language document, 
  then decides its own actions in a loop until it answers the user (Xu et al. 2024, AutoTOD, evaluation via ToolWOZ, arXiv: 2409.04617). 
  The pipeline code disappears: slots become tool arguments, and the policy module becomes text the agent reads.
- **Workflow graph with one flow per task.** A graph defines the allowed flows (book, modify, cancel, inform, escalate) and a small agent operates 
  inside each node. The graph gives control and predictability, the agents give flexibility. This is the design used by production customer service platforms.

Inside either shape, three behaviors make the agent reliable:
- **Iterative search.** On zero DB results, the agent retries with relaxed constraints instead of replying "nothing found", following the 
  Thought-Action-Observation loop of ReAct (Yao et al. 2022, arXiv: 2210.03629).
- **User clarification.** When the request is vague or ambiguous, the agent asks the user before acting, with a budget to avoid over-asking (Acikgoz et al. 2026, MAC, arXiv: 2512.13154).
- **Self-correction.** When a check fails, the agent reflects on why and retries with that reflection in context, replacing the rule-based 
  supervisor (Shinn et al. 2023, Reflexion, arXiv: 2303.11366).

Independent of the shape, the prompts themselves can improve: inject the slot schema and the policy flowchart directly into the prompt 
(Zhang et al. 2023, SGP-TOD), and retrieve similar past dialogues as few-shot examples per turn (Hudecek & Dušek 2023). Both can be tuned automatically with 
prompt optimizers such as DSPy instead of manual editing.

The agent decides the actions, the graph keeps it inside the allowed flows, and the three behaviors plus better prompts make each decision more reliable.

## 3. Further training

The fine-tuning in Experiment 3 stops at supervised LoRA. Two training steps can follow it, both only worthwhile for open-source models, since API models cannot be trained.

- **Preference alignment with DPO.** Take the Experiment 3 adapters, generate several response candidates per turn, let a strong LLM rank them (preferred vs 
  rejected), and train on these pairs with DPO (Rafailov et al. 2023, applied to task-oriented dialogue by Feng et al. 2025, DIMF). DPO is the simple 
  alternative to RLHF: no reward model, no RL loop, just a contrastive loss on pairs. Used to improve response quality and style on top of the existing adapters, 
  slot accuracy is not expected to move.
- **Fine-tune tool calling into small models.** The agentic designs of Section 2 assume the model can call tools reliably, which large API models can 
  successfully do, but small open-source models (around 8B and below) often fail at. Fine-tune them on tool-call examples generated from the dialogues 
  (the same approach as the Experiment 3 data, but with tool calls as targets instead of slot JSON). Used to make the agent architecture work with free local models instead of paid APIs.

Both steps come after the architecture choice, not before: DPO polishes the responses of whatever system exists, and tool-calling fine-tuning only matters 
if the agent design of Section 2 is adopted.

A practical rule for a real deployment: fine-tune only after 1000+ real dialogues are collected, since training earlier overfits on synthetic MultiWOZ data and 
wastes the personalization value of real customer conversations.

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

- Multi-user dialogue
  - Jo et al. 2023, "Multi-User MultiWOZ" arXiv: 2310.20479

    Two-user dialogues with an agent, plus a rewriter that turns the multi-user chat into a single-user query (Figure 1).
    Relevant if the system ever handles voice or group chat, where multiple speakers appear in one conversation

---

## How to Run These Experiments 

All the above extensions are worth trying as ablation experiments to compare architectures for a real customer service product, on both API-based and open-source models.

API models (GPT-4o-mini, Claude Haiku, etc.) need only API keys and budget, no GPU.

Open-source models (LLaMA, Qwen, etc.) of different sizes need a strong GPU for inference and fine-tuning.

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

Prices, credit programs, and GPU options listed above are as of mid-2026.

---
