# MLP4CS: Modular LLM Pipeline for Customer Service

A modular dialogue system for hotel and restaurant customer service, evaluated on MultiWOZ 2.2. The project compares three architectures: 
(1) a single-LLM baseline, (2) a zero-shot modular pipeline with separate DST and response generation components, and (3) a modular pipeline with fine-tuned LoRA adapters. 
The goal is to assess whether architectural decomposition and role-specific fine-tuning improve task completion, slot tracking accuracy, and policy compliance over a single-model approach.

---

## Architecture

**Experiment 1: Single LLM**
```
Dialogue History + DB → [LLM] → Response
```
One model handles everything. Dialogue history and full database go in, response comes out.

**Experiment 2: Zero-Shot Modular Pipeline**
```
Dialogue History → [DST LLM] → Slots → DB Query → [ResponseGen LLM] → Response
```
Two separate modules. DST extracts belief state slots, DB query finds matching entities, ResponseGen produces a grounded response.

Between and around these core modules, the pipeline also runs:
- **Policy:** checks if required booking slots are present before allowing a booking
- **Supervisor:** validates the response and triggers a retry if needed (max 2 attempts)
- **Lexicalizer:** replaces placeholders with real entity values from DB results
- **Memory:** stores the final response in conversation history for subsequent turns

**Experiment 3: Fine-Tuned Modular Pipeline**
```
Dialogue History → [DST LoRA] → Slots → DB Query → [ResponseGen LoRA] → Response
```
Same two-stage pipeline as Exp2, but powered by LoRA fine-tuned adapters instead of general-purpose models.

---

## Experiments

Three experiments isolate the effect of architecture and model specialization:

**Experiment 1: Single LLM Baseline.**
Tested with commercial API models and open-source models. Establishes the performance ceiling for monolithic approaches.

**Experiment 2: Zero-Shot Modular Pipeline.**
Tested in homogeneous (same model for both modules) and heterogeneous (different models) configurations, using both commercial API and open-source models. Tests whether splitting responsibilities improves accuracy without any training.

**Experiment 3: Fine-Tuned Modular Pipeline.**
Each module uses a LoRA fine-tuned open-source adapter (via Unsloth). Tests whether role-specific fine-tuning of small open-source models can compete with larger commercial models.

---

## Evaluation

All experiments are evaluated on the MultiWOZ 2.2 dev set (hotel + restaurant domains) using both custom and official metrics.

**Custom metrics:** Domain Precision, Intent Precision, Action Accuracy, Joint Goal Accuracy (JGA), Slot Recall, Slot F1, Hallucination Rate, Policy Violation Rate, System Correctness, Booking Rate, Latency per turn, Cost per run.

**Official metrics:** Inform Rate, Success Rate, Bleu, Combined Score, computed via the [Tomiinek MultiWOZ evaluator](https://github.com/Tomiinek/MultiWOZ_Evaluation).

---

## Setup
```bash
git clone <repo>
cd mlp4cs

python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
```

### Data
Clone the official MultiWOZ 2.2 dataset (dialogues + databases):
```bash
git clone https://github.com/budzianowski/multiwoz.git data/multiwoz_github
```

### API Keys

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

Required for Experiment 1 and Experiment 2 API-based configurations. Not needed for open-source or fine-tuned models.

---

## How to Run

Experiment configurations are defined in `src/config.py`. To run an experiment, uncomment the desired configuration in `config.py` and the corresponding experiment call in `main.py`, then run:
```bash
python -m src.main
```

Fine-tuning requires a GPU (e.g., Kaggle, Colab, or any CUDA-capable machine). The `scripts/` folder contains info about the fine-tuning process.

Pre-trained base models and fine-tuned adapters are not included in the repository due to size. To run Experiment 3 locally, download the base models and place them in `data/models/`, and place the fine-tuned adapters in `data/finetuned_models/`. See `src/config.py` for expected paths.

Results are saved to `results/` with three files per run: dataset-level, dialogue-level, and turn-level metrics. A summary is appended to `results/leaderboard.txt` along with a per-dialogue error analysis in `results/error_analysis.txt`.

---

## Project Structure

- `src/` - Source code with pipeline modules, experiment runners, evaluation, LLM interface, config
- `scripts/` - Fine-tuning data generation and LoRA training scripts
- `data/` - MultiWOZ 2.2 dataset, base models, fine-tuned adapters (not in repo)
- `docs/` - Documentation and notes
- `archived_results/` - Experiment runs with detailed results

---

## Results

*Results on MultiWOZ 2.2 dev set (hotel + restaurant domains only). Tomiinek metrics cover 2 of 5 leaderboard domains so they are not directly comparable to official MultiWOZ leaderboard scores.*

| Config   |   DomainP% |   IntentP% |   Action% |   JGA% |   SlotR% |   SlotF1% |   Hall% |   PolViol% |   SysCorr% |   Book% |   Inform% |   Success% |   BLEU |   Combined | Cost($)   |
|----------|------------|------------|-----------|--------|----------|-----------|---------|------------|------------|---------|-----------|------------|--------|------------|-----------|
| homo_gpt |       98.9 |       92.3 |        82 |   33.4 |     77.5 |      78.8 |     2.7 |        2.4 |       95.9 |    60.1 |      68.4 |       61.4 |   3.03 |      67.93 | $0.1930   |
| *To be updated after all experiments are complete.* | | | | | | |

---

## References

- **Dataset:** [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz) (Zang et al., 2020)
- **Evaluation:** [Tomiinek MultiWOZ Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation) (Nekvinda & Dusek, 2021)
- **Fine-tuning:** [Unsloth](https://github.com/unslothai/unsloth)

---
