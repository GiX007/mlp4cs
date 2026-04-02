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

### Experiment 1: Single-LLM Baseline

| Config | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU | Combined | Cost($) | Latency(s) |
|--------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|------|----------|---------|------------|
| gpt    | 99.1     | 89.0     | 78.5    | 23.2 | 61.5   | 67.9    | 4.8   | 0.9      | 95.8     | 55.2  | 53.8    | 42.1     | 2.80 | 50.75    | $2.6459 | 5.90s      |
| haiku  | 98.5     | 91.7     | 80.4    | 37.5 | 80.4   | 84.0    | 5.7   | 1.6      | 93.6     | 74.8  | 78.4    | 65.5     | 2.95 | 74.90    | $5.0387 | 13.91s     |

Per-Domain Breakdown

| Config | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|--------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| gpt    | hotel      | 98.9     | 87.6     | 80.1    | 17.3 | 61.3   | 69.0    | 4.3   | 1.3      | 95.5     | 52.6  | $1.4486 | 5.88s      |
| gpt    | restaurant | 99.5     | 91.0     | 76.4    | 30.2 | 61.8   | 66.8    | 5.5   | 0.5      | 96.1     | 52.6  | $1.1943 | 5.92s      |
| haiku  | hotel      | 97.9     | 90.0     | 82.6    | 27.9 | 77.4   | 82.6    | 5.5   | 2.3      | 93.0     | 68.1  | $2.7641 | 13.43s     |
| haiku  | restaurant | 99.2     | 93.7     | 77.6    | 49.3 | 84.7   | 86.3    | 6.1   | 0.8      | 94.3     | 79.3  | $2.2447 | 14.59s     |


---

### Experiment 2: Modular Zero-Shot Pipeline

| Config           | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU | Combined | Cost($) | Latency(s) |
|------------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|------|----------|---------|------------|
| homo_gpt         | 98.9     | 91.7     | 82.0    | 32.4 | 76.5   | 78.0    | 2.7   | 2.6      | 95.7     | 62.3  | 71.3    | 64.3     | 3.26 | 71.06    | $0.1936 | 3.37s      |
| homo_haiku       | 98.0     | 88.1     | 80.4    | 30.8 | 75.3   | 77.0    | 4.6   | 2.0      | 95.6     | 65.5  | 64.9    | 56.1     | 2.70 | 63.20    | $0.4179 | 2.36s      |
| hetero_gpt_haiku | 98.8     | 90.2     | 80.5    | 34.0 | 77.2   | 78.6    | 2.1   | 2.7      | 96.4     | 66.5  | 74.3    | 70.2     | 3.00 | 75.25    | $0.3051 | 2.46s      |
| hetero_haiku_gpt | 97.9     | 87.2     | 79.3    | 30.4 | 74.7   | 76.0    | 3.8   | 1.6      | 96.2     | 53.9  | 62.0    | 48.0     | 2.95 | 57.95    | $0.3019 | 2.48s      |

Per-Domain Breakdown

| Config           | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|------------------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| homo_gpt         | hotel      | 98.5     | 89.6     | 84.6    | 19.7 | 70.3   | 73.3    | 4.6   | 2.8      | 94.3     | 50.4  | $0.1100 | 3.43s      |
| homo_gpt         | restaurant | 99.5     | 94.3     | 78.7    | 47.7 | 84.2   | 83.9    | 0.7   | 2.3      | 97.4     | 77.9  | $0.0835 | 3.31s      |
| homo_haiku       | hotel      | 98.9     | 88.1     | 83.3    | 18.9 | 70.1   | 72.8    | 7.2   | 1.5      | 95.2     | 52.1  | $0.2283 | 2.45s      |
| homo_haiku       | restaurant | 97.0     | 88.1     | 76.9    | 44.2 | 81.2   | 81.7    | 2.4   | 2.5      | 96.0     | 79.1  | $0.1896 | 2.27s      |
| hetero_gpt_haiku | hotel      | 98.5     | 88.3     | 82.7    | 19.8 | 70.5   | 73.4    | 3.2   | 1.5      | 97.0     | 56.6  | $0.1738 | 2.51s      |
| hetero_gpt_haiku | restaurant | 99.2     | 92.5     | 77.7    | 51.0 | 85.5   | 85.2    | 1.0   | 4.1      | 95.6     | 73.8  | $0.1312 | 2.40s      |
| hetero_haiku_gpt | hotel      | 98.7     | 84.5     | 80.6    | 18.2 | 68.3   | 71.1    | 5.9   | 2.0      | 94.7     | 49.5  | $0.1645 | 2.56s      |
| hetero_haiku_gpt | restaurant | 97.0     | 90.3     | 77.8    | 44.4 | 81.9   | 81.6    | 1.8   | 1.2      | 97.8     | 65.5  | $0.1374 | 2.39s      |

---

## References

- **Dataset:** [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz) (Zang et al., 2020)
- **Evaluation:** [Tomiinek MultiWOZ Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation) (Nekvinda & Dusek, 2021)
- **Fine-tuning:** [Unsloth](https://github.com/unslothai/unsloth)

---
