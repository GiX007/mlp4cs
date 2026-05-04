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

---

## Models

*API baselines represent the most cost-effective commercial options available at the time of evaluation (February 2026). The goal is to maximize performance under realistic budget constraints, as would be the case in a production customer service deployment.*

| Model                 | Provider  | Release  | Params      | Quantization | Context | Training Data | Open Source      | Cost (in/out per 1K tokens) |
|-----------------------|-----------|----------|-------------|--------------|---------|---------------|------------------|-----------------------------|
| GPT-4o-mini           | OpenAI    | Jul 2024 | Undisclosed | N/A          | 128K    | Undisclosed   | No               | $0.15 / $0.60               |
| Claude 3 Haiku        | Anthropic | Oct 2025 | Undisclosed | N/A          | 200K    | Undisclosed   | No               | $0.25 / $1.25               |
| Llama-3.2-3B-Instruct | Meta      | Sep 2024 | 3.2B        | 4-bit (bnb)  | 128K    | 9T+ tokens    | Yes (Llama 3.2)  | Free                        |
| Llama-3.1-8B-Instruct | Meta      | Jul 2024 | 8.0B        | 4-bit (bnb)  | 128K    | 15T tokens    | Yes (Llama 3.1)  | Free                        |
| Qwen2.5-14B-Instruct  | Alibaba   | Sep 2024 | 14.7B       | 4-bit (bnb)  | 128K    | 18T tokens    | Yes (Apache 2.0) | Free                        |
| Qwen3-4B              | Alibaba   | Apr 2025 | 4.0B        | 4-bit (bnb)  | 32K     | 36T tokens    | Yes (Apache 2.0) | Free                        |
| Qwen3-8B              | Alibaba   | Apr 2025 | 8.2B        | 4-bit (bnb)  | 32K     | 36T tokens    | Yes (Apache 2.0) | Free                        |
| Qwen3-14B             | Alibaba   | Apr 2025 | 14.8B       | 4-bit (bnb)  | 32K     | 36T tokens    | Yes (Apache 2.0) | Free                        |
| Phi-4-14B             | Microsoft | Dec 2024 | 14B         | 4-bit (bnb)  | 16K     | 9.8T tokens   | Yes (MIT)        | Free                        |

---

## Results

*Results on MultiWOZ 2.2 test set (hotel + restaurant domains only). Tomiinek metrics cover 2 of 5 leaderboard domains so they are not directly comparable to official MultiWOZ leaderboard scores.*

### Experiment 1: Single-LLM Baseline


| Config     | JGA% | SlotF1% | Inform% | Success% | BLEU | Combined |
|------------|------|---------|---------|----------|------|----------|
| gpt        | 24.8 | 71.2    | 58.6    | 44.1     | 3.08 | 54.43    | 
| gpt-nano   | 31.4 | 78.7    | 33.9    | 19.4     | 3.96 | 30.61    | 
| haiku      | 44.8 | 88.3    | 90.9    | 83.3     | 3.67 | 90.77    |
| qwen3_4b   | 14.8 | 54.7    | 50.5    | 39.8     | 2.59 | 47.74    |
| qwen3_8b   | 32.3 | 82.1    | 43.0    | 31.2     | 4.48 | 41.58    | 
| qwen25_14b | 26.4 | 74.9    | 74.2    | 54.8     | 3.37 | 67.87    | 
| qwen3_14b  | 33.3 | 83.0    | 82.3    | 72.0     | 3.69 | 80.84    |
| llama31_8b | 5.2  | 33.1    | 14.0    | 2.20     | 2.30 | 10.40    | 
| phi4_14b   | 25.8 | 74.3    | 66.1    | 59.7     | 4.46 | 67.36    |

Per-Domain Breakdown

| Config     | Domain     | JGA% | SlotF1% | 
|------------|------------|------|---------|
| gpt        | hotel      | 17.9 | 68.1    | 
| gpt        | restaurant | 33.5 | 75.3    | 
| gpt-nano   | hotel      | 21.7 | 73.1    | 
| gpt-nano   | restaurant | 44.4 | 86.1    | 
| haiku      | hotel      | 37.2 | 86.5    | 
| haiku      | restaurant | 55.5 | 91.0    |
| qwen3_4b   | hotel      | 13.8 | 56.3    |
| qwen3_4b   | restaurant | 16.4 | 52.5    |
| qwen3_8b   | hotel      | 23.0 | 78.6    | 
| qwen3_8b   | restaurant | 45.1 | 86.9    | 
| qwen25_14b | hotel      | 20.5 | 71.4    | 
| qwen25_14b | restaurant | 34.4 | 79.5    | 
| qwen3_14b  | hotel      | 24.4 | 79.7    | 
| qwen3_14b  | restaurant | 45.2 | 87.8    | 
| llama31_8b | hotel      | 4.2  | 40.5    | 
| llama31_8b | restaurant | 6.6  | 23.5    |
| phi4_14b   | hotel      | 17.8 | 70.0    |
| phi4_14b   | restaurant | 36.0 | 80.1    |


---

### Experiment 2: Modular Zero-Shot Pipeline

| Config                    | JGA% | SlotF1% | Inform% | Success% | BLEU | Combined |
|---------------------------|------|---------|---------|----------|------|----------|
| homo_gpt                  | 35.7 | 81.7    | 79.0    | 72.0     | 3.14 | 78.64    |
| homo_haiku                | 45.1 | 87.3    | 90.3    | 83.9     | 4.25 | 91.35    |
| hetero_gpt_haiku          | 34.7 | 81.9    | 78.0    | 72.6     | 3.92 | 79.22    |
| hetero_haiku_gpt          | 44.6 | 87.0    | 88.2    | 81.2     | 2.95 | 87.65    |
| homo_qwen3_4b             | 26.7 | 77.2    | 68.3    | 59.7     | 3.11 | 67.11    |
| homo_qwen3_14b            | 33.4 | 80.4    | 73.7    | 68.3     | 5.00 | 76.00    |
| hetero_qwen25_qwen3_14b   | 30.1 | 76.9    | 73.7    | 60.8     | 4.54 | 71.79    |
| homo_qwen3_8b             | 34.2 | 81.3    | 51.6    | 42.5     | 5.10 | 52.15    |
| hetero_qwen3_14b_qwen3_8b | 33.7 | 81.6    | 52.7    | 45.2     | 5.20 | 54.15    |
| homo_llama32_3b           | 5.3  | 42.5    | 28.50   | 14.50    | 1.26 | 22.76    |
| homo_llama31_8b           | 18.1 | 63.4    | 55.4    | 35.5     | 1.53 | 46.98    |
| homo_phi4_14b             | 23.8 | 69.7    | 61.3    | 43.5     | 2.83 | 55.23    |
| hetero_qwen3_14b_phi4_14b | 33.7 | 81.1    | 68.3    | 62.9     | 2.91 | 68.51    |


Per-Domain Breakdown

| Config                    | Domain     | JGA% | SlotF1% |
|---------------------------|------------|------|---------|
| homo_gpt                  | hotel      | 25.7 | 78.3    |
| homo_gpt                  | restaurant | 48.8 | 86.2    |
| homo_haiku                | hotel      | 35.3 | 84.2    |
| homo_haiku                | restaurant | 58.7 | 91.9    |
| hetero_gpt_haiku          | hotel      | 25.0 | 79.0    |
| hetero_gpt_haiku          | restaurant | 47.3 | 85.6    |
| hetero_haiku_gpt          | hotel      | 35.6 | 84.3    |
| hetero_haiku_gpt          | restaurant | 58.0 | 91.3    |
| homo_qwen3_4b             | hotel      | 22.1 | 75.8    |
| homo_qwen3_4b             | restaurant | 32.6 | 79.1    |
| homo_qwen3_14b            | hotel      | 25.3 | 77.2    |
| homo_qwen3_14b            | restaurant | 44.6 | 84.9    |
| hetero_qwen25_qwen3_14b   | hotel      | 18.6 | 70.3    |
| hetero_qwen25_qwen3_14b   | restaurant | 44.6 | 85.2    |
| homo_qwen3_8b             | hotel      | 25.7 | 78.0    |
| homo_qwen3_8b             | restaurant | 45.3 | 85.7    |
| hetero_qwen3_14b_qwen3_8b | hotel      | 25.2 | 78.5    |
| hetero_qwen3_14b_qwen3_8b | restaurant | 44.9 | 86.1    |
| homo_llama32_3b           | hotel      | 6.1  | 43.7    |
| homo_llama32_3b           | restaurant | 4.7  | 41.7    |
| homo_llama31_8b           | hotel      | 13.7 | 62.4    |
| homo_llama31_8b           | restaurant | 23.5 | 64.6    |
| homo_phi4_14b             | hotel      | 16.8 | 65.0    |
| homo_phi4_14b             | restaurant | 33.6 | 75.9    |
| hetero_qwen3_14b_phi4_14b | hotel      | 27.2 | 77.9    |
| hetero_qwen3_14b_phi4_14b | restaurant | 42.3 | 85.5    |

---

### Experiment 3: Modular Fine-Tuned Pipeline

| Config                       | JGA% | SlotF1% | Inform% | Success% | BLEU  | Combined |
|------------------------------|------|---------|---------|----------|-------|----------|
| ft_homo_qwen3_4b             | 44.8 | 88.3    | 54.8    | 36.0     | 8.35  | 53.75    |
| ft_homo_qwen3_8b             | 47.3 | 89.1    | 53.2    | 48.4     | 11.35 | 62.15    |
| ft_homo_qwen3_14b            | 47.7 | 89.5    | 61.3    | 55.4     | 12.26 | 70.61    |
| ft_homo_llama32_3b           | 43.5 | 87.8    | 57.0    | 39.8     | 9.39  | 57.79    |
| ft_homo_llama31_8b           | 44.0 | 87.1    | 50.5    | 40.3     | 9.22  | 54.62    |
| ft_homo_phi4_14b             | 47.0 | 88.9    | 66.7    | 57.5     | 8.74  | 70.84    |
| ft_hetero_qwen3_14b_phi4_14b | 46.9 | 89.2    | 66.7    | 57.0     | 8.71  | 70.56    |

Per-Domain Breakdown

| Config                       | Domain     | JGA% | SlotF1% |
|------------------------------|------------|------|---------|
| ft_homo_qwen3_4b             | hotel      | 35.7 | 86.1    |
| ft_homo_qwen3_4b             | restaurant | 56.4 | 91.1    |
| ft_homo_qwen3_8b             | hotel      | 37.5 | 87.2    |
| ft_homo_qwen3_8b             | restaurant | 59.7 | 91.6    |
| ft_homo_qwen3_14b            | hotel      | 40.4 | 88.3    |
| ft_homo_qwen3_14b            | restaurant | 56.8 | 91.1    |
| ft_homo_llama32_3b           | hotel      | 35.7 | 85.6    |
| ft_homo_llama32_3b           | restaurant | 53.0 | 90.3    |
| ft_homo_llama31_8b           | hotel      | 37.7 | 86.0    |
| ft_homo_llama31_8b           | restaurant | 50.9 | 88.6    |
| ft_homo_phi4_14b             | hotel      | 38.8 | 87.5    |
| ft_homo_phi4_14b             | restaurant | 57.5 | 90.6    |
| ft_hetero_qwen3_14b_phi4_14b | hotel      | 39.4 | 88.0    |
| ft_hetero_qwen3_14b_phi4_14b | restaurant | 56.3 | 90.7    |

---

*Proudly the first LLM pipeline to ship from Chatzis, Greece.* 🇬🇷
