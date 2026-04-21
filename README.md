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
| Qwen2.5-14B-Instruct  | Alibaba   | Sep 2024 | 14.7B       | 4-bit (bnb)  | 128K    | 18T tokens    | Yes (Apache 2.0) | Free                        |
| Qwen3-8B              | Alibaba   | Apr 2025 | 8.2B        | 4-bit (bnb)  | 32K     | 36T tokens    | Yes (Apache 2.0) | Free                        |
| Qwen3-14B             | Alibaba   | Apr 2025 | 14.8B       | 4-bit (bnb)  | 32K     | 36T tokens    | Yes (Apache 2.0) | Free                        |

---

## Results

*Results on MultiWOZ 2.2 test set (hotel + restaurant domains only). Tomiinek metrics cover 2 of 5 leaderboard domains so they are not directly comparable to official MultiWOZ leaderboard scores.*

### Experiment 1: Single-LLM Baseline


| Config     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU | Combined | Cost($) | Latency(s) |
|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|------|----------|---------|------------|
| gpt        | 98.7     | 87.0     | 77.6    | 24.8 | 64.7   | 71.2    | 5.9   | 0.8      | 94.9     | 57.4  | 58.6    | 44.1     | 3.08 | 54.43    | $2.75   | 3.25s      |
| gpt-nano   | 96.9     | 84.1     | 75.3    | 31.4 | 74.6   | 78.7    | 12.4  | 0.3      | 97.2     | 33.8  | 33.9    | 19.4     | 3.96 | 30.61    | $1.80   | 2.10s      |
| haiku      | 98.8     | 90.1     | 82.8    | 44.8 | 84.3   | 88.3    | 4.4   | 3.6      | 92.6     | 76.7  | 90.9    | 83.3     | 3.67 | 90.77    | $21.03  | 3.04s      |
| qwen3_8b   | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —        | —    | —        | —       | —          |
| qwen25_14b | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —        | —    | —        | —       | —          |
| qwen3_14b  | 98.7     | 89.3     | 78.4    | 33.3 | 78.6   | 83.0    | 2.7   | 1.9      | 96.0     | 75.8  | 82.3    | 72.0     | 3.69 | 80.84    | $0.00   | 12.40s     |

Per-Domain Breakdown

| Config     | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|------------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| gpt        | hotel      | 99.2     | 84.5     | 77.0    | 17.9 | 60.7   | 68.1    | 6.0   | 0.6      | 95.2     | 51.9  | $1.53   | 3.23s      |
| gpt        | restaurant | 98.2     | 90.4     | 78.4    | 33.5 | 69.9   | 75.3    | 5.7   | 1.0      | 94.7     | 60.3  | $1.21   | 3.27s      |
| gpt-nano   | hotel      | 96.7     | 81.3     | 74.3    | 21.7 | 67.7   | 73.1    | 22.5  | 0.6      | 96.3     | 31.7  | $1.05   | 2.10s      |
| gpt-nano   | restaurant | 97.1     | 87.7     | 76.7    | 44.4 | 83.7   | 86.1    | 5.6   | 0.0      | 98.4     | 27.0  | $0.79   | 2.09s      |
| haiku      | hotel      | 99.8     | 90.2     | 82.2    | 37.2 | 81.6   | 86.5    | 5.9   | 4.0      | 90.8     | 74.0  | $11.82  | 3.07s      |
| haiku      | restaurant | 99.5     | 91.9     | 83.7    | 55.5 | 88.2   | 91.0    | 1.5   | 3.1      | 95.5     | 76.7  | $8.99   | 3.00s      |
| qwen3_8b   | hotel      | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| qwen3_8b   | restaurant | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| qwen25_14b | hotel      | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| qwen25_14b | restaurant | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| qwen3_14b  | hotel      | 99.8     | 89.7     | 78.3    | 24.4 | 73.9   | 79.7    | 3.0   | 0.0      | 97.4     | 77.5  | $0.00   | 12.56s     |
| qwen3_14b  | restaurant | 98.7     | 90.2     | 78.7    | 45.2 | 85.2   | 87.8    | 2.2   | 4.4      | 94.1     | 68.7  | $0.00   | 12.27s     |

---

### Experiment 2: Modular Zero-Shot Pipeline

| Config                    | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU | Combined | Cost($) | Latency(s) |
|---------------------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|------|----------|---------|------------|
| homo_gpt                  | 99.6     | 90.9     | 80.2    | 35.7 | 80.1   | 81.7    | 3.1   | 3.4      | 94.6     | 65.9  | 79.0    | 72.0     | 3.14 | 78.64    | $0.20   | 4.00s      |
| homo_haiku                | 99.8     | 91.0     | 81.1    | 45.1 | 84.0   | 87.3    | 0.8   | 5.1      | 94.4     | 75.2  | 90.3    | 83.9     | 4.25 | 91.35    | $2.10   | 4.73s      |
| hetero_gpt_haiku          | 99.8     | 91.5     | 80.2    | 34.7 | 80.3   | 81.9    | 1.0   | 4.7      | 94.5     | 69.7  | 78.0    | 72.6     | 3.92 | 79.22    | $0.78   | 3.70s      |
| hetero_haiku_gpt          | 99.8     | 91.2     | 80.7    | 44.6 | 83.8   | 87.0    | 2.1   | 3.7      | 95.4     | 71.4  | 88.2    | 81.2     | 2.95 | 87.65    | $1.53   | 4.86s      |
| homo_qwen3_14b            | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —        | —    | —        | —       | —          |
| hetero_qwen25_qwen3_14b   | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —        | —    | —        | —       | —          | 
| homo_qwen3_8b             | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —        | —    | —        | —       | —          |
| hetero_qwen3_14b_qwen3_8b | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —        | —    | —        | —       | —          |

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
| homo_qwen3_14b            | hotel      | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| homo_qwen3_14b            | restaurant | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| hetero_qwen25_qwen3_14b   | hotel      | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| hetero_qwen25_qwen3_14b   | restaurant | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| homo_qwen3_8b             | hotel      | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| homo_qwen3_8b             | restaurant | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| hetero_qwen3_14b_qwen3_8b | hotel      | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |
| hetero_qwen3_14b_qwen3_8b | restaurant | —        | —        | —       | —    | —      | —       | —     | —        | —        | —     | —       | —          |

---

### Experiment 3: Modular Fine-Tuned Pipeline

| Config             | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Inform% | Success% | BLEU  | Combined | Cost($) | Latency(s) |
|--------------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|----------|-------|----------|---------|------------|
| ft_homo_llama32_3b | 98.2     | 94.3     | 77.1    | 43.5 | 87.5   | 87.8    | 27.9  | 0.3      | 90.6     | 44.6  | 57.0    | 39.8     | 9.39  | 57.79    | $0.0000 | 2.12s      |
| ft_homo_qwen3_8b   | 99.3     | 95.7     | 78.9    | 47.3 | 88.1   | 89.1    | 22.1  | 0.6      | 92.6     | 48.3  | 53.2    | 48.4     | 11.35 | 62.15    | $0.0000 | 3.36s      |
| ft_homo_qwen3_14b  | 99.0     | 95.7     | 79.2    | 47.7 | 88.1   | 89.5    | 14.1  | 2.0      | 94.7     | 54.2  | 61.3    | 55.4     | 12.26 | 70.61    | $0.0000 | 3.42s      |

Per-Domain Breakdown

| Config               | Domain     | DomainP% | IntentP% | Action% | JGA% | SlotR% | SlotF1% | Hall% | PolViol% | SysCorr% | Book% | Cost($) | Latency(s) |
|----------------------|------------|----------|----------|---------|------|--------|---------|-------|----------|----------|-------|---------|------------|
| ft_homo_llama32_3b   | hotel      | 99.8     | 94.6     | 76.4    | 35.7 | 85.9   | 85.6    | 18.5  | 0.4      | 94.0     | 56.8  | $0.0000 | 2.16s      |
| ft_homo_llama32_3b   | restaurant | 96.3     | 93.8     | 78.0    | 53.0 | 89.4   | 90.3    | 38.2  | 0.2      | 86.5     | 67.4  | $0.0000 | 2.08s      |
| ft_homo_qwen3_8b     | hotel      | 99.8     | 95.2     | 77.7    | 37.5 | 86.0   | 87.2    | 19.5  | 0.4      | 94.0     | 74.2  | $0.0000 | 3.39s      |
| ft_homo_qwen3_8b     | restaurant | 98.7     | 96.4     | 80.5    | 59.7 | 90.9   | 91.6    | 24.8  | 0.8      | 90.8     | 72.6  | $0.0000 | 3.32s      |
| ft_homo_qwen3_14b    | hotel      | 100.0    | 95.3     | 78.7    | 40.4 | 86.7   | 88.3    | 13.3  | 1.2      | 95.9     | 82.6  | $0.0000 | 3.43s      |
| ft_homo_qwen3_14b    | restaurant | 97.7     | 96.2     | 79.8    | 56.8 | 89.8   | 91.1    | 15.1  | 3.0      | 93.2     | 79.4  | $0.0000 | 3.41s      |


---

## References

- **Dataset:** [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz) 
- **Evaluation:** [Tomiinek MultiWOZ Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation) 
- **Fine-tuning:** [Unsloth](https://github.com/unslothai/unsloth)
- **LoRA / QLoRA:** [PEFT](https://github.com/huggingface/peft)
- **Training loop:** [TRL](https://github.com/huggingface/trl)
- **Models:** [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

*Proudly the first LLM pipeline to ship from Chatzis, Greece.* 🇬🇷
