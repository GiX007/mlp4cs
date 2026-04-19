# Fine-Tuning Process

Implementation details, design decisions, and concrete examples for (Q)LoRA fine-tuning.

---

## 1. Overview
Each Exp3 config fine-tunes two LoRA adapters: one for DST, one for ResponseGen on MultiWOZ 2.2 train data. Training uses HuggingFace SFTTrainer (TRL library) with Unsloth's QLoRA for 4-bit base weights. Dev split drives validation loss and early stopping. Test split is untouched during training.

### Train / dev / test flow
- **Train:** MultiWOZ 2.2 train split → Alpaca-format JSON. Used to fit LoRA adapter weights
- **Dev:** MultiWOZ 2.2 dev split → Alpaca-format JSON. Used for eval loss every 100 steps and for early stopping
- **Test:** MultiWOZ 2.2 test split. Used only for final evaluation after model selection. Never seen during training

### Training and  inference prompt format
Fine-tuning assembles each example as a raw Alpaca string:

```
### Instruction:
<instruction>

### Input:
<input>

### Response:
<output><EOS>
```

No chat template (like in exp1, exp2), no `<think>` tokens. The Alpaca string is tokenized normally and passed to SFTTrainer. Inference must mirror this exact format as the pipeline's `build_dst_prompt()` and `build_respgen_prompt()` are the same functions used to generate the training data, guaranteeing train/inference consistency.

---

## 2. Configuration
Full config in `scripts/finetune.py`. Key values:

| Parameter               | Value           | Purpose                                                   |
|-------------------------|-----------------|-----------------------------------------------------------|
| Effective batch size    | 8               | `per_device_batch=2 × gradient_accumulation=4`            |
| Learning rate           | 2e-4            | Unsloth/QLoRA default, linear schedule                    |
| Warmup steps            | 10              | LR ramps up gradually at start                            |
| Max epochs              | 3               | Cap, early stopping decides actual end                    |
| LoRA rank / alpha       | 16 / 16         | All 7 attention + MLP modules targeted                    |
| Base precision          | 4-bit (NF4)     | QLoRA with frozen, adapters kept at 16-bit                |
| Optimizer               | adamw_8bit      | Pairs with 4-bit QLoRA                                    |
| Weight decay            | 0.001           | L2 regularization                                         |
| Seed                    | 3407            | Kept constant across all configs                          |
| Eval strategy           | every 100 steps | Full dev pass per eval event                              |
| Save strategy           | every 100 steps | Matches eval_steps for `load_best_model_at_end`           |
| Early stopping patience | 3               | Stop after 3 consecutive non-improving evals              |
| Max seq length          | 2048            | Max observed token length ~1,580 (DST) / ~1,326 (RespGen) |

---

## 3. How Training Works

### At each training step
1. Load a train batch of 2 samples.
2. Forward pass → compute loss.
3. Backward pass → compute gradients.
4. After 4 such steps (`gradient_accumulation_steps=4`): one `optimizer.step()` updates LoRA weights.
5. Log train loss every 10 optimizer steps.

### Validation during training
- Triggered every 100 optimizer steps (`eval_steps=100`)
- Training pauses, model switches to eval mode
- Full forward-only pass over the entire dev set (batches of 2, no gradient)
- Mean loss across all dev batches → logged as `eval_loss`
- A checkpoint (LoRA weights + optimizer state) is saved
- Training resumes

### Early stopping and best checkpoint
- `EarlyStoppingCallback(patience=3)` monitors `eval_loss`
- If eval_loss fails to improve for 3 consecutive evaluations, training stops
- `load_best_model_at_end=True` ensures the final adapter loaded is the checkpoint with minimum eval_loss and not the latest one
- `save_total_limit=3` keeps only the 3 most recent checkpoints on disk; the best checkpoint is preserved automatically regardless of this limit

### Expected behavior on the loss curves
- Train loss decreases continuously as the model fits train data
- Eval loss decreases, then plateaus, then may rise (overfitting)
- Best checkpoint is the step with minimum eval_loss before the rise begins

---

## 4. Step and Evaluation Math

### Formula

```
Steps per epoch = train_samples / effective_batch_size
Total steps (max) = steps_per_epoch × num_train_epochs
Total evals (max) = total_steps / eval_steps
```

With `effective_batch=8`, `num_train_epochs=3`, `eval_steps=100`.

### Worked example: DST

```
Total train samples = 10,846
Steps per epoch = 10,846 / 8 ≈ 1,356
Total steps (3 epochs) = 1,356 × 3 ≈ 4,068
Total evals = 4,068 / 100 ≈ 40
```

40 validation events across a full 3-epoch run → 40 eval_loss points on the plot.

### Worked example: RespGen

```
Total train samples = 13,174
Steps per epoch = 13,174 / 8 ≈ 1,647
Total steps (3 epochs) = 1,647 × 3 ≈ 4,940
Total evals = 4,940 / 100 ≈ 49
```

### If early stopping fires early

```
Example: stop triggered at step 2,000
Evals observed = 2,000 / 100 = 20
```

Fewer points but enough to show curve shape (plateau, rise, turn).

### Resolution vs cost
Smaller `eval_steps` means more eval points on the curve, but each eval takes time. One eval runs through the full dev set (~858 DST / ~1,038 RespGen samples). 
At `eval_steps=50` we'd get twice the points but also twice the eval time so hours added to the run. 
At `eval_steps=500` the run is faster, but we get only ~8 points, too few to see when overfitting starts. 
`eval_steps=100` is the middle ground and what HuggingFace and Unsloth tutorials use for this dataset size.

---

## 5. Example Scenarios

### Reading the loss curves

**Scenario A: model is still learning**

```
Epoch 1: train_loss=0.62, eval_loss=0.71
Epoch 2: train_loss=0.48, eval_loss=0.65
```

Both losses drop, eval follows train closely. No overfitting. Continue training.

**Scenario B: overfitting**

```
Step 500: train_loss=0.31, eval_loss=0.48
Step 600: train_loss=0.22, eval_loss=0.51
```

Train drops further, eval rises. Generalization is slipping. Best checkpoint is step 500. Early stopping hasn't fired yet (patience=3). 
If eval keeps rising through steps 700 and 800, early stopping triggers at step 800, and the final saved model is checkpoint-500 as taken before overfitting started.

### Full early-stopping sequence

```
Step 100: eval_loss=0.52   (best so far)
Step 200: eval_loss=0.45   (new best)
Step 300: eval_loss=0.41   (new best)
Step 400: eval_loss=0.43   (worse, patience 1/3)
Step 500: eval_loss=0.42   (worse, patience 2/3)
Step 600: eval_loss=0.44   (worse, patience 3/3 → STOP)

Training stops.
Disk contains: checkpoint-300 (best, preserved), checkpoint-500, checkpoint-600.
Trainer loads checkpoint-300 into model.
model.save_pretrained(...) writes the final adapter from checkpoint-300.
```

---

## 6. Outputs
Per adapter, saved under `data/finetuned_models/{model_alias}_{role}/`:

- LoRA weights (~20-40 MB)
- `training_summary.json`: full log_history, VRAM, runtime, trainable params
- `loss_plot.png`: train and eval loss curves vs step

---
