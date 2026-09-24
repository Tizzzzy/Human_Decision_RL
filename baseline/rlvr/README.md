# Verifiable-Reward RL Training for Qwen3.6-27B

## Overview

This module implements **verifiable-reward RL training** for Qwen3.6-27B. Unlike traditional reward models that estimate correctness probabilistically, this approach:

1. **Prompts the model** to generate BOTH an explanation AND an explicit judgment (AI/Human)
2. **Computes reward directly** by comparing the model's judgment to ground truth labels from the database
3. **Rewards correct judgments** (+1.0) and penalizes incorrect ones (-5.0)

This makes the reward signal "verifiable" because correctness is **binary and checkable** against the `text_guess` table in the database.

## Directory Structure

```
rlvr/
├── config.py                  # Centralized configuration
├── prompt_templates.py        # Prompt building and judgment parsing
├── dataset.py                 # Data loading from DB and text files
├── reward.py                  # Verifiable reward computation
├── train_grpo_verifiable.py   # Main GRPO training script
├── smoke_test.py              # Pre-training validation tests
├── run_smoke_test.sbatch      # SLURM script for smoke tests
├── run_train_grpo.sbatch      # SLURM script for full training
├── README.md                  # This file
└── checkpoints/               # Training outputs (created after first run)
    ├── final_adapter/         # Trained LoRA adapter
    ├── logs/                  # Training logs
    └── ...
```

## Quick Start

### Step 1: Run Smoke Tests (Recommended First)

```bash
# Option A: Submit SLURM job
sbatch /gpfs/projects/p32143/RL_human_decision/baseline/rlvr/run_smoke_test.sbatch

# Option B: Interactive GPU allocation (for quick testing)
salloc --partition=gengpu --gres=gpu:a100:1 --time=01:00:00 --mem=100G
cd /gpfs/projects/p32143/RL_human_decision/baseline/rlvr
conda activate ppo_2
python smoke_test.py
```

The smoke tests verify:
- ✓ Judgment parsing from model outputs
- ✓ Reward computation logic
- ✓ Dataset loading from DB
- ✓ Prompt formatting with tokenizer
- ✓ Model loading (GPU-only)

### Step 2: Run Full Training

After smoke tests pass:

```bash
# Submit training job
sbatch /gpfs/projects/p32143/RL_human_decision/baseline/rlvr/run_train_grpo.sbatch

# Monitor progress
tail -f /gpfs/projects/p32143/RL_human_decision/baseline/rlvr/logs/<job_id>.out
```

## Architecture

### 1. Data Pipeline

**Source:** `text_guess` table from `/gpfs/projects/p32143/RL_human_decision/statistic/text_2026-09-02.db`

**Key fields:**
- `text_id`: Text identifier
- `label_truth`: Ground truth label ("AI" or "Human")

**Process:**
1. Load all distinct `(text_id, label_truth)` pairs from database
2. Map to text files:
   - AI texts: `webscrape/social/SocialMedia_rewrite/{text_id}.txt`
   - Human texts: `webscrape/social/SocialMedia_Reddit/{text_id}.txt`
3. Stratified train/test split (85/15) by label
4. Build HF Dataset with formatted prompts

**Data Stats (approximate):**
- ~3,715 distinct (text_id, label_truth) pairs
- ~1,838 AI texts
- ~1,877 Human texts

### 2. Prompt Format

The policy is prompted to generate both explanations and judgments:

```
Task: Analyze the provided social media post for linguistic markers of AI or human authorship.

Constraints:
1. Start with bullet points of linguistic markers
2. No introduction, verdict summary, or preamble
3. Flat list (no sub-headings)
4. Every bullet: "Look for [category]" or "Look for [category] like '[example]'"
5. 2-6 bullet points total
6. After bullets, MUST end with: "JUDGMENT: AI" or "JUDGMENT: Human"

Post:
```
{text}
```

Explanation:
```

**Expected model output:**
```
Look for formal vocabulary and technical jargon.
Look for lack of contractions like "don't" or "it's".
Look for repetitive sentence structures.

JUDGMENT: AI
```

### 3. Reward Computation

For each generated completion:

1. **Parse judgment:** Extract "AI" or "Human" from the `JUDGMENT:` line
2. **Check ground truth:** Compare against `label_truth` from database
3. **Assign reward:**
   - ✓ Correct judgment: `+1.0`
   - ✗ Incorrect judgment: `-5.0`
   - ✗ Parse error (no valid judgment): `-5.0`

This creates a **verifiable, binary reward signal** that directly optimizes for judgment correctness.

### 4. Training Loop

Uses `GRPOTrainer` from TRL:

- **Algorithm:** GRPO (Group Relative Policy Optimization)
- **Policy:** Qwen3.6-27B with LoRA adapter
- **Reward:** Direct correctness check
- **Advantage estimation:** Group-relative baseline (no critic model needed)
- **Training strategy:** LoRA fine-tuning for efficiency

**Key hyperparameters:**
- LoRA: r=16, α=32
- Learning rate: 1e-5
- Batch size: 8 (per device)
- Gradient accumulation: 4
- Generation temperature: 1.0
- Group size (num_generations): 8

## Configuration

Edit [`config.py`](config.py) to adjust:

```python
# Dataset
MAX_TEXTS = None  # Set to 1000 for debugging
TRAIN_TEST_SPLIT = 0.85

# LoRA
LORA_R = 16
LORA_ALPHA = 32

# GRPO
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 8

# Reward
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -5.0
```

## Key Files Explained

### `prompt_templates.py`

- `build_policy_prompt(text, tokenizer)`: Formats text into a prompt asking for explanation + judgment
- `parse_generation(generated_text)`: Extracts judgment ("AI", "Human", or None) from model output

### `dataset.py`

- `load_distinct_texts_from_db()`: Queries database for unique (text_id, label_truth) pairs
- `build_hf_dataset(records, tokenizer)`: Converts to HuggingFace Dataset
- `get_train_test_datasets(tokenizer)`: Returns stratified train/test split

### `reward.py`

- `extract_judgment(completion_text)`: Parses model's judgment
- `compute_reward(predicted_judgment, true_label)`: Returns +1.0 (correct) or -5.0 (incorrect)
- `reward_func(prompts, completions, label_truth, ...)`: GRPOTrainer-compatible reward function

### `train_grpo_verifiable.py`

Main training loop:
1. Loads Qwen3.6-27B policy model
2. Builds LoRA config
3. Creates GRPOTrainer with verifiable reward function
4. Trains for N epochs
5. Saves final adapter to `checkpoints/final_adapter/`

## Expected Training Time

With 2x A100 GPUs:
- Model loading: 2-3 minutes
- Data loading: 1-2 minutes
- Per epoch (3,715 texts, batch_size=8): ~20-30 minutes
- **Total (3 epochs): ~1.5-2.5 hours**

Monitor with:
```bash
tail -f /gpfs/projects/p32143/RL_human_decision/baseline/rlvr/logs/<job_id>.out
```

## Output

After training completes, the trained model and metrics are saved to:
```
/gpfs/projects/p32143/RL_human_decision/baseline/rlvr/checkpoints/
```

### Model Files
```
final_adapter/
  ├── adapter_model.safetensors      # LoRA adapter weights
  ├── adapter_config.json            # LoRA configuration
  ├── config.json                    # Model configuration
  ├── tokenizer.json                 # Tokenizer
  └── special_tokens_map.json        # Token mappings
```

### Training Metrics & Plots
```
logs/
  ├── training_metrics.json          # Raw metrics (step, training/eval loss, lr, reward)
  ├── validation_metrics.json        # Final validation accuracy
  ├── training_loss.png              # Loss curves (4 plots in 1 figure)
  ├── training_loss.pdf              # PDF version of loss plot
  ├── train_vs_validation_loss.png   # Training vs validation loss comparison
  └── train_vs_validation_loss.pdf   # PDF version
```

The loss plots include:
1. **Main loss plot**: Raw loss curve, smoothed loss, learning rate schedule, reward progression
2. **Train vs validation plot**: Side-by-side training and validation loss curves
3. **Statistics box** — Summary metrics (min/max/avg loss, improvement %)

**Validation metrics saved:**
- `accuracy`: Final judgment accuracy on validation set
- `num_correct`: Number of correct judgments
- `num_total`: Total samples evaluated

## Visualizing Training Loss

After training completes, loss curves are automatically saved to:
- `checkpoints/logs/training_loss.png` — Main visualization (PNG)
- `checkpoints/logs/training_loss.pdf` — PDF version for printing

### Regenerating Plots

To regenerate plots after training or compare multiple runs:

```bash
# Regenerate from default location
python plot_training_loss.py

# Custom metrics file location
python plot_training_loss.py --metrics-file /path/to/training_metrics.json

# Compare multiple training runs
python plot_training_loss.py --compare \
  "run_1" checkpoints1/logs/training_metrics.json \
  "run_2" checkpoints2/logs/training_metrics.json \
  "run_3" checkpoints3/logs/training_metrics.json

# Add epoch boundary markers
python plot_training_loss.py --epochs 3 --samples-per-epoch 1000
```

### Metrics Recorded

The `training_metrics.json` file contains:
- `step`: Training step number
- `loss`: Training loss at each step
- `learning_rate`: Learning rate at each step
- `reward`: Reward signal (mean reward in batch)

### Plot Interpretation

**Loss Curve:**
- Should generally decrease over training
- Small fluctuations are normal (batch-to-batch variance)
- Large jumps may indicate instability

**Smoothed Loss:**
- Removes noise to show overall trend
- Easier to see if model is converging

**Learning Rate Schedule:**
- Shows how learning rate changes
- Linear decay is typical

**Reward Progression:**
- Shows model performance on reward signal
- Should trend positive if training is working

## Next Steps After Training

### 1. Inference

Use the trained model to generate new explanations + judgments:

```bash
python inference_generate_judgments.py  # (to be implemented)
```

### 2. Evaluation

Evaluate the trained model's judgment accuracy on test set:

```bash
python evaluate_judgment_accuracy.py  # (to be implemented)
```

### 3. Further Iteration

If accuracy is good, continue training for more epochs:

```bash
python train_grpo_verifiable.py --resume-from-checkpoint=<checkpoint_path>
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size in config.py, use 4x GPUs, or reduce MAX_TEXTS for testing |
| Model not found | Check `/projects/p32143/cache/huggingface/qwen36_27b/` exists with 52GB of weights |
| Database error | Verify DB path: `/gpfs/projects/p32143/RL_human_decision/statistic/text_2026-09-02.db` |
| Text files not found | Check webscrape directories have text files with correct naming |
| Judgment parsing fails | Model may be generating non-standard format; check sample outputs in logs |

## Differences from Previous RL Approach

| Aspect | Previous (RL/) | New (rlvr/) |
|--------|----------------|-----------|
| **Reward signal** | Frozen probe P(AI) (probabilistic) | Direct correctness check (binary/verifiable) |
| **Training data** | Undecided texts (no human decisions) | Decided texts with ground truth labels |
| **Model output** | Explanations only | Explanations + explicit judgment |
| **Verification** | Requires probe calibration | Binary: right or wrong |
| **Data size** | ~805 texts | ~3,715 texts |

## References

- TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
- LoRA: https://arxiv.org/abs/2106.09685
- Qwen3.6-27B: https://huggingface.co/Qwen/Qwen3.6-27B
