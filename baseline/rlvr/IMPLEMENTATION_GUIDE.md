# Implementation Guide: Verifiable-Reward RL

## Summary

This implementation trains Qwen3.6-27B to generate **both explanations AND correct AI/Human judgments**, with a **verifiable, binary reward signal** directly based on judgment correctness.

## Key Innovations

### 1. **Verifiable Reward**
- **Previous approach:** Used a frozen probe model to estimate P(AI) → probabilistic, requires careful calibration
- **New approach:** LLM generates explicit judgment → compare to ground truth → binary reward
- **Advantage:** Direct, interpretable, no need for auxiliary reward models

### 2. **Larger Training Set**
- **Previous approach:** ~805 undecided texts (texts without human decisions)
- **New approach:** ~3,715 texts with ground truth labels from database
- **Advantage:** 4.6× more training data with verified labels

### 3. **Two-Part Output**
- **Previous approach:** Explanation only
- **New approach:** Explanation + explicit judgment
- **Advantage:** Model explicitly learns to make correct decisions, not just plausible explanations

## Architecture Overview

```
Data Pipeline (dataset.py)
    ↓
    Load (text_id, label_truth) pairs from DB
    ↓
    Map to text files (SocialMedia_rewrite/ or SocialMedia_Reddit/)
    ↓
    Build prompts (prompt_templates.py)
    ↓
    Create HF Dataset

GRPO Training (train_grpo_verifiable.py)
    ↓
    Load Qwen3.6-27B + LoRA adapter
    ↓
    For each batch:
        - Generate completions (explanation + judgment)
        - Parse judgments (prompt_templates.py)
        - Compute rewards (reward.py)
        - Update via GRPO with group-relative advantage

Evaluation (evaluate_judgment_accuracy.py)
    ↓
    Load trained model + adapter
    ↓
    Generate predictions on test set
    ↓
    Compute accuracy, precision, recall, F1
    ↓
    Print confusion matrix
```

## File Responsibilities

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters and paths in one place |
| `prompt_templates.py` | Build prompts, parse judgments from model output |
| `dataset.py` | Query DB, load text files, build HF Dataset |
| `reward.py` | Extract judgment, compute reward (correct=+1, incorrect=-5) |
| `train_grpo_verifiable.py` | Main GRPO training loop with LoRA |
| `smoke_test.py` | Validate all components before training |
| `evaluate_judgment_accuracy.py` | Test set evaluation with metrics |
| `run_smoke_test.sbatch` | SLURM script for smoke tests |
| `run_train_grpo.sbatch` | SLURM script for full training |

## Step-by-Step Usage

### Phase 1: Validation

```bash
# Test on login node (CPU only)
cd /gpfs/projects/p32143/RL_human_decision/baseline/rlvr
conda activate ppo_2

# Run text-only tests (fast)
python smoke_test.py
# Expected: Tests 1-4 pass, test 5 skipped (no GPU)

# Then allocate GPU and run full tests
salloc --partition=gengpu --gres=gpu:a100:1 --time=01:00:00 --mem=100G
cd /gpfs/projects/p32143/RL_human_decision/baseline/rlvr
python smoke_test.py
# Expected: All tests pass
```

### Phase 2: Training

```bash
# Submit training job
cd /gpfs/projects/p32143/RL_human_decision/baseline/rlvr
sbatch run_train_grpo.sbatch

# Monitor progress
tail -f logs/<job_id>.out

# Expected runtime: 1.5-2.5 hours on 2x A100
# Output: checkpoints/final_adapter/
```

### Phase 3: Evaluation

```bash
# After training completes
conda activate ppo_2
python evaluate_judgment_accuracy.py --checkpoint-path checkpoints/final_adapter

# Or evaluate on subset for quick check
python evaluate_judgment_accuracy.py --checkpoint-path checkpoints/final_adapter --max-samples 100
```

## Prompt Design Details

### Input Prompt
```
Task: Analyze the provided social media post for linguistic markers of AI or human authorship.

[Detailed constraints about format...]

Post:
```
{text}
```

Explanation:
```

### Expected Output
```
Look for formal vocabulary and technical terminology.
Look for lack of contractions like "don't", "it's".
Look for repetitive sentence structures.

JUDGMENT: AI
```

### Parsing Logic
1. Find line starting with `JUDGMENT:`
2. Extract text after colon
3. Normalize to "AI" or "Human"
4. If not found or unrecognized, return None → reward -5.0

## Reward Computation

### Formula
```python
if predicted_judgment == ground_truth_label:
    reward = +1.0
else:
    reward = -5.0
```

### Batch-level Advantage (GRPO)
```
GRPO uses group-relative baselines:
- Group 8 completions for same prompt
- Advantages = rewards - group_mean_reward
- Update policy to maximize advantages
```

## Hyperparameter Tuning Guide

### If training is too slow:
```python
# config.py
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # ↓ reduce from 8
GRADIENT_ACCUMULATION_STEPS = 8  # ↑ increase to keep effective batch size
```

### If accuracy is not improving:
```python
# config.py
LEARNING_RATE = 5e-5  # Try higher LR
NUM_TRAIN_EPOCHS = 5  # Train longer
```

### If model runs out of memory:
```python
# config.py
MAX_COMPLETION_LENGTH = 256  # Limit explanation length
# Or allocate more GPUs in run_train_grpo.sbatch
```

## Expected Results

After 3 epochs of training (~2 hours on 2x A100):

**Judgment Accuracy:**
- Baseline (untrained): ~50-55% (random guessing)
- After RL training: 60-75%+ (estimated, depends on data quality)

**Per-class performance:**
- AI accuracy: 55-70%
- Human accuracy: 60-80%

**Reward statistics:**
- Mean training reward: Should increase from ~0 to +0.3 to +0.5+
- Distribution should shift towards +1.0 (correct judgments)

## Debugging

### Problem: Judgment parsing fails (returns None)
**Solution:**
1. Check sample outputs in logs
2. Model may be generating format variations (e.g., "JUDGMENT:AI" vs "JUDGMENT: AI")
3. Update parsing logic in `prompt_templates.parse_generation()` if needed
4. Consider stricter prompt instructions

### Problem: All rewards are -5.0
**Solution:**
1. Run smoke tests to verify DB and text files are accessible
2. Check sample prompts are formatting correctly
3. Ensure `label_truth` values match expected format ("AI" or "Human")
4. May indicate data quality issue

### Problem: Training doesn't converge
**Solution:**
1. Check reward signal is working (mix of +1.0 and -5.0)
2. Increase learning rate or training epochs
3. Reduce batch size for more frequent updates
4. Check model isn't stuck in local minimum

## Comparison to Previous RL Pipeline

| Feature | Previous RL (RL/) | Verifiable RL (rlvr/) |
|---------|-------------------|----------------------|
| **Data source** | Undecided texts (no human decisions) | Decided texts (ground truth) |
| **Reward type** | Frozen probe P(AI) | Direct correctness check |
| **Training objective** | Better explanations (proxy metric) | Correct AI/Human judgment |
| **Model output** | Explanation only | Explanation + Judgment |
| **Data size** | ~805 texts | ~3,715 texts |
| **Training time** | ~24 hours (4 GPU) | ~2 hours (2 GPU) |
| **Verification** | Requires probe calibration | Binary right/wrong |

## Next Steps

1. ✓ Run smoke tests to validate components
2. ✓ Run full GRPO training
3. ✓ Evaluate judgment accuracy on test set
4. ( ) Use trained model for inference on new texts
5. ( ) Compare explanation quality vs original model
6. ( ) Explore further training with curriculum learning
7. ( ) Deploy model for human evaluation

## References

- **GRPO:** https://huggingface.co/docs/trl/grpo_trainer
- **LoRA:** https://arxiv.org/abs/2106.09685
- **Qwen3.6-27B:** https://huggingface.co/Qwen/Qwen3.6-27B
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers

## Contact & Support

For issues, check:
1. Smoke test logs: `logs/*.out`
2. Training logs: `logs/<job_id>.out`
3. Data integrity: `SELECT COUNT(DISTINCT text_id, label_truth) FROM text_guess`
