# Validation Set Tracking & Evaluation

## Overview

The training pipeline now tracks **validation loss during training** and computes **final judgment accuracy on the validation set** after training completes.

## What Gets Tracked

### During Training (streaming)
In `checkpoints/logs/training_metrics.json`:
```json
{
  "step": [0, 10, 20, 30, ...],
  "training_loss": [2.543, 2.452, 2.389, 2.310, ...],
  "eval_loss": [2.510, 2.420, 2.359, ...],
  "learning_rate": [0.00001, 0.00001, ...],
  "reward": [0.125, 0.250, 0.375, ...]
}
```

- **training_loss**: Loss on training batch
- **eval_loss**: Loss on validation batch (when trainer evaluates)
- **learning_rate**: Current learning rate
- **reward**: Mean reward in batch

### After Training (final evaluation)
In `checkpoints/logs/validation_metrics.json`:
```json
{
  "accuracy": 0.7234,
  "num_correct": 402,
  "num_total": 556
}
```

- **accuracy**: % of correct judgments on validation set (0.0-1.0)
- **num_correct**: Number of correct predictions
- **num_total**: Total validation samples evaluated

## Automatic Plots

After training, two new plots are generated:

### 1. `training_loss.png` — Main Loss Plot
(Same 5-subplot figure as before)

### 2. `train_vs_validation_loss.png` — NEW!
Shows training loss vs validation loss:
- **Blue line**: Training loss (updated frequently)
- **Orange line**: Validation loss (less frequent, evaluated every few steps)

**Interpretation:**
- If lines track together → model generalizes well
- If validation loss diverges above training loss → overfitting
- If validation loss increases while training decreases → model is memorizing

## Output Files

```
checkpoints/logs/
├── training_metrics.json           # Step, loss, lr, reward (auto-saved during training)
├── validation_metrics.json         # Final accuracy (auto-computed after training)
├── training_loss.png              # Main loss curves
├── training_loss.pdf              # PDF version
├── train_vs_validation_loss.png   # NEW: Train vs validation
└── train_vs_validation_loss.pdf   # NEW: PDF version
```

## How It Works

### Validation Loss During Training

The `LossTrackingCallback` captures `eval_loss` from GRPOTrainer's logs:

```python
if "eval_loss" in logs:
    self.metrics["eval_loss"].append(logs["eval_loss"])
```

The trainer evaluates on validation set automatically if `eval_dataset` is provided. By default, evaluation happens every `eval_steps` (configurable in GRPOConfig).

### Final Validation Accuracy

After training, `compute_validation_accuracy()` runs:

```python
def compute_validation_accuracy(model, tokenizer, eval_dataset, num_samples=None):
    # For each validation sample:
    # 1. Generate judgment using trained model
    # 2. Parse judgment from output
    # 3. Compare to ground truth
    # 4. Return accuracy = num_correct / num_total
```

This evaluates the **trained model with LoRA adapter** on the full validation set (~556 samples).

## Key Differences

| Metric | Training Loss | Validation Loss | Final Accuracy |
|--------|--------------|-----------------|-----------------|
| **What** | Loss on training batches | Loss on validation batches | Judgment correctness on validation set |
| **When** | Every logging step | Every eval_steps (default: per epoch) | Once, at end of training |
| **Source** | GRPOTrainer logs | GRPOTrainer logs | Post-training evaluation |
| **Interpretation** | Should decrease | Should stay low (not diverge) | Final model performance (0.0-1.0) |

## Interpreting Validation Metrics

### Good Signs ✅
- Training loss: steadily decreasing
- Validation loss: decreases with training, stays close to training loss
- Final accuracy: 65-75%+ (vs ~50% baseline random)

### Concerning Signs ⚠️
- Validation loss: diverges above training loss (overfitting)
- Validation loss: stays constant while training decreases (not generalizing)
- Final accuracy: near 50% (model not learning)

### Excellent Signs 🎯
- Training and validation loss track together
- Both decrease smoothly
- Final accuracy: 75%+

## Configuration

In `config.py`, modify evaluation behavior:

```python
# GRPO Training Configuration
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8
# Note: eval_strategy and eval_steps are set in GRPOConfig
```

The GRPOTrainer automatically evaluates at:
- End of each epoch (default)
- Or every N steps if configured

## Regenerating Plots

To regenerate train vs validation plot:

```bash
python plot_training_loss.py

# Automatically generates all plots:
# - training_loss.png (main)
# - train_vs_validation_loss.png (NEW)
# - training_loss_epochs.png (if epoch markers provided)
```

## Usage Example

```bash
# 1. Start training (metrics auto-recorded)
sbatch run_train_grpo.sbatch

# 2. After training, check results
tail -100 checkpoints/logs/training_loss.out

# 3. View validation metrics
cat checkpoints/logs/validation_metrics.json
# Output: {"accuracy": 0.7234, "num_correct": 402, "num_total": 556}

# 4. View plots
open checkpoints/logs/training_loss.png
open checkpoints/logs/train_vs_validation_loss.png

# 5. If unsatisfied, retrain with different settings
# Edit config.py and run training again
```

## Technical Implementation

### LossTrackingCallback Enhancement

Modified to capture validation metrics:

```python
class LossTrackingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Records training_loss, eval_loss, learning_rate, reward
        if "eval_loss" in logs:
            self.metrics["eval_loss"].append(logs["eval_loss"])
```

### Validation Accuracy Function

New function `compute_validation_accuracy()`:

```python
def compute_validation_accuracy(model, tokenizer, eval_dataset, num_samples=None):
    # Uses trained model with LoRA adapter
    # Generates judgment for each validation sample
    # Compares to ground truth
    # Returns accuracy metrics
```

### Post-Training Evaluation

Added to `main()` after training completes:

```python
print("[5.5/6] Computing validation set accuracy...")
val_metrics = compute_validation_accuracy(eval_model, policy_tokenizer, eval_dataset)
print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
```

## Performance Impact

- **Validation loss tracking**: Already handled by GRPOTrainer
- **Final accuracy computation**: ~5-10 minutes (depends on validation set size)
- **Plotting**: < 1 second

Total overhead: Minimal, mainly from final accuracy evaluation

## Troubleshooting

### No validation metrics file
**Problem**: `validation_metrics.json` wasn't created

**Solution**:
- Check training completed fully
- Ensure eval_dataset was created successfully
- Check logs for errors during evaluation

### Validation loss not appearing
**Problem**: `eval_loss` not in training_metrics.json

**Solution**:
- GRPOTrainer may not evaluate at that step
- Check eval_steps in trainer config
- Increase eval frequency if desired

### Accuracy seems wrong
**Problem**: Final accuracy doesn't match evaluation_judgment_accuracy.py

**Solution**:
- post-training accuracy uses greedy decoding
- evaluate_judgment_accuracy.py uses sampling
- Different settings → different results
- Run both and compare

## Files Modified for Validation

| File | Changes |
|------|---------|
| `train_grpo_verifiable.py` | Added validation accuracy function, post-training eval |
| `plot_utils.py` | Added `plot_train_vs_validation()` function |
| `plot_training_loss.py` | Auto-generates train vs validation plot |
| `LOSS_TRACKING.md` | Mentions validation metrics |
| `README.md` | Documents validation outputs |

## Next Steps

1. ✓ Training now tracks validation loss
2. ✓ Final accuracy computed automatically
3. ✓ Validation plots generated
4. ( ) Compare train vs validation loss to detect overfitting
5. ( ) If overfitting detected, tune hyperparameters (LoRA r, dropout, etc.)
6. ( ) If underfitting, train longer or increase model capacity

## Related Commands

```bash
# Monitor training (in another terminal)
tail -f checkpoints/logs/<job_id>.out

# Check validation metrics after training
cat checkpoints/logs/validation_metrics.json

# View train vs validation plot
open checkpoints/logs/train_vs_validation_loss.png

# Regenerate all plots
python plot_training_loss.py

# Run detailed evaluation with sampling
python evaluate_judgment_accuracy.py --checkpoint-path checkpoints/final_adapter
```
