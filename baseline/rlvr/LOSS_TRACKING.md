# Loss Tracking & Visualization Guide

## Overview

The training pipeline now automatically tracks and visualizes training metrics including:
- **Training loss** over time
- **Learning rate** schedule
- **Reward signal** progression
- **Training statistics** (min/max/avg loss, improvement %)

## What Gets Recorded

During training, the following metrics are saved to `checkpoints/logs/training_metrics.json`:

```json
{
  "step": [0, 10, 20, 30, ...],
  "training_loss": [2.5432, 2.4521, 2.3891, 2.3102, ...],
  "eval_loss": [2.5102, 2.4201, 2.3591, ...],
  "learning_rate": [0.00001, 0.00001, 0.00001, ...],
  "reward": [0.125, 0.250, 0.375, ...]
}
```

After training completes, validation accuracy is saved to `checkpoints/logs/validation_metrics.json`:

```json
{
  "accuracy": 0.7234,
  "num_correct": 402,
  "num_total": 556
}
```

## Auto-Generated Plots

After training completes, two visualizations are automatically created:

### `checkpoints/logs/training_loss.png` (PNG)
Main visualization with 5 subplots:
1. **Loss over time** — Raw training loss curve
2. **Smoothed loss** — Moving average (window=5) for trend visualization
3. **Learning rate schedule** — How LR changes during training
4. **Reward progression** — Model's reward signal over time
5. **Statistics box** — Summary metrics:
   - Total steps
   - Initial/final/min/max loss
   - Average loss
   - Loss change (absolute)
   - Improvement percentage

### `checkpoints/logs/training_loss.pdf` (PDF)
Same content as PNG, optimized for printing.

## Regenerating Plots

If you want to regenerate plots after training, or compare multiple runs:

```bash
# Basic usage - plots from default location
python plot_training_loss.py

# Custom metrics file
python plot_training_loss.py --metrics-file /custom/path/training_metrics.json

# Save to different directory
python plot_training_loss.py --output-dir /custom/output

# Compare multiple training runs
python plot_training_loss.py --compare \
  "baseline" checkpoints1/logs/training_metrics.json \
  "with_warmup" checkpoints2/logs/training_metrics.json \
  "higher_lr" checkpoints3/logs/training_metrics.json

# Add epoch boundaries to plot
python plot_training_loss.py \
  --epochs 3 \
  --samples-per-epoch 1000
```

## Implementation Details

### Loss Tracking Callback

The `LossTrackingCallback` class (in `train_grpo_verifiable.py`) hooks into GRPOTrainer:

```python
class LossTrackingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called when trainer logs metrics
        # Records: step, loss, learning_rate, reward
        
    def on_train_end(self, args, state, control, **kwargs):
        # Called when training ends
        # Saves final metrics to JSON
```

The callback:
- Records metrics every logging step (default: every 5 steps)
- Saves to JSON after every 10 steps (plus at end of training)
- Captures: step, loss, learning_rate, reward

### Plotting Module

The `plot_utils.py` module provides:

```python
plot_training_loss(metrics_file, output_dir)
  # Main plotting function - creates 5-subplot figure

plot_loss_vs_epochs(metrics_file, num_epochs, samples_per_epoch, output_dir)
  # Alternative view with epoch boundaries

compare_training_runs(metrics_files_dict, output_dir)
  # Compare multiple training runs on one plot

_smooth_curve(values, window)
  # Helper: moving average smoothing
```

## Interpreting the Plots

### Loss Curve
- **Good**: Steadily decreasing
- **OK**: Decreases with minor fluctuations (batch-to-batch variance)
- **Concerning**: Constant or increasing loss
- **Very concerning**: Large spikes or erratic behavior

### Smoothed Loss
- Shows the underlying trend more clearly
- Helps distinguish signal from noise
- Easier to see if model is converging

### Learning Rate
- Shows how LR changes during training
- Default: Linear decay from initial LR to near-zero
- Should decay smoothly without jumps

### Reward Progression
- Shows how model performance evolves
- With binary rewards (+1/-5):
  - Reward mean should trend towards +1 if training works
  - Starting near 0 (random)
  - Ideally reaching +0.5 or higher by end

### Statistics Box
- **Initial loss**: What model started with
- **Final loss**: What model achieved
- **Improvement %**: (1 - final/initial) × 100
  - 50% improvement means final loss is half of initial
  - 100% improvement means loss is 0 (unlikely)

## Troubleshooting

### No metrics file found
**Problem**: Training completed but `training_metrics.json` is empty or missing

**Solution**:
- Check that logging is enabled: `LOGGING_STEPS = 5` in config.py
- Verify training ran more than 5 steps
- Check directory permissions: `ls -la checkpoints/logs/`

### Plot looks strange
**Problem**: Metrics recorded but visualization looks odd

**Solution**:
- Check raw JSON: `cat checkpoints/logs/training_metrics.json`
- Regenerate with: `python plot_training_loss.py`
- Try with different smoothing window: Edit `plot_utils.py`

### Comparison plot fails
**Problem**: Trying to compare runs but error occurs

**Solution**:
- Verify each metrics file exists and is valid JSON
- Use full paths: `/absolute/path/to/training_metrics.json`
- Check file permissions

## Example Workflow

```bash
# 1. Start training
sbatch run_train_grpo.sbatch

# 2. Monitor in real-time (in another terminal)
tail -f logs/<job_id>.out

# 3. After training, check plots
open checkpoints/logs/training_loss.png

# 4. If unsatisfied, regenerate with different smoothing
python plot_training_loss.py  # Regenerate

# 5. Compare with another run
python plot_training_loss.py --compare \
  "run_1" checkpoints1/logs/training_metrics.json \
  "run_2" checkpoints2/logs/training_metrics.json
```

## Files Related to Loss Tracking

| File | Purpose |
|------|---------|
| `train_grpo_verifiable.py` | Main training script with LossTrackingCallback |
| `plot_utils.py` | Plotting utility functions |
| `plot_training_loss.py` | Standalone script for regenerating plots |
| `checkpoints/logs/training_metrics.json` | Raw metrics (auto-generated) |
| `checkpoints/logs/training_loss.png` | Main loss plot (auto-generated) |
| `checkpoints/logs/training_loss.pdf` | PDF version (auto-generated) |

## Advanced Usage

### Custom Callback Behavior

To modify how metrics are recorded, edit the `LossTrackingCallback` class:

```python
# Save more frequently
if state.global_step % 5 == 0:  # Change 10 to 5
    self._save_metrics()

# Track additional metrics
if "some_new_metric" in logs:
    self.metrics["new_key"].append(logs["some_new_metric"])
```

### Custom Plot Styling

Modify `plot_utils.py` to change plot appearance:

```python
# Change colors
color="#1f77b4"  # Change hex color code

# Change figure size
fig = plt.figure(figsize=(14, 10))  # Change dimensions

# Change smoothing window
smoothed_loss = _smooth_curve(metrics["loss"], window=5)  # Increase window
```

## Dependencies

Loss tracking requires:
- `matplotlib` — For plotting
- `json` — Built-in, for saving metrics
- `torch` + `transformers` — For training
- `trl` — For GRPOTrainer (already required)

All dependencies are included in the `ppo_2` conda environment.

## Performance Impact

- **Metrics recording**: < 1% overhead per training step
- **File I/O**: ~100ms every 10 steps (negligible)
- **Disk space**: ~100 KB per 1000 training steps

Loss tracking has minimal performance impact and should not slow training noticeably.
