# Quick Start: Verifiable-Reward RL

## 30-Second Overview

Train Qwen3.6-27B to generate **correct AI/Human judgments** using direct reward signal:
- **Prompt:** "Analyze text for AI/Human markers, then judge: AI or Human?"
- **Reward:** +1 if correct, -5 if wrong
- **Data:** ~3,715 texts with ground truth labels from database
- **Time:** ~2 hours on 2× A100

## Commands

### 1. Quick Validation (5 min on login node)
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/rlvr
conda activate ppo_2
python smoke_test.py
```
✓ If tests 1-4 pass, you're good to proceed
⚠ Test 5 requires GPU (skip on login node)

### 2. Full Validation (30 min, needs GPU)
```bash
sbatch run_smoke_test.sbatch
tail -f logs/<job_id>.out
```
✓ All tests should pass

### 3. Train (1.5-2.5 hours, needs 2× A100)
```bash
sbatch run_train_grpo.sbatch
tail -f logs/<job_id>.out
```
✓ Model saved to: `checkpoints/final_adapter/`
✓ Loss plot auto-generated: `checkpoints/logs/training_loss.png`

### 3b. View Loss Curves (Optional)
```bash
# Loss plots are automatically generated during training
# View the plot with your image viewer
open checkpoints/logs/training_loss.png  # macOS
# or: xdg-open checkpoints/logs/training_loss.png  # Linux
# or: copy checkpoints/logs/training_loss.png to local machine

# Regenerate plots anytime:
python plot_training_loss.py
```

### 4. Evaluate (30 min)
```bash
conda activate ppo_2
python evaluate_judgment_accuracy.py --checkpoint-path checkpoints/final_adapter
```
✓ Shows accuracy, precision, recall on test set

## Expected Output

After training, check logs:
```bash
tail -100 logs/<job_id>.out
```

Key lines to look for:
```
[1/5] Loading policy model and tokenizer...
[2/5] Loading datasets...
  Train: 3159, Test: 556
[3/5] Building LoRA config...
[4/5] Building GRPO trainer...
[5/5] Starting GRPO training...

[Training progress with steps and rewards...]

Training complete! Saving final model...
Final model saved to /gpfs/projects/p32143/RL_human_decision/baseline/rlvr/checkpoints/final_adapter
```

## File Structure

```
rlvr/
├── config.py                      # All settings in one place
├── prompt_templates.py            # Build prompts, parse judgments
├── dataset.py                     # Load from DB and text files
├── reward.py                      # Compute reward (correct/incorrect)
├── train_grpo_verifiable.py       # Main training loop
├── smoke_test.py                  # Pre-training checks
├── evaluate_judgment_accuracy.py  # Post-training evaluation
├── run_smoke_test.sbatch          # SLURM script for tests
├── run_train_grpo.sbatch          # SLURM script for training
├── README.md                      # Full documentation
├── IMPLEMENTATION_GUIDE.md        # Technical details
├── QUICKSTART.md                  # This file
└── checkpoints/                   # Created after first run
    ├── final_adapter/             # Trained LoRA weights
    ├── logs/                      # Training logs
    └── ...
```

## Troubleshooting

### Smoke tests fail with DB error
```bash
# Verify database exists
ls -lh /gpfs/projects/p32143/RL_human_decision/statistic/text_2026-09-02.db

# Verify text files exist
ls /gpfs/projects/p32143/RL_human_decision/webscrape/social/SocialMedia_Reddit/*.txt | wc -l
ls /gpfs/projects/p32143/RL_human_decision/webscrape/social/SocialMedia_rewrite/*.txt | wc -l
```

### Training is slow
- Reduce `PER_DEVICE_TRAIN_BATCH_SIZE` in `config.py` from 8 to 4
- Increase `GRADIENT_ACCUMULATION_STEPS` to 8

### Out of memory
- Use 4× GPUs instead of 2
- Set `MAX_COMPLETION_LENGTH = 256` in `config.py`
- Reduce `MAX_TEXTS = 1000` in `config.py` for testing

### Model doesn't improve
- Check logs for reward values (should be mix of +1 and -5)
- Increase `NUM_TRAIN_EPOCHS` from 3 to 5
- Try `LEARNING_RATE = 5e-5` instead of 1e-5

## Key Differences from Existing RL Pipeline

| Aspect | Old RL/ | New rlvr/ |
|--------|---------|-----------|
| Training data | 805 texts (undecided) | 3,715 texts (with labels) |
| Reward signal | Frozen probe P(AI) | Direct correctness check |
| Model output | Explanation | Explanation + Judgment |
| GPU requirement | 4× A100 | 2× A100 |

## Architecture at a Glance

```
Database (text_guess table)
        ↓
    Load unique (text_id, label_truth) pairs
        ↓
    Read text files from disk
        ↓
    Build prompts asking for judgment
        ↓
    Create HF Dataset
        ↓
    GRPO Training Loop:
        - Generate (explanation + judgment)
        - Parse judgment
        - Compare to ground truth
        - Reward: +1 (correct) or -5 (wrong)
        - Update policy via group-relative advantage
        ↓
    Save trained LoRA adapter
        ↓
    Evaluate on test set
        ↓
    Report accuracy metrics
```

## Next Steps

1. Run smoke tests: `sbatch run_smoke_test.sbatch`
2. Run training: `sbatch run_train_grpo.sbatch`
3. Evaluate: `python evaluate_judgment_accuracy.py --checkpoint-path checkpoints/final_adapter`
4. Optional: Fine-tune hyperparameters in `config.py` and retrain

## Links

- Full docs: [README.md](README.md)
- Implementation details: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- Configuration: [config.py](config.py)
