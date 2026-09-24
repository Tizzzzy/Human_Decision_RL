# Smoke Test Guide

## What Was Fixed

The lazy-loading issue has been **fixed**. Previously:
- `extract_representation()` and `probe_p_ai()` were called before the frozen models were loaded
- This caused: `AttributeError: 'NoneType' object has no attribute 'apply_chat_template'`

**Fix applied**: Both functions now call their respective lazy-load functions to ensure models are loaded before use.

## How to Run the Smoke Test

The smoke test loads Qwen3.6-27B (52GB) and requires a GPU. It will **NOT work on login nodes**.

### Option 1: Run as an Interactive SLURM Job (Recommended)

```bash
# Allocate a single GPU node for 30 minutes
salloc --partition=gengpu --gres=gpu:1 --time=00:30:00 --mem=200G

# Once allocated, run the smoke test:
cd /gpfs/projects/p32143/RL_human_decision/RL/train
source /home/kjj8053/anaconda3/etc/profile.d/conda.sh
conda activate ppo_2
python smoke_test_reward.py
```

### Option 2: Submit as a Batch Job

Create a temporary SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=smoke_test_reward
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=200G
#SBATCH --output=/gpfs/projects/p32143/RL_human_decision/RL/train/logs/smoke_test_%j.out

source /home/kjj8053/anaconda3/etc/profile.d/conda.sh
conda activate ppo_2
cd /gpfs/projects/p32143/RL_human_decision/RL/train
python smoke_test_reward.py
```

Then submit:
```bash
sbatch smoke_test.sbatch
tail -f logs/smoke_test_*.out  # Monitor output
```

## What the Smoke Test Checks

1. **Representation Extraction** — Frozen Qwen3.6-27B extracts [5120] hidden vectors
2. **Probe Ensemble** — 5-model ensemble loads and outputs P(AI) ∈ [0,1]
3. **Reward Formula** — `1 - BCE(p_ai, y)` behaves correctly for various inputs
4. **Eval Prompt Format** — Matches db_to_prompt.py exactly (string integrity check)
5. **Policy Model** (Optional) — Qwen3.6-27B loads via AutoModelForCausalLM and generates

## Expected Runtime

- Model loading: 2-5 minutes (first time only, depends on I/O)
- Tests 1-4: ~1 minute combined
- Test 5 (optional policy generation): 1-2 minutes
- **Total: ~5-10 minutes on a single GPU**

## Failure Scenarios

| Error | Cause | Solution |
|-------|-------|----------|
| Out of memory | Model too large for GPU | Request larger GPU (H100 or multiple GPUs) |
| `FileNotFoundError` on cache | Model cache missing | Check `/projects/p32143/cache/huggingface/qwen36_27b/` exists |
| `FileNotFoundError` on ensemble | Ensemble weights missing | Run Task 1 to generate `trained_ensemble.pth` |
| Timeout (180s) | GPU too busy | Submit job and wait for allocation |
| CUDA error | GPU driver issues | Switch to a different GPU node |

## After Smoke Test Passes ✓

Once all 5 tests pass, you can safely run the full RL training:

```bash
sbatch /gpfs/projects/p32143/RL_human_decision/RL/train/run_train_grpo.sbatch
```

## Key Configuration Updates

As of 2026-09-07, the following config has been updated:
- `MAX_COMPLETION_LENGTH`: 512 (was 256, accommodates longer explanations)
- `MAX_PROMPT_LENGTH`: 5096 (was 1024, handles larger texts)
- Probe path: `simulator/trained_ensemble.pth` (5-model ensemble, not single model)

These match the improvements made during your model training.
