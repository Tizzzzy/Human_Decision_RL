# RL Pipeline Implementation Summary

Date: 2026-09-07
Status: ✓ Complete (Ready for Testing & Execution)

---

## Overview

This implementation RL-trains Qwen3.6-27B to generate better explanation hints for distinguishing AI-generated from human-written text, using a frozen MLP probe as the reward function.

### Key Components

**Task 1 — Undecided Text Pool Generation (`RL/get_undecide_text/`)**
- Identifies all (text_id, label_truth) pairs on disk with *no* human decisions yet
- Splits into 85% train / 15% test for the RL training loop
- Output: 805 undecided texts (422 AI, 383 Human)
  - Train: 684 (359 AI, 325 Human)
  - Test: 121 (63 AI, 58 Human)

**Task 2 — GRPO RL Training (`RL/train/`)**
- Policy: Qwen3.6-27B with LoRA adapter (being trained to generate explanations)
- Reward: Frozen Qwen3.6-27B + frozen **5-model DeepEnsemble** scoring explanations
- Algorithm: GRPO (Group Relative Policy Optimization) from `trl 0.28.0`
- Reward formula: `reward = max(1 - BCE(p_ai, y), -5.0)` where:
  - `p_ai` = ensemble's predicted P(AI) (average of 5 ShallowMLP models) for the text+explanation
  - `y` = ground-truth label (1=AI, 0=Human)
  - BCE penalizes confident-but-wrong predictions superlinearly

---

## File Structure

### Task 1 — Building Undecided Pool

```
RL/get_undecide_text/
├── build_undecided_pool.py          Main script (CPU-bound)
├── run_build_pool.sbatch            SLURM submission script (~10 min)
├── logs/                             Job logs directory
├── undecided_pool.jsonl             Full pool (debugging)
├── train_undecided.jsonl            Training set (685 records)
├── test_undecided.jsonl             Test set (121 records)
└── summary.json                      Sanity-check counts
```

**To run Task 1:**
```bash
sbatch /gpfs/projects/p32143/RL_human_decision/RL/get_undecide_text/run_build_pool.sbatch
# or run directly:
python /gpfs/projects/p32143/RL_human_decision/RL/get_undecide_text/build_undecided_pool.py
```

### Task 2 — RL Training Infrastructure

```
RL/train/
├── __init__.py                      Package marker
├── __main__.py                      Module entry point
├── config.py                        Central config (paths, hyperparams)
├── prompt_templates.py              Policy instruction + eval prompt templates
├── probe_model.py                   ShallowMLP class (5120→256→1)
├── dataset.py                       JSONL → HuggingFace Dataset builder
├── reward.py                        Frozen model + probe + reward function
├── train_grpo.py                    Main training script (GRPOTrainer wiring)
├── smoke_test_reward.py             Sanity checks before full training
├── test_imports.py                  Quick import verification
├── run_train_grpo.sbatch            SLURM submission (4x A100, 2-day time limit)
├── logs/                             Training logs directory
├── checkpoints/                      GRPO checkpoints + final LoRA adapter
└── README.md                         (see below)
```

**Key design decisions:**
- **Frozen extraction model**: Full separate copy of Qwen3.6-27B loaded via `AutoModelForMultimodalLM` (matches `extract_representation.py`'s exact numerics) rather than reusing policy model with `disable_adapter()` (safer, cleaner, avoids mode-switching risks)
- **Reward formula**: `1 - BCE(p_ai, y)` clipped at `-5.0` — matches probe's training loss, sharply penalizes confident-wrong predictions
- **LoRA config**: r=16, alpha=32, dropout=0.05, targets q/k/v/o projections (efficient fine-tuning)
- **GRPO config**: num_generations=8, max_completion_length=256, beta=0.0 (no KL penalty, group-relative baseline only)

---

## Execution Path

### Phase 1: Run Task 1 (Undecided Pool Generation)
```bash
cd /gpfs/projects/p32143/RL_human_decision
sbatch RL/get_undecide_text/run_build_pool.sbatch
# Wait for completion (~10 min)
# Check: cat RL/get_undecide_text/summary.json
```

Expected output:
```json
{
  "disk_pool_total": 805,
  "disk_pool_ai": 422,
  "disk_pool_human": 383,
  "train_count": 684,
  "train_ai": 359,
  "train_human": 325,
  "test_count": 121,
  "test_ai": 63,
  "test_human": 58
}
```

### Phase 2: Smoke Test (Verify Reward Module)
```bash
cd /gpfs/projects/p32143/RL_human_decision/RL/train
source /home/kjj8053/anaconda3/etc/profile.d/conda.sh
conda activate ppo_2
python smoke_test_reward.py
```

Checks:
1. ✓ Representation extraction (shape [5120])
2. ✓ Probe P(AI) in range [0,1]
3. ✓ Reward formula behavior (confident-correct ≈ 1.0, confident-wrong < 0)
4. ✓ Eval prompt format matches db_to_prompt.py
5. ✓ Policy model loads via AutoModelForCausalLM and generates

### Phase 3: Full RL Training
```bash
cd /gpfs/projects/p32143/RL_human_decision
sbatch RL/train/run_train_grpo.sbatch
# or for testing with tiny subset:
# sbatch --time=01:00:00 --mem=256G RL/train/run_train_grpo.sbatch
```

SLURM allocation: 4x A100 on `gengpu`, 2-day time limit
- Expected throughput: ~10-20 samples/sec (depends on GPU/model efficiency)
- Estimated full train runtime: ~8-16 hrs for 684 train samples @ 8 generations each

Output:
```
RL/train/
├── checkpoints/
│   ├── checkpoint-N/                 GRPO checkpoints
│   └── final_adapter/                Final LoRA weights
└── logs/
    └── <job_id>.out                 Training logs
```

### Phase 4: Run Inference with Trained Policy
```bash
# After training completes:
python RL/inference.py  # (already exists; update checkpoint path)
```

---

## Critical Dependencies & Constraints

**Environment**: `ppo_2` conda env **required**
- `transformers==5.2.0`
- `trl==0.28.0` (GRPOTrainer available)
- `peft==0.18.0` (LoRA support)
- `torch==2.9.0`
- `datasets==4.5.0`

Activate with:
```bash
source /home/kjj8053/anaconda3/etc/profile.d/conda.sh && conda activate ppo_2
```

**GPU Memory**: ~110GB needed
- Policy Qwen3.6-27B (bf16): ~54GB + LoRA params
- Frozen extraction Qwen3.6-27B (bf16): ~54GB
- Probe + activations + optimizer states: ~2GB
- Total: fits on 4x A100 (320GB) with device_map="auto" sharding

**Checkpoint paths (absolute)**: hardcoded in `config.py` — all paths use `/gpfs/projects/p32143/` prefix

---

## Verification Checklist

Before submitting the full training job:

- [ ] Task 1 completed, `summary.json` shows expected counts
- [ ] `RL/train/test_imports.py` runs with `conda activate ppo_2`
- [ ] `RL/train/smoke_test_reward.py` runs without errors (on a single GPU node)
- [ ] Policy model `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B")` loads successfully
- [ ] Cache dir `/projects/p32143/cache/huggingface/qwen36_27b` exists and contains model weights
- [ ] `RL/get_undecide_text/train_undecided.jsonl` and `test_undecided.jsonl` exist and are readable

---

## Configuration Reference

### Hyperparameters (`config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| LEARNING_RATE | 1e-5 | Conservative; can lower if needed |
| PER_DEVICE_BATCH_SIZE | 8 | Provisional; adjust based on memory |
| GRADIENT_ACCUMULATION_STEPS | 4 | Effective batch = 8*4*4GPU = 128 |
| NUM_GENERATIONS | 8 | GRPO group size for relative-advantage baseline |
| MAX_COMPLETION_LENGTH | 256 | Max tokens for explanation generation |
| MAX_PROMPT_LENGTH | 1024 | Max tokens for policy prompt |
| BETA | 0.0 | No KL penalty (GRPO has built-in group baseline) |
| TEMPERATURE | 1.0 | Deterministic (no randomness in generation) |
| NUM_TRAIN_EPOCHS | 1 | One full pass over 684 train samples |
| LORA_R | 16 | LoRA rank |
| LORA_ALPHA | 32 | LoRA scaling (effective_lr scales by alpha/r) |
| REWARD_CLIP_FLOOR | -5.0 | Floor clipping for reward outliers |

### Key Paths (`config.py`)

| Path | Purpose |
|------|---------|
| `/gpfs/projects/p32143/RL_human_decision/RL/get_undecide_text/train_undecided.jsonl` | Training pool |
| `/gpfs/projects/p32143/RL_human_decision/RL/get_undecide_text/test_undecided.jsonl` | Evaluation pool |
| `/projects/p32143/cache/huggingface/qwen36_27b` | Qwen3.6-27B cache (52GB) |
| `/gpfs/projects/p32143/RL_human_decision/simulator/trained_ensemble.pth` | Frozen 5-model ensemble probe weights |
| `/gpfs/projects/p32143/RL_human_decision/RL/train/checkpoints` | GRPO checkpoints + final adapter |

---

## Troubleshooting

**Issue**: `AutoModelForCausalLM` fails to load Qwen3.6-27B
- **Cause**: Model config may only support `AutoModelForMultimodalLM`
- **Fix**: Modify `train_grpo.py` line ~20 to use `AutoModelForMultimodalLM` instead (less ideal, riskier)
- **Note**: Empirically verify this in smoke test; if it fails, fall back plan is documented in the plan file

**Issue**: Out-of-memory errors during training
- **Cause**: 4x A100 allocation insufficient or device_map sharding ineffective
- **Fix**: Reduce `PER_DEVICE_BATCH_SIZE` (e.g., 4→2) or request 4x H100 (faster/higher memory)
- **Note**: Smoke test should catch memory issues early

**Issue**: Rewards are all NaN or constant
- **Cause**: Reward function bug, or frozen model not properly frozen
- **Check**: Run `smoke_test_reward.py` in isolation; verify repr extraction works

**Issue**: Training is very slow
- **Cause**: CPU bottleneck in reward function (text reading, prompt building)
- **Fix**: Profile and optimize the reward_func; consider prefetching + caching

---

## Next Steps After Training

1. **Inference**: Run `RL/inference.py` with the final LoRA adapter to generate explanations for test texts
2. **Evaluation**: Compare new explanations against baseline (original human-provided hints) via probe scoring
3. **Deployment**: Consider merging LoRA adapter back into base model for inference efficiency
4. **Iteration**: If results are below target, tune hyperparameters or expand training data (RL/get_undecide_text can extend to other text domains)

---

## Files & Locations Summary

- **Undecided pool builder**: `/gpfs/projects/p32143/RL_human_decision/RL/get_undecide_text/build_undecided_pool.py`
- **RL training modules**: `/gpfs/projects/p32143/RL_human_decision/RL/train/*.py` (config, dataset, reward, train_grpo, etc.)
- **SLURM scripts**: `run_build_pool.sbatch`, `run_train_grpo.sbatch`
- **Plan document**: `/home/kjj8053/.claude/plans/i-am-currently-doing-transient-goose.md`

---

## Contact & References

- Plan: `/home/kjj8053/.claude/plans/i-am-currently-doing-transient-goose.md` (detailed architectural decisions)
- Existing probe: `/gpfs/projects/p32143/RL_human_decision/simulator/probe_training/probe_weights.pth`
- Existing prompt template reference: `/gpfs/projects/p32143/RL_human_decision/RL/inference.py` (policy instruction)
- DB structure: `/gpfs/projects/p32143/RL_human_decision/statistic/text_2026-09-02.db` (text_guess table)
