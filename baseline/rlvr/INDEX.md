# File Index: Verifiable-Reward RL Pipeline

## Read These First
- **[00_START_HERE.md](00_START_HERE.md)** — Complete overview, quick start, key stats
- **[QUICKSTART.md](QUICKSTART.md)** — 30-second overview + copy-paste commands

## Documentation
- **[README.md](README.md)** — Full architecture, config guide, troubleshooting
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** — Design rationale, detailed workflow
- **[INDEX.md](INDEX.md)** — This file

## Source Code

### Core Training Pipeline
- **[config.py](config.py)** — Hyperparameters, paths, configuration (100+ lines)
- **[prompt_templates.py](prompt_templates.py)** — Build prompts, parse judgments (120+ lines)
- **[dataset.py](dataset.py)** — Load from DB, build HF Dataset (190+ lines)
- **[reward.py](reward.py)** — Compute verifiable rewards (85+ lines)
- **[train_grpo_verifiable.py](train_grpo_verifiable.py)** — Main GRPO training loop (200+ lines)

### Validation & Evaluation
- **[smoke_test.py](smoke_test.py)** — Pre-training validation tests (340+ lines)
- **[evaluate_judgment_accuracy.py](evaluate_judgment_accuracy.py)** — Post-training evaluation (290+ lines)

### SLURM Scripts
- **[run_smoke_test.sbatch](run_smoke_test.sbatch)** — Submit smoke tests to GPU
- **[run_train_grpo.sbatch](run_train_grpo.sbatch)** — Submit full training job

## Directory Structure After First Run

```
rlvr/
├── checkpoints/
│   ├── final_adapter/          # Trained LoRA weights (main output)
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   ├── config.json
│   │   └── ...
│   └── logs/
│       ├── <job_id>.out        # Training logs
│       ├── <job_id>.err        # Error logs
│       └── ...
├── config.py
├── dataset.py
├── evaluate_judgment_accuracy.py
├── prompt_templates.py
├── reward.py
├── run_smoke_test.sbatch
├── run_train_grpo.sbatch
├── smoke_test.py
├── train_grpo_verifiable.py
├── 00_START_HERE.md
├── IMPLEMENTATION_GUIDE.md
├── QUICKSTART.md
├── README.md
└── INDEX.md
```

## Workflow Map

```
Quick Start
├─ Read 00_START_HERE.md
├─ Run smoke_test.py
│  └─ All 5 tests pass?
│     ├─ YES: Continue
│     └─ NO: Check README.md troubleshooting
└─ Run training
   ├─ sbatch run_train_grpo.sbatch
   └─ Monitor: tail -f logs/<job_id>.out

After Training
├─ Check checkpoints/final_adapter/ exists
├─ Run evaluate_judgment_accuracy.py
└─ Review metrics

Configuration Tuning
├─ Edit config.py
├─ Rerun training
└─ Compare results
```

## Key Files by Purpose

### "I want to..."

**...understand what this does**
→ Start with [00_START_HERE.md](00_START_HERE.md)

**...run it right now**
→ See [QUICKSTART.md](QUICKSTART.md)

**...change the prompt format**
→ Edit [prompt_templates.py](prompt_templates.py)

**...change hyperparameters**
→ Edit [config.py](config.py)

**...understand the architecture**
→ Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

**...debug an error**
→ Check [README.md](README.md) troubleshooting section

**...modify reward logic**
→ Edit [reward.py](reward.py)

**...evaluate trained model**
→ Run [evaluate_judgment_accuracy.py](evaluate_judgment_accuracy.py)

**...check if everything works**
→ Run [smoke_test.py](smoke_test.py)

## Smoke Test Results

All 5 smoke tests pass:

```
✓ PASS: Prompt Parsing
✓ PASS: Reward Computation
✓ PASS: Dataset Loading (3,715 texts loaded)
✓ PASS: Prompt Building
✓ PASS: Model Loading (skips on login node)
```

## File Sizes

```
Documentation:
  00_START_HERE.md                   5.2 KB
  QUICKSTART.md                      4.0 KB
  README.md                          8.5 KB
  IMPLEMENTATION_GUIDE.md            7.6 KB
  INDEX.md                           2.0 KB (this file)

Code:
  config.py                          2.7 KB
  dataset.py                         5.5 KB
  prompt_templates.py                3.3 KB
  reward.py                          2.9 KB
  train_grpo_verifiable.py           6.6 KB
  smoke_test.py                      9.1 KB
  evaluate_judgment_accuracy.py      8.1 KB

Scripts:
  run_smoke_test.sbatch              1.3 KB
  run_train_grpo.sbatch              1.3 KB

Total: ~67 KB
```

## Quick Reference

### Training Timeline

| Phase | Command | Duration | Output |
|-------|---------|----------|--------|
| Validation | `python smoke_test.py` | 5 min | ✓ All pass |
| Training | `sbatch run_train_grpo.sbatch` | 1.5-2.5 hrs | `checkpoints/final_adapter/` |
| Evaluation | `python evaluate_judgment_accuracy.py` | 30 min | Accuracy, precision, recall, F1 |

### GPU Requirements

- Minimum: 1× A100 (testing)
- Recommended: 2× A100 (training, ~2 hours)
- Faster: 4× A100 (training, ~1 hour)

### Data Summary

- Source: `text_guess` table
- Size: 3,715 unique texts
- Classes: AI (1,838) + Human (1,877)
- Split: 85% train (3,159) / 15% test (556)

---

**Start here:** [00_START_HERE.md](00_START_HERE.md)
