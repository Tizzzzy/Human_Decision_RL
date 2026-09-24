# Preference-based RL Training Workflow

## Overview

This workflow uses an LLM judge to select the better explanation from pairs, then trains a model using DPO (Direct Preference Optimization) based on these preferences.

## Workflow Steps

### Step 1: Generate Explanation Pairs
**Location**: `/gpfs/projects/p32143/Human_RL_baselines/data/explanations_pairs.json`
**Status**: ✅ Already Generated

4,520 explanation pairs have been created with 2 explanations per text:
- 674 pairs for Human texts (337 texts × 2)
- 610 pairs for AI texts (305 texts × 2)

Each pair contains:
```json
{
  "reddit_1001_AI.txt": {
    "explanation_1": {
      "content": "Explanation text...",
      "source_path": "SocialMedia_Reddit_explanation/reddit_1001.txt"
    },
    "explanation_2": {
      "content": "Alternative explanation...",
      "source_path": "SocialMedia_Reddit_explanation_claude/reddit_1001.txt"
    }
  }
}
```

### Step 2: Judge Explanation Pairs ⚙️ DO THIS NEXT
**Script**: `rank_explanations_judge_v2.py` (NEW)
**Purpose**: Use LLM judge to select the better explanation from each pair

#### What the Script Does:
1. Loads the 4,520 explanation pairs
2. For each pair, asks Qwen/Qwen3.6-27B: "Which explanation is better?"
3. Judge selects one of the two explanations
4. Creates preference pairs: `{chosen_explanation, rejected_explanation}`
5. Outputs:
   - `preference_pairs_ai.json` - AI text preferences
   - `preference_pairs_human.json` - Human text preferences

#### Running the Script:
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL

python rank_explanations_judge_v2.py
```

#### Output Format:
```json
{
  "reddit_1001": {
    "text": "The original text...",
    "chosen_explanation": "The explanation the judge selected...",
    "chosen_source": "SocialMedia_Reddit_explanation/reddit_1001.txt",
    "rejected_explanations": [
      {
        "source": "SocialMedia_Reddit_explanation_claude/reddit_1001.txt",
        "text": "The rejected explanation..."
      }
    ]
  }
}
```

**Expected Output**:
- AI preference pairs: ~305
- Human preference pairs: ~337
- Total: ~642 preference pairs

**Time Estimate**: 2-4 hours (depends on GPU availability)

### Step 3: Train Model with DPO
**Script**: `train_rl_dpo.py` (existing, works with judge output)
**Purpose**: Fine-tune a model to generate better explanations using the judged preferences

#### What the Script Does:
1. Loads preference pairs from Step 2
2. Converts them to DPO training format
3. Trains Qwen/Qwen3-4B-Instruct-2507 using DPO
4. Saves fine-tuned model to cache directory

#### Running the Script:
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL

python train_rl_dpo.py
```

#### Training Configuration:
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Method**: DPO (Direct Preference Optimization)
- **Epochs**: 3
- **Learning Rate**: 5e-5
- **Beta (KL penalty)**: 0.1
- **LoRA Rank**: 16

**Time Estimate**: 2-6 hours (depends on GPU availability and dataset size)

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│ explanations_pairs.json (4,520 pairs)              │
│ ├─ AI texts: 610 explanations (305 texts × 2)     │
│ └─ Human texts: 674 explanations (337 texts × 2)  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: LLM Judge (Qwen/Qwen3.6-27B)               │
│ rank_explanations_judge_v2.py                       │
│                                                      │
│ "Which explanation is better?"                      │
│ • Compares 2 explanations per text                 │
│ • Outputs preference: chosen vs rejected           │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Preference Pairs (from judge)                       │
│ ├─ preference_pairs_ai.json (~305 pairs)           │
│ └─ preference_pairs_human.json (~337 pairs)        │
│ Total: ~642 preference pairs                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: DPO Training                                │
│ train_rl_dpo.py                                     │
│                                                      │
│ Model: Qwen/Qwen3-4B-Instruct-2507                 │
│ • Learn to generate chosen explanations            │
│ • Avoid rejected explanations                      │
│ • 3 epochs, LoRA fine-tuning                       │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Fine-tuned Model                                    │
│ Output: /projects/p32143/cache/rl_dpo_qwen34b/     │
│         final_model/                                │
│                                                      │
│ Ready to generate better explanations!             │
└─────────────────────────────────────────────────────┘
```

## Key Differences from Original Script

### Original `rank_explanations_judge.py`:
- Selects best from **6 explanations** per text
- Works with Gemma-4 generated explanations
- Outputs (chosen, [rejected1, rejected2, ...])

### New `rank_explanations_judge_v2.py`:
- Selects best from **2 explanations** per text
- Works with the explanation pairs JSON you generated
- Outputs (chosen, [rejected])
- **Simpler judge prompt** (binary choice vs 6-way)
- **Faster execution** (fewer comparisons)
- **Integrates seamlessly** with the DPO training

## Files Location Reference

```
/gpfs/projects/p32143/
├── Human_RL_baselines/                           ← Explanation pairs
│   └── data/
│       └── explanations_pairs.json                (INPUT)
│
├── RL_human_decision/
│   ├── baseline/
│   │   └── preference_RL/
│   │       ├── rank_explanations_judge.py        (original - uses 6 explanations)
│   │       ├── rank_explanations_judge_v2.py     (NEW - uses 2 explanations)
│   │       ├── train_rl_dpo.py                   (DPO training)
│   │       ├── preference_pairs_ai.json          (OUTPUT from judge)
│   │       ├── preference_pairs_human.json       (OUTPUT from judge)
│   │       └── PREFERENCE_RL_WORKFLOW.md         (this file)
│   │
│   └── text_data/
│       └── social/
│           ├── SocialMedia_Reddit/               (Human texts)
│           └── SocialMedia_rewrite/              (AI texts)
│
└── cache/
    └── rl_dpo_qwen34b/
        └── final_model/                          (TRAINED MODEL OUTPUT)
```

## Configuration Parameters

### Judge Script (`rank_explanations_judge_v2.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_PATH` | `/projects/p32143/cache/qwen36_27b` | Judge model location |
| `TENSOR_PARALLEL_SIZE` | 2 | GPU parallel processes |
| `GPU_MEMORY_UTILIZATION` | 0.85 | Max GPU memory usage |
| `TEMPERATURE` | 0.5 | Response diversity (lower = more focused) |
| `TOP_P` | 0.95 | Nucleus sampling threshold |
| `MAX_TOKENS` | 1024 | Max output length |

### DPO Training Script (`train_rl_dpo.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_NAME` | Qwen/Qwen3-4B-Instruct-2507 | Model to fine-tune |
| `LEARNING_RATE` | 5e-5 | Training step size |
| `NUM_TRAIN_EPOCHS` | 3 | How many times to go through data |
| `BETA` | 0.1 | KL divergence penalty weight |
| `MAX_LENGTH` | 512 | Max tokens in training example |

## Monitoring & Troubleshooting

### During Judge Execution:
```bash
# Watch progress
watch -n 5 'tail -20 /var/log/gpu_utilization.log'

# Expected output pattern:
# 📝  reddit_1001: Judge response: 1
# ✓ Saved 305 AI preference pairs...
# ✓ Saved 337 Human preference pairs...
```

### During DPO Training:
```bash
# Monitor training
tail -f /projects/p32143/cache/rl_dpo_qwen34b/logs/events*

# Check GPU usage
nvidia-smi

# Expected: Model should be training, loss should decrease
```

### Common Issues:

| Issue | Cause | Solution |
|-------|-------|----------|
| "File not found" error | Wrong path to explanations_pairs.json | Verify path is `/gpfs/projects/p32143/Human_RL_baselines/data/explanations_pairs.json` |
| "CUDA out of memory" | GPU insufficient | Reduce `TENSOR_PARALLEL_SIZE` or enable CPU offloading |
| Judge always picks 1 | Parse error in response | Check `JUDGE_PROMPT_TEMPLATE` format |
| Low preference pair count | Text file read errors | Verify text files exist in correct directories |
| DPO training crashes | Dataset format issue | Check preference_pairs_*.json format matches expected |

## Expected Results

### After Judge Execution:
- Files: `preference_pairs_ai.json` and `preference_pairs_human.json`
- Sample statistics:
  ```
  AI preferences: ~305
  Human preferences: ~337
  Total: ~642
  
  Most selected explanation sources:
  - SocialMedia_Reddit_explanation: 45%
  - SocialMedia_Reddit_explanation_claude: 35%
  - SocialMedia_Reddit_explanation_gemini: 20%
  ```

### After DPO Training:
- Fine-tuned model at: `/projects/p32143/cache/rl_dpo_qwen34b/final_model/`
- Training metrics:
  - DPO loss: should decrease over epochs
  - Accuracy on preference pairs: should approach 80-90%
  - Training time: 2-6 hours

## Next Steps After Training

1. **Evaluate**: Test the fine-tuned model on held-out texts
2. **Compare**: Compare with original model explanations
3. **Iterate**: Generate more preferences, refine model
4. **Deploy**: Use fine-tuned model in production

## Workflow Comparison: Old vs New

### Old Workflow (Original Script)
```
6 pre-generated explanations per text
    ↓
Judge selects best of 6
    ↓
Create preference pairs
    ↓
Train model
```

### New Workflow (With Your Data)
```
explanations_pairs.json (2 explanations per text)
    ↓
Judge compares 2 explanations
    ↓
Create preference pairs
    ↓
Train model (DPO)
```

**Key Benefit**: More focused preferences that directly train the model to choose between similar explanations, which is more realistic for RL training.

## Summary

1. ✅ **Step 1 Complete**: Explanation pairs JSON generated (4,520 pairs)
2. ⚙️ **Step 2 Ready**: Judge the pairs using `rank_explanations_judge_v2.py`
3. ⏳ **Step 3 Waiting**: Train model using `train_rl_dpo.py`

Run Step 2 now to generate preference pairs, then Step 3 to train the model!

```bash
# Step 2: Judge explanations
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL
python rank_explanations_judge_v2.py

# Step 3: Train with DPO
python train_rl_dpo.py
```
