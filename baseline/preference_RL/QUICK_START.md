# Quick Start: LLM Judge + DPO Training

## TL;DR - Running the Full Pipeline

```bash
# Step 1: Generate preferences using LLM judge (NEW)
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL
python rank_explanations_judge_v2.py

# Step 2: Train model using DPO
python train_rl_dpo.py
```

That's it! See below for details.

---

## Step-by-Step Instructions

### Prerequisites ✅
- [x] Explanation pairs JSON: `/gpfs/projects/p32143/Human_RL_baselines/data/explanations_pairs.json` (5.3 MB, 4,520 pairs)
- [x] Text files: `/gpfs/projects/p32143/RL_human_decision/text_data/social/SocialMedia_Reddit/*.txt` and `.../SocialMedia_rewrite/*.txt`
- [x] Inference file: `/gpfs/projects/p32143/Human_RL_baselines/inference_humanRL.json`
- [x] Judge model access: `/projects/p32143/cache/qwen36_27b`
- [ ] GPU availability (2+ GPUs recommended)

### Step 1: Judge Explanation Pairs (Binary Choice)

**File**: `rank_explanations_judge_v2.py`
**Input**: explanation_pairs.json (4,520 pairs)
**Output**: preference_pairs_ai.json, preference_pairs_human.json (~642 total)
**Time**: 2-4 hours

#### What it does:
1. Loads 4,520 explanation pairs
2. Separates AI texts (305) from Human texts (337)
3. For each pair, asks LLM judge: "Which explanation is better?"
4. Creates preference pairs: {chosen, rejected}

#### Run:
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL

# Run the judge script
python rank_explanations_judge_v2.py

# Expected output:
# ✓ Saved 305 AI preference pairs to preference_pairs_ai.json
# ✓ Saved 337 Human preference pairs to preference_pairs_human.json
# Total preference pairs: 642
```

#### Monitor Progress:
```bash
# In another terminal, watch the output
tail -f rank_explanations_judge_v2.py.log

# Or check GPU usage
nvidia-smi -l 1  # Update every 1 second
```

#### Expected Results:
```
AI preference pairs: 305
Human preference pairs: 337
Total preference pairs: 642

Chosen explanation sources (AI texts):
  SocialMedia_rewrite_explanation: ~100
  SocialMedia_rewrite_explanation_claude: ~100
  SocialMedia_rewrite_explanation_gemini: ~105

Chosen explanation sources (Human texts):
  SocialMedia_Reddit_explanation: ~130
  SocialMedia_Reddit_explanation_claude: ~107
  SocialMedia_Reddit_explanation_gemini: ~100
```

---

### Step 2: Train Model with DPO

**File**: `train_rl_dpo.py`
**Input**: preference_pairs_ai.json, preference_pairs_human.json
**Output**: Fine-tuned model at `/projects/p32143/cache/rl_dpo_qwen34b/final_model/`
**Time**: 2-6 hours

#### What it does:
1. Loads preference pairs from judge output
2. Converts to DPO training format
3. Fine-tunes Qwen/Qwen3-4B-Instruct-2507
4. Saves final model with LoRA weights

#### Run:
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL

# Run DPO training
python train_rl_dpo.py

# Expected output:
# Loaded 642 preference pairs
# Created dataset with 642 examples
# Starting DPO training...
# [Training progress...]
# Final model saved to: /projects/p32143/cache/rl_dpo_qwen34b/final_model
```

#### Monitor Training:
```bash
# Watch training loss
tail -f /projects/p32143/cache/rl_dpo_qwen34b/logs/events*

# Check GPU memory
nvidia-smi
# Expected: ~15-20 GB per GPU with batch_size=4

# Monitor training speed
# Expected: ~1-2 examples/sec with 2 GPUs
```

#### Expected Results:
```
Training completed in ~3-4 hours

Final metrics:
- DPO loss: 0.3-0.5 (decreased from initial)
- Training samples: 642
- Epochs completed: 3
- Model saved: /projects/p32143/cache/rl_dpo_qwen34b/final_model/

Checkpoint history:
- checkpoint-200
- checkpoint-400
- final_model
```

---

## File Locations Reference

### Inputs
```
/gpfs/projects/p32143/Human_RL_baselines/
├── data/explanations_pairs.json          ← Judge reads this (4,520 pairs)
└── inference_humanRL.json                ← Gets text labels from this

/gpfs/projects/p32143/RL_human_decision/text_data/social/
├── SocialMedia_Reddit/                   ← Human texts
│   └── reddit_XXXX.txt
└── SocialMedia_rewrite/                  ← AI texts
    └── reddit_XXXX.txt
```

### Outputs (Judge)
```
/gpfs/projects/p32143/RL_human_decision/baseline/preference_RL/
├── preference_pairs_ai.json              ← 305 pairs, save this
└── preference_pairs_human.json           ← 337 pairs, save this
```

### Outputs (DPO Training)
```
/projects/p32143/cache/rl_dpo_qwen34b/
├── checkpoint-200/                       ← Intermediate checkpoint
├── checkpoint-400/                       ← Intermediate checkpoint
├── final_model/                          ← Final trained model ⭐
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
└── logs/                                 ← Training logs
```

---

## Troubleshooting

### Problem: "File not found: explanations_pairs.json"
**Solution**:
```bash
# Check if file exists
ls -lh /gpfs/projects/p32143/Human_RL_baselines/data/explanations_pairs.json

# Should show: -rw-rw-r-- ... 5.3M ... explanations_pairs.json

# If not found, regenerate it:
cd /gpfs/projects/p32143
python generate_explanations_json.py
```

### Problem: "CUDA out of memory"
**Solution**:
1. Reduce parallel GPUs in judge script:
   ```python
   TENSOR_PARALLEL_SIZE = 1  # Instead of 2
   ```

2. Reduce batch size in DPO training:
   ```python
   per_device_train_batch_size=2,  # Instead of 4
   ```

3. Enable gradient offloading in DPO

### Problem: Judge always outputs number 1
**Solution**:
- Check that prompt is being parsed correctly
- Verify `JUDGE_PROMPT_TEMPLATE` is valid
- Check LLM response format in logs
- Try lowering `TEMPERATURE` from 0.5 to 0.3

### Problem: Very few preference pairs generated (<100)
**Solution**:
1. Check text file paths exist:
   ```bash
   ls /gpfs/projects/p32143/RL_human_decision/text_data/social/SocialMedia_Reddit/*.txt | wc -l
   # Should show: ~337
   
   ls /gpfs/projects/p32143/RL_human_decision/text_data/social/SocialMedia_rewrite/*.txt | wc -l
   # Should show: ~305
   ```

2. Check inference file labels:
   ```python
   import json
   with open('/gpfs/projects/p32143/Human_RL_baselines/inference_humanRL.json') as f:
       data = json.load(f)
       print(f"Total records: {len(data)}")
       labels = set(r['label'] for r in data)
       print(f"Labels: {labels}")
   ```

### Problem: DPO training crashes during dataset creation
**Solution**:
- Verify preference pair JSON format:
  ```python
  import json
  with open('preference_pairs_ai.json') as f:
      data = json.load(f)
      sample = list(data.values())[0]
      print(json.dumps(sample, indent=2))
  ```

- Check required keys: `text`, `chosen_explanation`, `rejected_explanations`

---

## What Happens Inside

### Judge Script Flow:
```
1. Load 4,520 explanation pairs
2. Load text labels from inference file
3. Separate into AI (305) and Human (337) pairs
4. For each pair:
   a. Load the actual text
   b. Format judge prompt with 2 explanations
   c. Ask LLM: "Which is better?"
   d. Parse response (1 or 2)
   e. Save preference
5. Output preference_pairs_ai.json and preference_pairs_human.json
```

### DPO Training Flow:
```
1. Load preference pairs (642 total)
2. Format as DPO dataset:
   - prompt: "Analyze this text..."
   - chosen: preferred explanation
   - rejected: non-preferred explanation
3. Load Qwen/Qwen3-4B-Instruct model
4. Setup LoRA fine-tuning
5. Train for 3 epochs:
   - Optimize: log(P_chosen) - log(P_rejected)
   - Add KL penalty (β=0.1)
6. Save fine-tuned model
```

---

## Tips & Tricks

### Speed Up Judge Execution:
```python
# In rank_explanations_judge_v2.py
GPU_MEMORY_UTILIZATION = 0.9  # Use more GPU memory
TEMPERATURE = 0.3             # Lower temperature = faster
MAX_TOKENS = 256              # Reduce output tokens
```

### Reduce DPO Training Time:
```python
# In train_rl_dpo.py
NUM_TRAIN_EPOCHS = 1          # Instead of 3
per_device_train_batch_size = 8  # Larger batch
gradient_accumulation_steps = 1  # Less accumulation
```

### Monitor Resources:
```bash
# Check GPU utilization in real-time
watch -n 1 nvidia-smi

# Check disk space
df -h /projects/p32143/cache/

# Check RAM usage
free -h
```

### Save Bandwidth (if running remote):
```bash
# Judge output is only ~10 MB total
# DPO model is ~10 GB (compressed with LoRA)

# You can compress results:
tar -czf preference_pairs.tar.gz preference_pairs_*.json
```

---

## Expected Timeline

| Step | Time | Status |
|------|------|--------|
| Setup & verification | 10 min | ✅ Now |
| Judge execution | 2-4 hours | ⏳ After setup |
| DPO training | 2-6 hours | ⏳ After judge |
| **Total** | **4-10 hours** | - |

Exact times depend on:
- GPU availability and type
- Text sizes (some texts are very long)
- Model load/cache efficiency

---

## After Training: What's Next?

### 1. Evaluate the Model:
```bash
# Load fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/projects/p32143/cache/rl_dpo_qwen34b/final_model/"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/projects/p32143/cache/rl_dpo_qwen34b/final_model/"
)

# Generate explanation for a test text
input_text = "Your test text here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256)
print(tokenizer.decode(outputs[0]))
```

### 2. Compare with Original:
- Run original Qwen/Qwen3-4B on same text
- Compare explanation quality
- Measure user preference

### 3. Integrate into Pipeline:
- Use fine-tuned model for new explanation generation
- Or run both and use preference data to select better

### 4. Iterate:
- Generate new explanations with fine-tuned model
- Judge again with LLM
- Re-train DPO (continuous improvement)

---

## Summary

```
✅ Prerequisite: Explanation pairs JSON generated
  └─ 4,520 pairs from /gpfs/projects/p32143/Human_RL_baselines/data/

⏳ Step 1: Judge pairs (NEW script)
  └─ Run: python rank_explanations_judge_v2.py
  └─ Output: ~642 preference pairs

⏳ Step 2: Train with DPO (existing script)
  └─ Run: python train_rl_dpo.py
  └─ Output: Fine-tuned model at /projects/p32143/cache/rl_dpo_qwen34b/final_model/

✅ Result: Model trained on human-judged preferences
```

**Ready? Let's go!**
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL
python rank_explanations_judge_v2.py
```
