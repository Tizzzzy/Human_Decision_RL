# Inference Guide: Generate New Explanations

## Overview

After completing GRPO RL training, use the trained policy model to generate new AI-detection explanations for a set of texts.

**Input**: `/gpfs/projects/p32143/RL_human_decision/RL/get_new_explanation/inference_humanRL.json`  
**Output**: `/gpfs/projects/p32143/RL_human_decision/RL/get_new_explanation/inference_with_new_explanations.json`

## Quick Start

### Option 1: Batch Job (Recommended)

```bash
cd /gpfs/projects/p32143/RL_human_decision/RL
sbatch run_inference_generate_explanations.sbatch

# Monitor progress
tail -f logs/<job_id>.out
```

### Option 2: Interactive Job

```bash
# Allocate GPU for 1 hour
salloc --partition=gengpu --gres=gpu:a100:2 --time=01:00:00 --mem=200G

# Run inference
cd /gpfs/projects/p32143/RL_human_decision/RL
conda activate ppo_2
python inference_generate_explanations.py
```

## What the Script Does

1. **Loads input data** from `inference_humanRL.json` (records with text, original explanations)
2. **Loads trained policy** (Qwen3.6-27B + LoRA adapter from `final_adapter/`)
3. **For each text**:
   - Builds a policy prompt asking the model to explain linguistic markers
   - Generates a new explanation (max 256 tokens)
   - Stores both original and new explanations
4. **Saves results** to `inference_with_new_explanations.json`

## Output Format

Each record in the output JSON will include:
```json
{
  "text_id": "reddit_2932",
  "label": "Human",
  "text": "...",
  "text_path": "SocialMedia_Reddit/reddit_2932.txt",
  "explanation": "... original explanation ...",
  "new_explanation": "... newly generated explanation ..."
}
```

## Generation Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_NEW_TOKENS` | 256 | Limit explanation length |
| `TEMPERATURE` | 1.0 | Sampling temperature (1.0 = moderate randomness) |
| `TOP_P` | 1.0 | Nucleus sampling (1.0 = no restriction) |
| `REPETITION_PENALTY` | 1.1 | Penalize repeated tokens |
| `DO_SAMPLE` | True | Use sampling (vs greedy decoding) |

Adjust these in `inference_generate_explanations.py` if you want different behavior.

## Expected Runtime

- Model loading: 2-5 minutes (first time)
- Generation: ~5-10 seconds per text (with 2x A100)
- For ~200 texts: ~20-30 minutes total
- For ~1000 texts: ~1.5-2.5 hours

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use 4x GPU allocation or reduce batch processing |
| Model not found | Check that `/projects/p32143/cache/huggingface/qwen36_27b/` exists |
| LoRA adapter not found | Ensure `checkpoints/final_adapter/` exists and contains `adapter_model.safetensors` |
| Slow generation | Check GPU utilization; may need to adjust `device_map` |

## Next Steps After Inference

1. **Evaluate new explanations** — Compare against original explanations via probe scoring
2. **A/B test** — Use new explanations in human studies or with downstream models
3. **Iterate** — If results are good, can fine-tune further or extend to more texts

## Sample Command

```bash
# Quick test on a single GPU (1 hour timeout)
sbatch --gres=gpu:a100:1 --time=01:00:00 run_inference_generate_explanations.sbatch
```

## Files

- **Main script**: `inference_generate_explanations.py`
- **SLURM script**: `run_inference_generate_explanations.sbatch`
- **Input data**: `get_new_explanation/inference_humanRL.json`
- **Output data**: `get_new_explanation/inference_with_new_explanations.json`
