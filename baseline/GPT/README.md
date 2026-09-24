# GPT Inference for Social Media Post Analysis

## Overview

This directory contains scripts to generate explanations for social media posts using OpenAI's GPT models. The explanations help determine whether a post was written by AI or a human.

**Key Feature**: Output format is identical to `preference_RL/inference.py` for easy comparison.

## What's Included

### Scripts
- **`gpt_inference.py`** - Main inference script using OpenAI API
- **`compare_explanations.py`** - Compare GPT output with preference_RL output

### Documentation
- **`QUICK_START_GPT.md`** - Fast setup (5 minutes)
- **`GPT_INFERENCE_GUIDE.md`** - Complete reference guide
- **`COMPARISON_WITH_PREFERENCE_RL.md`** - Detailed comparison

## Quick Start (5 minutes)

### 1. Setup
```bash
# Install OpenAI package
pip install openai

# Set API key
export OPENAI_API_KEY='sk-your-key-here'
```

### 2. Run
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python gpt_inference.py
```

### 3. Check Results
```bash
ls -lh inference_results/
cat inference_results/inference_gpt_gpt-4o_*.json | head -20
```

**Time**: ~10-15 minutes for 642 texts  
**Cost**: ~$0.30-$6 depending on model

## Features

✅ **Compatible with Preference_RL** - Same prompt, same output format  
✅ **Multiple Models** - Test gpt-4o, gpt-4, gpt-3.5-turbo  
✅ **Customizable** - Adjust temperature, max_tokens, etc.  
✅ **Rate Limited** - Automatic delays to avoid API throttling  
✅ **Well Documented** - Complete guides for all scenarios  
✅ **Comparison Ready** - Built-in comparison with other models  

## Models Available

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `gpt-3.5-turbo` | Fast | Good | Low (~$0.30) |
| `gpt-4o` | Medium | Excellent | Medium (~$1.20) |
| `gpt-4` | Slow | Best | High (~$6.00) |

## How It Works

```
Input CSV (test_simulator.csv)
    ↓ Reads text_id, text_path, label
    ↓
Load Text Files
    ↓ From SocialMedia_Reddit/ and SocialMedia_rewrite/
    ↓
Generate Explanations
    ↓ Call OpenAI GPT API with prompt
    ↓ 642 texts × ~1 sec/text ≈ 10-15 minutes
    ↓
Save Results JSON
    ↓ Same format as preference_RL/inference.py
    ↓
inference_gpt_gpt-4o_20240806_143022.json
```

## Configuration

Edit these variables in `gpt_inference.py`:

```python
GPT_MODEL = "gpt-4o"              # Model to use
TEMPERATURE = 1.0                  # 0=deterministic, 2=random
MAX_TOKENS = 200                   # Max explanation length
RATE_LIMIT_DELAY = 0.5            # Seconds between API calls
```

## Output Format

### File Location
```
inference_results/inference_gpt_gpt-4o_20240806_143022.json
```

### JSON Structure
```json
[
  {
    "text_id": "reddit_2932",
    "label": "Human",
    "text": "The actual social media post...",
    "text_path": "SocialMedia_Reddit/reddit_2932.txt",
    "explanation": "- Look for conversational tone.\n- Look for personal anecdotes like 'I remember when'..."
  },
  ...
]
```

## Explanation Prompt

The script uses this prompt (identical to preference_RL):

```
Task: Analyze the provided Social Media post for linguistic markers of AI or human authorship.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list.
2. Do NOT provide an introduction, a final verdict, or a summary.
3. Do NOT group the bullet points into sub-headings (keep it a flat list).
4. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category or type of phrasing]."
    - "Look for [linguistic category or type of phrasing] like '[short example]'."
5. When providing examples, you may list multiple words separated by commas.
6. Do NOT use bolding, italics, or any special Markdown formatting.
7. Limit the output to exactly 2 to 6 bullet points.
```

## Comparing with Preference_RL

### Generate Both Explanations
```bash
# 1. Generate GPT explanations
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python gpt_inference.py

# 2. Generate preference_RL explanations
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL
python inference.py

# 3. Compare
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python compare_explanations.py
```

### Key Differences
| Aspect | GPT | Preference_RL |
|--------|-----|---------------|
| **Model Training** | Pre-trained on web | Fine-tuned on preferences |
| **Access** | API (requires key) | Local (requires GPU) |
| **Speed** | ~1 sec/text | ~0.5 sec/text |
| **Cost** | $0.30-6 | Free |
| **Reproducibility** | May change | Fixed |
| **Customization** | Prompt engineering | Full fine-tuning |

See `COMPARISON_WITH_PREFERENCE_RL.md` for detailed analysis.

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### "Rate limit exceeded"
Increase `RATE_LIMIT_DELAY` in script:
```python
RATE_LIMIT_DELAY = 2.0  # Wait 2 seconds between requests
```

### "Model not available"
Check available models on OpenAI dashboard. Ensure API key has access.

### "API timeout"
Check internet connection or try again later.

## Use Cases

### 1. Quick Baseline
Generate explanations to understand what different models produce:
```bash
python gpt_inference.py
```

### 2. Benchmark Comparison
Compare GPT performance against your fine-tuned model:
```bash
python compare_explanations.py
```

### 3. New Test Data
Generate explanations for newly collected posts:
```python
# Modify test CSV path in script
TEST_CSV = "path/to/new_test_data.csv"
python gpt_inference.py
```

### 4. Model Selection
Test different GPT models to find best quality/cost tradeoff:
```bash
# Edit GPT_MODEL for each run
python gpt_inference.py  # gpt-4o (default)
python gpt_inference.py  # gpt-4 (best quality)
python gpt_inference.py  # gpt-3.5-turbo (cheapest)
```

## Cost Estimation

For 642 texts:

| Model | Est. Cost | Duration | Per Text |
|-------|-----------|----------|----------|
| gpt-3.5-turbo | ~$0.30 | 10-12 min | ~$0.0005 |
| gpt-4o | ~$1.20 | 12-15 min | ~$0.002 |
| gpt-4 | ~$6.00 | 20-30 min | ~$0.01 |

Monitor actual costs: https://platform.openai.com/account/usage/overview

## File Structure

```
/gpfs/projects/p32143/RL_human_decision/baseline/GPT/
├── gpt_inference.py                          ← Main script ⭐
├── compare_explanations.py                   ← Comparison utility
├── README.md                                 ← This file
├── QUICK_START_GPT.md                        ← 5-minute guide
├── GPT_INFERENCE_GUIDE.md                    ← Full reference
├── COMPARISON_WITH_PREFERENCE_RL.md          ← Detailed comparison
└── inference_results/                        ← Output directory
    ├── inference_gpt_gpt-4o_20240806_143022.json
    ├── inference_gpt_gpt-4_20240806_154530.json
    └── ...

Related:
/gpfs/projects/p32143/RL_human_decision/baseline/preference_RL/
├── inference.py                              ← For comparison
├── inference_results/
└── ...
```

## Key Statistics

### Input Data
- **Total texts**: 642 (305 AI + 337 Human)
- **Source**: `/gpfs/projects/p32143/RL_human_decision/simulator/test_simulator.csv`
- **Text format**: Markdown with explanation instructions

### Expected Output
- **Format**: JSON with text_id, label, text, explanation
- **Explanations**: 2-6 bullet points each
- **Processing time**: 10-15 minutes
- **Output size**: ~50-100 MB JSON

## Next Steps

### 1. Run GPT Inference
```bash
python gpt_inference.py
```

### 2. Compare with Baseline (optional)
```bash
python compare_explanations.py
```

### 3. Integrate Results
- Use explanations in your analysis pipeline
- Compare with human preferences
- Evaluate quality metrics

### 4. Iterate
- Test different models
- Adjust temperature/max_tokens
- Fine-tune prompts if needed

## Documentation

| Doc | Purpose | Read Time |
|-----|---------|-----------|
| `QUICK_START_GPT.md` | Get started fast | 5 min |
| `GPT_INFERENCE_GUIDE.md` | Complete reference | 15 min |
| `COMPARISON_WITH_PREFERENCE_RL.md` | Understand differences | 10 min |
| `README.md` | This file | 5 min |

## FAQ

**Q: Which model should I use?**
A: Start with `gpt-4o` (good quality/cost balance). Upgrade to `gpt-4` if you need best quality.

**Q: Can I run this without an API key?**
A: No, GPT inference requires OpenAI API access. Use preference_RL/inference.py for local inference.

**Q: How much will this cost?**
A: Approximately $0.30-$6 depending on model. Monitor at https://platform.openai.com/account/usage/overview

**Q: Can I modify the prompt?**
A: Yes, edit `POLICY_INSTRUCTION` in the script. Keep the format similar to preference_RL for fair comparison.

**Q: How do I compare quality?**
A: Use `compare_explanations.py` or manually review sample outputs.

## Support

- **OpenAI API**: https://platform.openai.com/docs/api-reference
- **Pricing**: https://openai.com/pricing
- **Python SDK**: https://github.com/openai/openai-python
- **Usage**: https://platform.openai.com/account/usage/overview

## Version

- Created: 2026-08-06
- Status: Production Ready ✅
- Python: 3.8+
- Dependencies: openai, pandas

---

**Ready to start?** See `QUICK_START_GPT.md` or run:

```bash
export OPENAI_API_KEY='sk-...'
python gpt_inference.py
```
