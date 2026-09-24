# Quick Start: GPT Inference

## TL;DR

```bash
# 1. Set OpenAI API key
export OPENAI_API_KEY='sk-...'

# 2. Run inference
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python gpt_inference.py

# 3. Check results
ls -lh inference_results/
cat inference_results/inference_gpt_gpt-4o_*.json | head -50
```

## Prerequisites (5 minutes)

### 1. Install OpenAI Package
```bash
pip install openai
```

### 2. Get OpenAI API Key
- Go to: https://platform.openai.com/account/api-keys
- Create new secret key
- Copy it somewhere safe

### 3. Set Environment Variable
```bash
# Add to ~/.bashrc or run in terminal
export OPENAI_API_KEY='sk-your-key-here'

# Verify it's set
echo $OPENAI_API_KEY
```

## Running (10-15 minutes for 642 texts)

```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python gpt_inference.py
```

### Expected Output
```
======================================================================
GPT Inference: Generate explanations using gpt-4o
======================================================================

[Setup] Initializing OpenAI client...
[Data] Loading test CSV...
[Data] Loaded 642 test samples
[Data] Reading text files...
[Data] Successfully loaded 642 texts

[Inference] Generating explanations using gpt-4o...
  [1/642] Generating explanation for reddit_2932... done
  [2/642] Generating explanation for reddit_1341... done
  ...
  [642/642] Generating explanation for reddit_XXX... done

[Results] Building results JSON...
[Results] Saved 642 results to /projects/p32143/RL_human_decision/baseline/GPT/inference_results/inference_gpt_gpt-4o_20240806_143022.json

======================================================================
Summary
======================================================================
Total texts processed: 642
Model used: gpt-4o
Temperature: 1.0
Max tokens: 200
Results file: /projects/p32143/RL_human_decision/baseline/GPT/inference_results/inference_gpt_gpt-4o_20240806_143022.json

Label distribution:
  AI: 305
  Human: 337
```

## Customization

### Use Different Model
Edit `gpt_inference.py`:
```python
GPT_MODEL = "gpt-4"  # Options: gpt-4, gpt-4-turbo-preview, gpt-4o, gpt-3.5-turbo
```

### Adjust Parameters
```python
TEMPERATURE = 0.7  # Lower = more consistent, Higher = more creative
MAX_TOKENS = 256   # Max length of explanation
RATE_LIMIT_DELAY = 1.0  # Seconds between requests (increase if rate limited)
```

## Output

### File Location
```
/projects/p32143/RL_human_decision/baseline/GPT/inference_results/
└── inference_gpt_gpt-4o_20240806_143022.json
```

### Format
```json
[
  {
    "text_id": "reddit_2932",
    "label": "Human",
    "text": "The actual post text...",
    "text_path": "SocialMedia_Reddit/reddit_2932.txt",
    "explanation": "- Look for conversational phrasing.\n- Look for personal anecdotes like 'I remember when'."
  },
  ...
]
```

## Cost Check

| Model | Estimated Cost |
|-------|----------------|
| gpt-3.5-turbo | ~$0.30 |
| gpt-4o | ~$1.20 |
| gpt-4 | ~$6.00 |

## Comparison with Preference_RL

| Aspect | GPT | Preference_RL |
|--------|-----|---------------|
| **Cost** | $0.30-6 | Free |
| **Speed** | ~1 sec/text | ~0.5 sec/text |
| **Quality** | Different style | Trained on preferences |
| **Access** | Internet required | Local (GPU) |

See `COMPARISON_WITH_PREFERENCE_RL.md` for detailed comparison.

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-...'
python gpt_inference.py
```

### "Invalid API key"
- Check OpenAI dashboard
- Verify key format: `sk-...`
- No extra spaces or quotes

### "Rate limit exceeded"
- Increase `RATE_LIMIT_DELAY` to 1.0 or 2.0
- Spread requests over time
- Try `gpt-3.5-turbo` (higher limit)

### "Model not available"
- Verify model name: `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`
- Check API key has access
- See OpenAI dashboard for available models

## Next Steps

### Option 1: Compare with Preference_RL
```bash
# Generate preference_RL explanations
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL
python inference.py

# Compare outputs
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python compare_explanations.py
```

### Option 2: Evaluate Quality
- Sample 10-20 explanations from each
- Compare with human judgment
- Rate clarity, relevance, usefulness

### Option 3: Use in Pipeline
- Save results JSON
- Load in analysis script
- Compare model predictions
- Measure performance

## File Locations

```
/gpfs/projects/p32143/RL_human_decision/
├── baseline/
│   ├── GPT/
│   │   ├── gpt_inference.py              ← Run this
│   │   ├── compare_explanations.py       ← Compare with preference_RL
│   │   ├── GPT_INFERENCE_GUIDE.md        ← Full guide
│   │   ├── COMPARISON_WITH_PREFERENCE_RL.md
│   │   ├── QUICK_START_GPT.md            ← This file
│   │   └── inference_results/            ← Outputs here
│   │
│   └── preference_RL/
│       ├── inference.py                  ← For comparison
│       └── inference_results/
│
├── simulator/
│   └── test_simulator.csv                ← Input data
│
└── text_data/social/
    ├── SocialMedia_Reddit/
    └── SocialMedia_rewrite/
```

## FAQ

**Q: Why is it slow?**
A: API latency + rate limiting. Try `gpt-3.5-turbo` for faster inference.

**Q: Can I run multiple models at once?**
A: Not recommended (rate limits). Run sequentially with delays between.

**Q: How do I stop it?**
A: Press `Ctrl+C`. You'll be charged only for completed requests.

**Q: Can I resume if interrupted?**
A: Not with current script. It processes all texts in order.

**Q: Which model is best?**
A: Start with `gpt-4o` (good quality/cost balance). Try `gpt-4` for best quality, `gpt-3.5-turbo` for cheapest.

**Q: How do I compare with my own model?**
A: Load both JSON files, compare explanations for same text_id. See `compare_explanations.py`.

## Support

- **API Issues**: https://help.openai.com
- **Pricing**: https://openai.com/pricing
- **Python SDK**: https://github.com/openai/openai-python
- **Full Guide**: `GPT_INFERENCE_GUIDE.md`

---

**Ready? Run:** `python gpt_inference.py`

💡 **Tip**: Monitor your API usage: https://platform.openai.com/account/usage/overview
