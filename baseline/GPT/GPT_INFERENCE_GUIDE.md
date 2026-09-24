# GPT Inference Script Guide

## Overview

This script generates explanations for social media texts using OpenAI's GPT models (GPT-4, GPT-4o, GPT-3.5-turbo).

The output format is identical to the preference_RL inference script for easy comparison between models.

## Prerequisites

### 1. Install OpenAI Package
```bash
pip install openai
```

### 2. Set OpenAI API Key
```bash
# Option 1: Export as environment variable
export OPENAI_API_KEY='sk-...'

# Option 2: Create .env file
echo 'OPENAI_API_KEY=sk-...' > .env

# Option 3: Set in script (not recommended for security)
# Modify the API_KEY line in gpt_inference.py
```

### 3. Verify Test Data
```bash
# Check test CSV exists
ls -lh /gpfs/projects/p32143/RL_human_decision/simulator/test_simulator.csv

# Check social media text files exist
ls /gpfs/projects/p32143/RL_human_decision/text_data/social/SocialMedia_Reddit/*.txt | wc -l
ls /gpfs/projects/p32143/RL_human_decision/text_data/social/SocialMedia_rewrite/*.txt | wc -l
```

## Configuration

Edit these variables in `gpt_inference.py`:

| Parameter | Default | Options | Purpose |
|-----------|---------|---------|---------|
| `GPT_MODEL` | `gpt-4o` | `gpt-4`, `gpt-4-turbo-preview`, `gpt-4o`, `gpt-3.5-turbo` | Model to use |
| `TEMPERATURE` | `1.0` | 0.0-2.0 | Randomness (0=deterministic, 2=very random) |
| `MAX_TOKENS` | `200` | 50-500 | Max length of explanation |
| `TOP_P` | `1.0` | 0.0-1.0 | Nucleus sampling (higher = more diverse) |
| `RATE_LIMIT_DELAY` | `0.5` | 0.1-5.0 | Seconds between API calls |

## Running the Script

### Basic Usage
```bash
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT

# Run with default settings (GPT-4o)
python gpt_inference.py
```

### With Custom Model
```bash
# Modify GPT_MODEL in script, then run
python gpt_inference.py
```

### Expected Output
```
======================================================================
GPT Inference: Generate explanations using gpt-4o
======================================================================

[Setup] Initializing OpenAI client...
[Data] Loading test CSV from /gpfs/projects/p32143/RL_human_decision/simulator/test_simulator.csv...
[Data] Loaded 642 test samples
[Data] Reading text files...
[Data] Successfully loaded 642 texts

[Inference] Generating explanations using gpt-4o...
  [1/642] Generating explanation for reddit_2932... done
  [2/642] Generating explanation for reddit_1341... done
  ...

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

## Output Format

### JSON Structure
```json
[
  {
    "text_id": "reddit_2932",
    "label": "Human",
    "text": "The actual social media post text...",
    "text_path": "SocialMedia_Reddit/reddit_2932.txt",
    "explanation": "- Look for conversational phrasing.\n- Look for personal anecdotes like 'I remember'..."
  },
  ...
]
```

### Output File Location
Results are saved to:
```
/projects/p32143/RL_human_decision/baseline/GPT/inference_results/
├── inference_gpt_gpt-4o_20240806_143022.json
├── inference_gpt_gpt-3.5-turbo_20240806_154530.json
└── ...
```

## Explanation Prompt

The script uses this exact prompt for all models:

```
Task: Analyze the provided Social Media post for linguistic markers of AI or human authorship.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list.
2. Do NOT provide an introduction, a final verdict, or a summary.
3. Do NOT group the bullet points into sub-headings (keep it a flat list).
4. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category or type of phrasing]."
    - "Look for [linguistic category or type of phrasing] like '[short example]'."
5. When providing examples, you may list multiple words separated by commas (e.g., like "word 1", "word 2").
6. Do NOT use bolding, italics, or any special Markdown formatting.
7. Limit the output to exactly 2 to 6 bullet points.

Post:
```{text}```

Explanation:
```

This prompt is identical to the one used in `preference_RL/inference.py` for consistency.

## Cost Estimation

### API Costs
Approximate costs for processing 642 texts (based on OpenAI pricing):

| Model | Input Cost | Output Cost | Total Cost |
|-------|-----------|------------|-----------|
| `gpt-3.5-turbo` | ~$0.10 | ~$0.20 | ~$0.30 |
| `gpt-4` | ~$2.00 | ~$4.00 | ~$6.00 |
| `gpt-4o` | ~$0.40 | ~$0.80 | ~$1.20 |
| `gpt-4-turbo` | ~$1.50 | ~$3.00 | ~$4.50 |

**Note**: Actual costs depend on:
- Text lengths (varies in test data)
- Explanation lengths (varies by model)
- API price changes
- Usage of vision/other features

## Troubleshooting

### "OPENAI_API_KEY not set"
**Solution**:
```bash
export OPENAI_API_KEY='sk-your-key-here'
python gpt_inference.py
```

### "Invalid API key"
**Solution**:
1. Verify key is correct in OpenAI dashboard
2. Check key hasn't expired
3. Ensure no extra spaces or quotes: `sk-xxxx`

### "Rate limit exceeded"
**Solution**:
1. Increase `RATE_LIMIT_DELAY` (e.g., 1.0 or 2.0 seconds)
2. Run during off-peak hours
3. Contact OpenAI support for rate limit increase

### "Model not found"
**Solution**:
1. Check model name is correct: `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`
2. Verify API key has access to that model
3. Check OpenAI dashboard for available models

### "Timeout error"
**Solution**:
1. Check internet connection
2. Try again later (OpenAI infrastructure issues)
3. Increase timeout in OpenAI client configuration

## Comparing with Preference_RL Inference

### Similarities
- ✅ Same input: `test_simulator.csv`
- ✅ Same prompt format
- ✅ Same output JSON structure
- ✅ Same label distribution (AI/Human)
- ✅ Same text content analysis

### Differences
| Aspect | GPT | Preference_RL |
|--------|-----|---------------|
| **Model** | OpenAI's GPT | Fine-tuned Qwen |
| **Speed** | ~1 sec/text | ~0.5 sec/text |
| **Cost** | ~$1-6 | Free (local) |
| **Quality** | Different reasoning | Trained on preferences |
| **Customization** | Prompt only | Full fine-tuning |

## Running Comparisons

### Generate with Both Models
```bash
# 1. Run GPT inference
cd /gpfs/projects/p32143/RL_human_decision/baseline/GPT
python gpt_inference.py

# 2. Run preference_RL inference
cd /gpfs/projects/p32143/RL_human_decision/baseline/preference_RL
python inference.py

# 3. Compare outputs (see next section)
```

### Comparison Script
```python
import json

# Load both outputs
with open('/projects/p32143/RL_human_decision/baseline/GPT/inference_results/inference_gpt_gpt-4o_*.json') as f:
    gpt_results = json.load(f)

with open('/projects/p32143/RL_human_decision/baseline/preference_RL/inference_results/inference_*.json') as f:
    pref_results = json.load(f)

# Compare explanations for same text_id
for gpt_res in gpt_results[:5]:
    text_id = gpt_res['text_id']
    pref_res = next((r for r in pref_results if r['text_id'] == text_id), None)
    
    print(f"\n=== {text_id} ({gpt_res['label']}) ===")
    print(f"GPT-4o:\n{gpt_res['explanation']}")
    print(f"\nPreference_RL:\n{pref_res['explanation']}")
```

## Advanced: Custom Model Comparison

### Run Multiple Models in Sequence
```bash
# Create a loop to run multiple models
for model in "gpt-4o" "gpt-4" "gpt-3.5-turbo"; do
    # Modify GPT_MODEL in script
    sed -i "s/GPT_MODEL = .*/GPT_MODEL = \"$model\"/" gpt_inference.py
    
    # Run inference
    python gpt_inference.py
    
    # Wait between runs to avoid rate limiting
    sleep 60
done
```

## Next Steps

### 1. Compare Outputs
```bash
# Compare GPT explanations with preference_RL explanations
python /gpfs/projects/p32143/RL_human_decision/baseline/GPT/compare_explanations.py
```

### 2. Evaluate Quality
- Human evaluation: Which explanations are better?
- Automatic metrics: Similarity, length, coverage
- Classification impact: Do explanations help detect AI vs Human?

### 3. Integrate with Training
- Use GPT explanations for new training data
- Compare with preference_RL explanations in DPO training
- Evaluate fine-tuned model on GPT-generated explanations

## Cost-Benefit Analysis

### Use GPT Inference When:
- ✅ You need quick baseline explanations
- ✅ You want to compare with proprietary models
- ✅ You have budget for API calls
- ✅ You need the latest model capabilities
- ✅ You want to test different models quickly

### Use Preference_RL Inference When:
- ✅ You want free (local) inference
- ✅ You need fast generation (no API latency)
- ✅ You've trained a custom model
- ✅ You want consistent, reproducible explanations
- ✅ You have limited API budgets

## Support

### Common Questions

**Q: Which model should I use?**
A: Start with `gpt-4o` (good balance of quality and cost), then compare with `gpt-4` for highest quality or `gpt-3.5-turbo` for lowest cost.

**Q: Can I run this without an API key?**
A: No, GPT inference requires OpenAI API access. Use preference_RL/inference.py for local inference.

**Q: How long does it take?**
A: ~10-15 minutes for 642 texts (at ~1 sec/text + API overhead), depending on model and latency.

**Q: Can I modify the prompt?**
A: Yes, modify `POLICY_INSTRUCTION` in the script. Keep the format consistent with preference_RL for comparison fairness.

**Q: How do I save costs?**
A: Use `gpt-3.5-turbo` instead of `gpt-4o`, or batch requests if OpenAI supports it.

## References

- OpenAI API Docs: https://platform.openai.com/docs/api-reference
- Model Pricing: https://openai.com/pricing
- Python SDK: https://github.com/openai/openai-python
