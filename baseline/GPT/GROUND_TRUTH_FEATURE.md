# Ground Truth Feature in GPT Inference

## Overview

The GPT inference script has been enhanced to support **ground truth (label-aware) explanation generation**. The model can now generate explanations while knowing whether a post is AI or Human-written, but without revealing this information in the explanations themselves.

## What is Ground Truth?

**Ground Truth** = the actual label (AI or Human) of each post in the test dataset.

- **Without Ground Truth**: Model generates explanations for any post, not knowing if it's AI or Human
- **With Ground Truth**: Model knows the actual category but is instructed to generate markers that would help detect it, WITHOUT revealing the answer

## Use Cases

### 1. **More Targeted Explanations**
When the model knows the actual category, it can generate more specific and accurate linguistic markers for that category:

```
Without truth: "Look for patterns that might indicate AI or Human authorship"
With truth (AI): "Look for patterns that specifically indicate AI authorship"
With truth (Human): "Look for patterns that specifically indicate human authorship"
```

### 2. **Evaluation & Analysis**
- Compare explanations with vs without ground truth
- See if the model generates different insights when it "knows" the answer
- Evaluate explanation quality for each category separately

### 3. **Training Data Generation**
- Generate high-quality explanations for training a preference model
- The model can be more focused knowing what it's analyzing

## Configuration

### Enable/Disable Ground Truth

In `gpt_inference.py`, modify this variable:

```python
# Toggle between with/without ground truth
USE_GROUND_TRUTH = True  # Set to False to use original prompt without ground truth
```

**Default**: `True` (ground truth enabled)

## How It Works

### With Ground Truth Enabled

The prompt becomes:

```
Task: Analyze the provided Social Media post for linguistic markers of AI or human authorship.

IMPORTANT INFORMATION (DO NOT REVEAL IN YOUR EXPLANATION):
Ground truth: This post was written by a {label}.

Your task: Generate linguistic markers that would help someone detect that this post 
was written by a {label}, WITHOUT explicitly saying the answer.

[Rest of constraints...]
```

**Key Points**:
1. Model is told the actual category (AI or Human)
2. Model is explicitly instructed NOT to reveal this
3. Model generates markers specific to that category
4. Explanations are hidden from the human evaluator in your system
5. They help the model generate more targeted, accurate explanations

### Without Ground Truth

Uses the original prompt:

```
Task: Analyze the provided Social Media post for linguistic markers of 
AI or human authorship.

[Constraints...]
```

Model generates generic explanations without knowing the actual category.

## Output Differences

### Filename Indicates Mode

```
# With ground truth
inference_gpt_gpt-5.6-luna_with_truth_20240806_143022.json

# Without ground truth
inference_gpt_gpt-5.6-luna_without_truth_20240806_143022.json
```

### Console Output Shows Mode

```
======================================================================
GPT Inference: Generate explanations using gpt-5.6-luna
Mode: WITH ground truth
Note: Model knows ground truth but will NOT reveal it in explanations
======================================================================

[Inference] Generating explanations using gpt-5.6-luna (with ground truth)...
  [1/642] Generating explanation for reddit_2932 (Human)... done
  [2/642] Generating explanation for reddit_1341 (Human)... done
```

## Example Outputs

### Same Text, Different Modes

**Text**: "Just finished my morning coffee and headed to the gym. Feeling motivated!"

#### Without Ground Truth
```
- Look for casual conversational phrasing like "just" and "feeling"
- Look for personal routine descriptions with everyday activities
- Look for emoji use or exclamation marks indicating enthusiasm
```

#### With Ground Truth (Human)
```
- Look for informal personal narratives about daily routines and feelings
- Look for natural enthusiasm markers like exclamation points in casual contexts
- Look for specific time references to daily activities mixed with personal emotions
```

#### With Ground Truth (AI)
```
- Look for overly enthusiastic or formulaic motivational language patterns
- Look for generic action phrases like "heading to" combined with broad goals
- Look for enthusiasm that follows a predictable structure without personal details
```

## Comparison Workflow

### Generate Both Versions

```bash
cd /projects/p32143/RL_human_decision/baseline/GPT

# 1. Generate WITH ground truth
python gpt_inference.py
# Output: inference_gpt_gpt-5.6-luna_with_truth_*.json

# 2. Switch mode
# Edit gpt_inference.py: USE_GROUND_TRUTH = False

# 3. Generate WITHOUT ground truth
python gpt_inference.py
# Output: inference_gpt_gpt-5.6-luna_without_truth_*.json

# 4. Compare
python compare_explanations_with_truth.py
```

### Create Comparison Script

Create `compare_explanations_with_truth.py`:

```python
import json
import glob

def compare_truth_modes():
    """Compare explanations with and without ground truth."""
    
    # Load both versions
    with_truth_files = glob.glob("inference_results/inference_gpt_*_with_truth_*.json")
    without_truth_files = glob.glob("inference_results/inference_gpt_*_without_truth_*.json")
    
    if not with_truth_files or not without_truth_files:
        print("Need both versions to compare")
        return
    
    with_truth = json.load(open(with_truth_files[-1]))  # Latest file
    without_truth = json.load(open(without_truth_files[-1]))
    
    # Create mappings
    with_by_id = {r['text_id']: r for r in with_truth}
    without_by_id = {r['text_id']: r for r in without_truth}
    
    print("\n" + "="*70)
    print("Ground Truth Mode Comparison")
    print("="*70)
    
    # Sample comparisons
    sample_ids = list(with_by_id.keys())[:5]
    
    for text_id in sample_ids:
        with_r = with_by_id[text_id]
        without_r = without_by_id[text_id]
        
        print(f"\n{'='*70}")
        print(f"Text ID: {text_id} ({with_r['label']})")
        print(f"{'='*70}")
        
        print(f"\nWITH Ground Truth ({with_r['label']}):")
        print(with_r['explanation'][:300])
        
        print(f"\nWITHOUT Ground Truth:")
        print(without_r['explanation'][:300])
        
        print(f"\nDifferences:")
        print(f"  With truth length: {len(with_r['explanation'])}")
        print(f"  Without truth length: {len(without_r['explanation'])}")

if __name__ == "__main__":
    compare_truth_modes()
```

## Performance Impact

### Speed
**Minimal** - Ground truth doesn't affect speed (still ~1 sec/text)

### Quality
**Likely Improved** - Model can generate more targeted explanations

### Cost
**Same** - Token counts similar for both modes

## Design Decisions

### Why Not Reveal Ground Truth in Output?

The ground truth is hidden in your system (not shown to human evaluators) because:

1. **Fair Evaluation**: Humans evaluate based on explanation quality alone, not label bias
2. **Research**: You can study if informed explanations are better/different
3. **Integration**: The explanation system doesn't reveal the answer, maintains the mystery

### Why This Matters

- Model can generate more **focused** markers
- Explanations are more **category-specific**
- Better for training downstream models
- More **honest** - model knows but focuses on markers, not the answer

## Advanced: Custom Ground Truth Prompts

To customize the ground truth prompt, edit `POLICY_INSTRUCTION_WITH_TRUTH`:

```python
POLICY_INSTRUCTION_WITH_TRUTH = """Task: Analyze the provided Social Media post...

IMPORTANT INFORMATION (DO NOT REVEAL):
Ground truth: This post was written by a {label}.

YOUR CUSTOM INSTRUCTION HERE:
Generate explanations that...

[Rest of constraints]"""
```

## Troubleshooting

### All Explanations Look the Same

**Issue**: Ground truth not being used
**Solution**: Verify `USE_GROUND_TRUTH = True` in script and labels are being loaded

### Model Reveals the Answer

**Issue**: Model saying "This is AI because..." or "This is Human because..."
**Solution**: The prompt tells it not to. If happening, try:
- Lower temperature (more consistent)
- Strengthen the "DO NOT REVEAL" instruction
- Try different model

### Unexpected Label Format

**Issue**: Labels aren't "AI" or "Human"
**Solution**: Verify test_simulator.csv has correct labels:
```bash
cut -d',' -f6 test_simulator.csv | sort | uniq -c
```

## Comparison with preference_RL

| Aspect | GPT with Truth | preference_RL | GPT without Truth |
|--------|---|---|---|
| **Ground Truth** | ✅ Knows label | Trained on labels | ❌ Doesn't know |
| **Output Quality** | Specific | General | General |
| **Prompt** | Informed | Learned | Generic |
| **Use Case** | Analysis | Production | Baseline |

## Summary

**Ground Truth Feature Benefits**:
- ✅ More targeted explanations
- ✅ Category-specific markers
- ✅ Better for analysis and evaluation
- ✅ No performance cost
- ✅ Model doesn't reveal the answer

**Usage**:
```bash
# Default (with ground truth)
python gpt_inference.py

# Without ground truth
# Edit: USE_GROUND_TRUTH = False
# python gpt_inference.py
```

**Output Files**:
- `inference_gpt_*_with_truth_*.json` - Generated with ground truth
- `inference_gpt_*_without_truth_*.json` - Generated without ground truth

---

See `README.md` for general usage and `GPT_INFERENCE_GUIDE.md` for detailed reference.
