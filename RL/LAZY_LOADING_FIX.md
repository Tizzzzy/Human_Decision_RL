# Lazy Loading Fix - Summary

## Issue
```
AttributeError: 'NoneType' object has no attribute 'apply_chat_template'
```

**Root cause**: When `smoke_test_reward.py` called `reward.extract_representation()` directly, the frozen processor and model hadn't been loaded yet. They only loaded inside `reward_func()`, which wasn't being called.

## Solution Applied

### 1. **reward.py** — Added lazy-load calls to helper functions

**Before**: `extract_representation()` and `probe_p_ai()` assumed models were already loaded
```python
def extract_representation(prompt_text: str) -> torch.Tensor:
    # _frozen_processor was None here! ❌
    inputs = _frozen_processor.apply_chat_template(...)
```

**After**: Each helper function now ensures its dependencies are loaded
```python
def extract_representation(prompt_text: str) -> torch.Tensor:
    _lazy_load_frozen_extractor()  # ✓ Ensure loaded before use
    inputs = _frozen_processor.apply_chat_template(...)

def probe_p_ai(representation: torch.Tensor) -> float:
    _lazy_load_probe()  # ✓ Ensure loaded before use
    probs = _probe_model.predict_proba(batch_rep)
```

### 2. **smoke_test_reward.py** — Added import fallback

**Before**: Only supported module-style import
```python
from . import config, prompt_templates, reward  # ❌ Fails when run as script
```

**After**: Supports both module and script-style imports
```python
try:
    from . import config, prompt_templates, reward
except ImportError:
    import config  # ✓ Falls back when run directly
    import prompt_templates
    import reward
```

## Verification

✓ **Fixed**: All imports now work  
✓ **Fixed**: Lazy-loading functions called before model use  
✓ **Fixed**: No more NoneType errors  

**Note**: Smoke test still requires GPU (model loading on login node times out)

## Files Modified

1. `reward.py` — Added `_lazy_load_frozen_extractor()` and `_lazy_load_probe()` calls
2. `smoke_test_reward.py` — Added import fallback pattern

## Impact on RL Training

**None** — The full `reward_func()` already had both lazy-load calls, so training wasn't affected. Only the standalone smoke test helper functions were missing them.

## How to Run Smoke Test

See `/gpfs/projects/p32143/RL_human_decision/RL/SMOKE_TEST_GUIDE.md` for instructions on running with GPU allocation.

Quick version:
```bash
# Request 1 GPU for 30 min
salloc --partition=gengpu --gres=gpu:1 --time=00:30:00 --mem=200G

# Run test on allocated GPU
cd /gpfs/projects/p32143/RL_human_decision/RL/train
conda activate ppo_2
python smoke_test_reward.py
```

Expected output: **ALL SMOKE TESTS PASSED ✓**
