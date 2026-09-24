# Update Summary: Integration of New 5-Model Ensemble Simulator

**Date**: 2026-09-07  
**Change**: Updated RL training code to use the new `trained_ensemble.pth` (5-model DeepEnsemble) instead of the old single `probe_weights.pth`

## Files Modified

### 1. `config.py`
- Updated `PROBE_WEIGHTS_PATH` from `probe_training/probe_weights.pth` to `simulator/trained_ensemble.pth`
- All other hyperparameters remain unchanged

### 2. `probe_model.py` 
- Added `DeepEnsemble` class alongside existing `ShallowMLP`
- `DeepEnsemble` wraps 5 ShallowMLP models (input_dim=5120, hidden_dim=512)
- Includes `save()` and `load()` methods to persist/restore ensemble checkpoints
- `predict_proba()` returns ensemble-averaged probabilities across all 5 models

### 3. `reward.py`
- Updated module docstring to reference the ensemble probe
- Modified `_lazy_load_probe()` to instantiate `DeepEnsemble(num_models=5)` and call `.load()` instead of manually loading a single state_dict
- Updated `probe_p_ai()` to work with ensemble's `predict_proba()` method:
  - Takes single representation [5120] as input
  - Wraps in batch dimension [1, 5120]
  - Calls `ensemble.predict_proba()` which returns [1, 1] numpy array
  - Extracts scalar probability value
- Added numpy import
- Ensemble fully frozen (all parameters) during inference

### 4. `smoke_test_reward.py`
- Updated Test 2 description to reference "Probe Ensemble" instead of "Probe"
- Added type checking to verify output is a float

### 5. `IMPLEMENTATION_SUMMARY.md`
- Updated section on Task 2 to note "5-model DeepEnsemble" in the reward section
- Updated paths table to point to `trained_ensemble.pth`

## Architecture Change

**Before**: 
- Reward: Frozen Qwen3.6-27B + single ShallowMLP(5120, 256, hidden_dim)
- P(AI): Single model output

**After**:
- Reward: Frozen Qwen3.6-27B + DeepEnsemble(5 x ShallowMLP(5120, 512))
- P(AI): Average of 5 model outputs (more robust, calibrated)

## Compatibility

✓ All module imports verified in `ppo_2` conda environment  
✓ Reward function interface unchanged (still accepts prompts, completions, **kwargs)  
✓ Backward compatible with existing GRPOTrainer integration  
✓ No changes needed to `train_grpo.py` or `dataset.py`

## Testing

Run smoke test to verify ensemble loads and generates rewards:

```bash
cd /gpfs/projects/p32143/RL_human_decision/RL/train
source /home/kjj8053/anaconda3/etc/profile.d/conda.sh && conda activate ppo_2
python smoke_test_reward.py
```

Expected output: All 5 tests pass, including:
- Test 2: Ensemble P(AI) outputs in [0, 1]
- Test 3: Reward formula works correctly
- Test 5: Policy model loads and generates (if attempted)

## Next Steps

1. Run `smoke_test_reward.py` to verify the ensemble probe integration works
2. Run `sbatch run_train_grpo.sbatch` to start GRPO training with the ensemble

No changes needed to the GRPO training script or undecided pool builder.
