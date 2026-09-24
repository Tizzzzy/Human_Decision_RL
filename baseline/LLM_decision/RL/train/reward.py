"""
Reward function module for GRPO training.

Loads a frozen copy of Qwen3.6-27B (via AutoModelForMultimodalLM, matching extract_representation.py)
and the frozen 5-model DeepEnsemble probe, then exposes a reward_func(prompts, completions, **kwargs)
callable compatible with trl.GRPOTrainer.

The reward function:
  1. Takes a generated explanation from the policy
  2. Builds an evaluation prompt (text + explanation) for the probe
  3. Extracts the last-token hidden representation via frozen Qwen3.6-27B
  4. Scores it with the frozen ensemble (5 ShallowMLP models) to get P(AI)
  5. Compares to ground truth and returns a reward (using BCE loss formula)
"""

import math
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# Handle relative imports when run from different contexts
try:
    from . import config, prompt_templates
    from .probe_model import ShallowMLP, DeepEnsemble
except ImportError:
    import config
    import prompt_templates
    from probe_model import ShallowMLP, DeepEnsemble


# ============================================================================
# Module-level singletons (lazy-loaded on first use)
# ============================================================================
_frozen_processor = None
_frozen_model = None
_probe_model = None


def _lazy_load_frozen_extractor():
    """
    Lazy-load the frozen Qwen3.6-27B model (via AutoModelForMultimodalLM, matching extract_representation.py).

    This is called once; subsequent calls reuse the cached model.
    Cached in module-level globals to avoid reloading the huge model on every reward_func call.
    """
    global _frozen_processor, _frozen_model

    if _frozen_model is None:
        print("[Reward] Loading frozen Qwen3.6-27B processor...")
        _frozen_processor = AutoProcessor.from_pretrained(config.FROZEN_MODEL_NAME)

        print("[Reward] Loading frozen Qwen3.6-27B model (AutoModelForMultimodalLM)...")
        _frozen_model = AutoModelForMultimodalLM.from_pretrained(
            config.FROZEN_MODEL_NAME,
            device_map="auto",
            cache_dir=config.FROZEN_CACHE_DIR,
            trust_remote_code=True,
        )
        _frozen_model.eval()

        # Freeze parameters
        for param in _frozen_model.parameters():
            param.requires_grad = False

        print("[Reward] Frozen Qwen3.6-27B model loaded and frozen")


def _lazy_load_probe():
    """
    Lazy-load the frozen probe (DeepEnsemble of 5 models).

    Loads from trained_ensemble.pth which contains 5 ShallowMLP model state_dicts.
    Each model: Linear(5120, 512) -> LayerNorm(512) -> GELU -> Dropout -> Linear(512, 1)
    """
    global _probe_model

    if _probe_model is None:
        print("[Reward] Loading frozen probe ensemble from", config.PROBE_WEIGHTS_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _probe_model = DeepEnsemble(num_models=5, input_dim=5120, device=device)
        _probe_model.load(str(config.PROBE_WEIGHTS_PATH))
        _probe_model.set_eval()

        # Freeze all parameters
        for model in _probe_model.models:
            for param in model.parameters():
                param.requires_grad = False

        print("[Reward] Frozen probe ensemble (5 models) loaded and frozen")


def extract_representation(prompt_text: str) -> torch.Tensor:
    """
    Extract the last-token, last-layer hidden representation from the frozen Qwen3.6-27B.

    Exactly mirrors the logic in simulator/preprocess_data/extract_representation.py:
      1. Build messages: [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
      2. apply_chat_template + tokenize
      3. Forward pass with output_hidden_states=True
      4. Extract hidden_states[-1][0, -1, :] (last layer, last token, 5120-dim)

    Args:
        prompt_text: The evaluation prompt string

    Returns:
        Tensor of shape [5120] (float32 on CPU)
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }
    ]

    inputs = _frozen_processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_frozen_model.device)

    with torch.no_grad():
        outputs = _frozen_model(**inputs, output_hidden_states=True)

    # Extract: [batch=0, token=-1 (last), all dims]
    last_token_final_layer = outputs.hidden_states[-1][0, -1, :]

    return last_token_final_layer.cpu().float()


def probe_p_ai(representation: torch.Tensor) -> float:
    """
    Get P(AI) from the frozen probe ensemble.

    Args:
        representation: Tensor of shape [5120]

    Returns:
        float in [0, 1]: P(AI) (ensemble-averaged probability)
    """
    # DeepEnsemble.predict_proba expects batch input, returns numpy array
    batch_rep = representation.unsqueeze(0)  # [1, 5120]
    probs = _probe_model.predict_proba(batch_rep)  # [1, 1] numpy array
    p_ai = float(probs[0, 0])

    return p_ai


def compute_reward(p_ai: float, y: float, eps: float = 1e-7) -> float:
    """
    Compute reward by comparing P(AI) to ground-truth label.

    Formula: reward = max(1 - BCE(p_ai, y), REWARD_CLIP_FLOOR)
      where BCE = -(y*log(p) + (1-y)*log(1-p))

    Intuition:
      - y=1 (AI text): reward is high when p_ai is high and confident (close to 1)
      - y=0 (Human text): reward is high when p_ai is low and confident (close to 0)
      - Confident-but-wrong predictions get large negative rewards (BCE -> infinity)
      - Floor clipping (-5.0) prevents outliers from dominating GRPO's relative-advantage baseline

    Args:
        p_ai: Predicted P(AI) from probe (in [0,1])
        y: Ground truth binary label (0 = Human, 1 = AI)
        eps: Small epsilon to avoid log(0)

    Returns:
        float: reward value, typically in range [-5, 1]
    """
    # Clip probabilities to avoid log(0)
    p = max(eps, min(p_ai, 1 - eps))

    # Binary cross-entropy
    bce = -(y * math.log(p) + (1 - y) * math.log(1 - p))

    # Reward: 1 - BCE, clipped at floor
    reward = max(1.0 - bce, config.REWARD_CLIP_FLOOR)

    return reward


def reward_func(
    prompts: List[str],
    completions: List[str],
    text_content: Optional[List[str]] = None,
    label_truth_numeric: Optional[List[int]] = None,
    text_id: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    GRPOTrainer-compatible reward function.

    Called during training: for each sample, the policy generates a completion (explanation),
    and this function scores how well that explanation helps the probe/reader correctly
    identify the AI vs. human nature of the text.

    Args:
        prompts: List of policy prompts (not used directly in this implementation;
                 included for API compatibility with GRPOTrainer)
        completions: List of generated explanations from the policy (raw text, not tokenized)
        text_content: List of original texts (required; passed from dataset)
        label_truth_numeric: List of ground-truth binary labels (required; 0=Human, 1=AI)
        text_id: List of text IDs (optional; for debugging)
        **kwargs: Any extra columns from dataset are passed here

    Returns:
        List of float rewards, one per sample, aligned with inputs
    """
    # Lazy-load frozen models on first call
    _lazy_load_frozen_extractor()
    _lazy_load_probe()

    if text_content is None or label_truth_numeric is None:
        raise ValueError("text_content and label_truth_numeric are required for reward_func")

    rewards = []

    for i in range(len(completions)):
        try:
            # Get components
            explanation = completions[i].strip()
            text = text_content[i]
            label = label_truth_numeric[i]

            # Build evaluation prompt (text + explanation)
            eval_prompt = prompt_templates.build_eval_prompt(text, explanation)

            # Extract representation from frozen model
            representation = extract_representation(eval_prompt)

            # Score with frozen probe
            p_ai = probe_p_ai(representation)

            # Compute reward
            reward = compute_reward(p_ai, float(label))

            rewards.append(reward)

        except Exception as e:
            print(f"[Reward] Error processing sample {i}: {e}")
            # On error, return a neutral reward
            rewards.append(0.0)

    return rewards
