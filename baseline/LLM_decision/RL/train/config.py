"""
Centralized configuration for the RL training pipeline.
All paths, model names, and hyperparameters in one place.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path("/gpfs/projects/p32143/RL_human_decision")
TRAIN_POOL_PATH = PROJECT_ROOT / "/baseline/LLM_decision/RL/get_undecide_text/train_undecided.jsonl"
TEST_POOL_PATH = PROJECT_ROOT / "/baseline/LLM_decision/RL/get_undecide_text/test_undecided.jsonl"
PROBE_WEIGHTS_PATH = PROJECT_ROOT / "/baseline/LLM_decision/probe_training/ensemble_models.pth"
OUTPUT_DIR = PROJECT_ROOT / "/baseline/LLM_decision/RL/train/checkpoints"
LOG_DIR = PROJECT_ROOT / "/baseline/LLM_decision/RL/train/logs"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Models
# ============================================================================
POLICY_MODEL_NAME = "Qwen/Qwen3.6-27B"
POLICY_CACHE_DIR = "/projects/p32143/cache/huggingface/qwen36_27b"

FROZEN_MODEL_NAME = "Qwen/Qwen3.6-27B"
FROZEN_CACHE_DIR = "/projects/p32143/cache/huggingface/qwen36_27b"

# ============================================================================
# LoRA Configuration
# ============================================================================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ============================================================================
# GRPO Training Configuration
# ============================================================================
# Generation params
NUM_GENERATIONS = 8  # group size for GRPO's relative-advantage baseline
MAX_COMPLETION_LENGTH = 512  # max tokens for explanation generation
MAX_PROMPT_LENGTH = 5096  # max tokens for policy prompt

# Optimization
LEARNING_RATE = 1e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
BF16 = True  # use bfloat16 for efficiency
NUM_TRAIN_EPOCHS = 1  # provisional; can adjust based on dataset size / throughput

# GRPO-specific
BETA = 0.0  # no KL penalty / no reference model (GRPO has built-in group-relative baseline)
TEMPERATURE = 1.0
SAVE_STEPS = 50
LOGGING_STEPS = 5

# ============================================================================
# Reward Configuration
# ============================================================================
REWARD_CLIP_FLOOR = -5.0  # floor clipping for rewards to bound outliers
# Formula: reward = max(1 - BCE(p_ai, y), REWARD_CLIP_FLOOR)
# where BCE = -(y*log(p) + (1-y)*log(1-p))
