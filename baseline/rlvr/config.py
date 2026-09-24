"""
Configuration for verifiable-reward RL training on Qwen3.6-27B.

This approach trains the model to generate both explanations AND correct judgments,
with reward directly based on whether the judgment matches ground truth.
"""

from pathlib import Path

# ========================================================================
# Paths
# ========================================================================
PROJECT_ROOT = Path("/gpfs/projects/p32143/RL_human_decision")
DB_PATH = PROJECT_ROOT / "statistic" / "text_2026-07-27.db"
TEXT_DATA_ROOT = PROJECT_ROOT / "webscrape" / "social"

OUTPUT_DIR = Path("/gpfs/projects/p32143/RL_human_decision/baseline/rlvr/checkpoints")
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR_WEIGHTS = OUTPUT_DIR / "cache"

# Dataset cache
DATASET_CACHE_DIR = OUTPUT_DIR / "dataset_cache"

# ========================================================================
# Model Configuration
# ========================================================================
POLICY_MODEL_NAME = "Qwen/Qwen3.6-27B"
POLICY_CACHE_DIR = "/projects/p32143/cache/huggingface/qwen36_27b"

# ========================================================================
# LoRA Configuration
# ========================================================================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ========================================================================
# GRPO Training Configuration
# ========================================================================
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
NUM_GENERATIONS = 8  # Group size for GRPO
MAX_COMPLETION_LENGTH = 512  # Max tokens for explanation + judgment
MAX_PROMPT_LENGTH = 5096
BETA = 0.0  # No KL penalty
TEMPERATURE = 1.0
BF16 = True
SAVE_STEPS = 50
LOGGING_STEPS = 5

# ========================================================================
# Reward Configuration
# ========================================================================
REWARD_CORRECT = 1.0  # Reward for correct judgment
REWARD_INCORRECT = -3.0  # Penalty for incorrect judgment

# ========================================================================
# Dataset Configuration
# ========================================================================
TRAIN_TEST_SPLIT = 0.85  # 85% train, 15% test
RANDOM_SEED = 42
MAX_TEXTS = None  # Set to None for all, or int for debugging (e.g., 1000)

# ========================================================================
# Processing Configuration
# ========================================================================
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
