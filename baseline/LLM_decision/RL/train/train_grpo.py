"""
Main GRPO RL training script.

Wires together:
  - Policy: Qwen3.6-27B with LoRA adapter (being trained)
  - Reward function: frozen Qwen3.6-27B + frozen probe
  - Dataset: undecided text pool from Task 1
  - Training loop: GRPOTrainer from trl

Usage:
  conda activate ppo_2
  python train_grpo.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

try:
    # When run as a module (python -m RL.train or similar)
    from . import config, dataset, reward
except ImportError:
    # When run directly as a script
    import config
    import dataset
    import reward


def load_policy_model_and_tokenizer():
    """
    Load the Qwen3.6-27B policy model and tokenizer.

    Note: Uses AutoModelForCausalLM (not AutoModelForMultimodalLM), which is what
    GRPOTrainer's internal .generate() and forward-pass machinery needs.
    """
    print("[Policy] Loading tokenizer from", config.POLICY_MODEL_NAME)
    policy_tokenizer = AutoTokenizer.from_pretrained(
        config.POLICY_MODEL_NAME,
        cache_dir=config.POLICY_CACHE_DIR,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    print("[Policy] Loading model from", config.POLICY_MODEL_NAME)
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.POLICY_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.POLICY_CACHE_DIR,
        trust_remote_code=True,
    )

    print(f"[Policy] Model loaded: {policy_model.num_parameters():,} parameters")
    return policy_model, policy_tokenizer


def build_lora_config() -> LoraConfig:
    """
    Build LoRA configuration for efficient fine-tuning.

    Targets the key attention components (q, k, v, o projections).
    """
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.LORA_TARGET_MODULES,
    )


def build_grpo_config() -> GRPOConfig:
    """
    Build GRPO training configuration.

    GRPO (Group Relative Policy Optimization) is a scalable on-policy RL algorithm
    that uses a group-relative baseline for advantage estimation (no critic model needed).
    """
    return GRPOConfig(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.01,
        bf16=config.BF16,
        gradient_checkpointing=True,
        logging_steps=config.LOGGING_STEPS,
        logging_dir=str(config.LOG_DIR),
        save_steps=config.SAVE_STEPS,
        save_total_limit=3,
        report_to="none",  # disable wandb/tensorboard for now
        # GRPO-specific
        num_generations=config.NUM_GENERATIONS,
        max_completion_length=config.MAX_COMPLETION_LENGTH,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        beta=config.BETA,  # no KL penalty
        temperature=config.TEMPERATURE,
    )


def main():
    print("\n" + "="*70)
    print("             GRPO RL TRAINING: Qwen3.6-27B Explanation Policy        ")
    print("="*70 + "\n")

    # ========================================================================
    # 1. Load policy model and tokenizer
    # ========================================================================
    print("[1/5] Loading policy model and tokenizer...")
    policy_model, policy_tokenizer = load_policy_model_and_tokenizer()

    # ========================================================================
    # 2. Load datasets
    # ========================================================================
    print("\n[2/5] Loading datasets...")
    train_dataset = dataset.get_train_dataset(policy_tokenizer)
    eval_dataset = dataset.get_eval_dataset(policy_tokenizer)

    # ========================================================================
    # 3. Setup LoRA
    # ========================================================================
    print("\n[3/5] Building LoRA config...")
    lora_config = build_lora_config()
    print(f"[LoRA] r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    print(f"[LoRA] target_modules={lora_config.target_modules}")

    # ========================================================================
    # 4. Setup GRPO trainer
    # ========================================================================
    print("\n[4/5] Building GRPO trainer...")
    grpo_config = build_grpo_config()
    print(f"[GRPO] num_generations={grpo_config.num_generations}")
    print(f"[GRPO] max_completion_length={grpo_config.max_completion_length}")
    print(f"[GRPO] learning_rate={grpo_config.learning_rate}")
    print(f"[GRPO] batch_size={grpo_config.per_device_train_batch_size}")
    print(f"[GRPO] gradient_accumulation_steps={grpo_config.gradient_accumulation_steps}")
    print(f"[GRPO] bf16={grpo_config.bf16}")
    print(f"[GRPO] beta (KL penalty)={grpo_config.beta}")

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[reward.reward_func],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=policy_tokenizer,
        peft_config=lora_config,
    )

    # ========================================================================
    # 5. Train
    # ========================================================================
    print("\n[5/5] Starting GRPO training...")
    print("="*70)
    trainer.train()

    # ========================================================================
    # Save final model
    # ========================================================================
    print("\n" + "="*70)
    print("Training complete! Saving final model...")
    final_model_dir = config.OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(final_model_dir))
    policy_tokenizer.save_pretrained(str(final_model_dir))
    print(f"Final model saved to {final_model_dir}")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
