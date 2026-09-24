"""
Main GRPO RL training script for verifiable-reward approach.

Trains Qwen3.6-27B to generate explanations AND correct AI/Human judgments.
Reward is based on whether the judgment matches ground truth labels.

Wires together:
  - Policy: Qwen3.6-27B with LoRA adapter (being trained)
  - Reward function: Direct correctness check against ground truth
  - Dataset: (text_id, label_truth) pairs from text_guess table
  - Training loop: GRPOTrainer from trl

Usage:
  conda activate ppo_2
  python train_grpo_verifiable.py
"""

import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

try:
    # When run as a module
    from . import config, dataset, reward, plot_utils
except ImportError:
    # When run directly as a script
    import config
    import dataset
    import reward
    try:
        import plot_utils
    except ImportError:
        plot_utils = None


class LossTrackingCallback(TrainerCallback):
    """Custom callback to track loss and metrics during training."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_file = self.output_dir / "training_metrics.json"
        self.metrics = {
            "step": [],
            "training_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "reward": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the model logs metrics."""
        if logs is None:
            return

        # Extract relevant metrics
        if "loss" in logs:
            self.metrics["step"].append(state.global_step)
            self.metrics["training_loss"].append(logs["loss"])
            self.metrics["learning_rate"].append(logs.get("learning_rate", 0))

            # Try to extract reward-related metrics
            if "reward" in logs:
                self.metrics["reward"].append(logs["reward"])

            # Try to extract eval loss if available
            if "eval_loss" in logs:
                self.metrics["eval_loss"].append(logs["eval_loss"])

            # Save metrics periodically
            if state.global_step % 10 == 0:
                self._save_metrics()

    def _save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.loss_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self._save_metrics()


def compute_validation_accuracy(model, tokenizer, eval_dataset, num_samples=None):
    """
    Compute judgment accuracy on the validation set.

    Args:
        model: Trained policy model with LoRA adapter
        tokenizer: HF tokenizer
        eval_dataset: HF Dataset with validation samples
        num_samples (int): Limit evaluation to N samples (None = all)

    Returns:
        dict: Metrics with keys: accuracy, num_correct, num_total
    """
    from tqdm import tqdm

    # Limit dataset if requested
    if num_samples is not None:
        eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))

    print(f"\n[Computing validation accuracy on {len(eval_dataset)} samples...]")

    model.eval()
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(eval_dataset)):
            prompt = sample["prompt"]
            true_label = sample["label_truth"]

            try:
                # Generate completion
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5096).to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False,  # Use greedy decoding for consistent evaluation
                    repetition_penalty=1.1,
                )

                # Decode and parse judgment
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                judgment = prompt_templates.parse_generation(generated_text)[1]

                if judgment is None:
                    continue

                # Check if correct
                if judgment.upper() == true_label.upper():
                    num_correct += 1
                num_total += 1

            except Exception as e:
                print(f"Error at sample {idx}: {e}")
                continue

    accuracy = num_correct / max(num_total, 1)

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
    }


def load_policy_model_and_tokenizer():
    """
    Load the Qwen3.6-27B policy model and tokenizer.

    Uses AutoModelForCausalLM for compatibility with GRPOTrainer's
    generate() and forward-pass machinery.
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
        report_to="none",  # disable wandb/tensorboard
        # GRPO-specific
        num_generations=config.NUM_GENERATIONS,
        max_completion_length=config.MAX_COMPLETION_LENGTH,
        beta=config.BETA,  # no KL penalty
        temperature=config.TEMPERATURE,
    )


def main():
    print("\n" + "="*70)
    print("    GRPO RL TRAINING: Qwen3.6-27B with Verifiable Reward Signal")
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
    print(f"\n[Reward] Correct: +{config.REWARD_CORRECT}, Incorrect: {config.REWARD_INCORRECT}")

    # Create callback for loss tracking
    loss_callback = LossTrackingCallback(config.LOG_DIR)

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[reward.reward_func],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=policy_tokenizer,
        peft_config=lora_config,
        callbacks=[loss_callback],
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

    # ========================================================================
    # Compute validation accuracy
    # ========================================================================
    print("\n[5.5/6] Computing validation set accuracy...")
    try:
        # Reload model with adapter for evaluation
        from peft import PeftModel
        eval_model = AutoModelForCausalLM.from_pretrained(
            config.POLICY_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=config.POLICY_CACHE_DIR,
            trust_remote_code=True,
        )
        eval_model = PeftModel.from_pretrained(eval_model, str(final_model_dir))

        val_metrics = compute_validation_accuracy(eval_model, policy_tokenizer, eval_dataset)

        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['num_correct']}/{val_metrics['num_total']})")

        # Save validation metrics
        val_metrics_file = config.LOG_DIR / "validation_metrics.json"
        with open(val_metrics_file, "w") as f:
            json.dump(val_metrics, f, indent=2)
        print(f"  Saved to {val_metrics_file}")

        del eval_model

    except Exception as e:
        print(f"Warning: Could not compute validation accuracy: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Plot loss curves
    # ========================================================================
    print("\n[6/6] Plotting training loss...")
    metrics_file = config.LOG_DIR / "training_metrics.json"
    if metrics_file.exists() and plot_utils is not None:
        try:
            plot_utils.plot_training_loss(str(metrics_file), output_dir=str(config.OUTPUT_DIR))
            print(f"Loss plot saved to {config.OUTPUT_DIR / 'training_loss.png'}")
        except Exception as e:
            print(f"Warning: Could not plot loss: {e}")
    else:
        print("Warning: No metrics file found for plotting")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
