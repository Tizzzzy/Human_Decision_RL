"""
Train Qwen/Qwen3-4B-Instruct-2507 using DPO (Direct Preference Optimization)
with preference pairs from Qwen/Qwen3.5-9B judge.

DPO learns from paired preferences without requiring a reward model.
For each (text, chosen_explanation, rejected_explanations) tuple,
we optimize the model to prefer chosen over rejected.
"""

import json
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CACHE_DIR = "/projects/p32143/cache/huggingface"
BASE_OUTPUT_DIR = "/projects/p32143/cache/rl_dpo_qwen34b"

PREFERENCE_AI_JSON = "preference_pairs_ai.json"
PREFERENCE_HUMAN_JSON = "preference_pairs_human.json"

# DPO hyperparameters
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
BETA = 0.1  # Weight of the DPO loss (KL divergence penalty)
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256

# ==========================================
# Helper Functions
# ==========================================
def load_preferences(json_files):
    """Load preference pairs from JSON files."""
    all_pairs = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_pairs.update(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return all_pairs

def create_dpo_dataset(preference_pairs, tokenizer):
    """
    Create DPO dataset from preference pairs.

    DPO format requires:
    - prompt: the input to the model
    - chosen: the preferred completion
    - rejected: the non-preferred completion

    In our case:
    - prompt: "Analyze this text for AI/human markers:\n\n{text}\n\nExplanation:"
    - chosen: the judge's selected explanation
    - rejected: one of the rejected explanations (or combined)
    """
    prompts = []
    chosen = []
    rejected = []

    for text_id, pair in preference_pairs.items():
        text = pair['text']
        chosen_exp = pair['chosen_explanation']
        rejected_exps = pair['rejected_explanations']

        # Create prompt for generating explanations
        instruction = f"""Task: Analyze the provided social media post for linguistic markers of AI or human authorship.

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
```
{text}
```

Explanation:\n"""

        messages = [{"role": "user", "content": instruction}]

        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        prompts.append(formatted_prompt)
        chosen.append(chosen_exp)

        # For simplicity, use the first rejected explanation
        # In practice, could create multiple pairs per text
        if rejected_exps:
            rejected.append(rejected_exps[0]['text'])
        else:
            # Fallback if no rejected explanations
            rejected.append("No alternative explanation available.")

    return {
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected
    }

def main():
    print("="*80)
    print("DPO RL Training: Qwen/Qwen3-4B-Instruct-2507")
    print("="*80)

    # ==========================================
    # 1. Load preference pairs
    # ==========================================
    print("\n[1/5] Loading preference pairs...")
    preference_pairs = load_preferences([PREFERENCE_AI_JSON, PREFERENCE_HUMAN_JSON])

    if not preference_pairs:
        print("❌ No preference pairs found. Exiting.")
        return

    print(f"Loaded {len(preference_pairs)} preference pairs")

    # ==========================================
    # 2. Create DPO dataset
    # ==========================================
    print("\n[2/5] Creating DPO dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_data = create_dpo_dataset(preference_pairs, tokenizer)
    dataset = Dataset.from_dict(dpo_data)
    print(f"Created dataset with {len(dataset)} examples")

    # Sanity check
    print(f"\nSample data:")
    print(f"  Prompt: {dataset[0]['prompt'][:200]}...")
    print(f"  Chosen (first 100 chars): {dataset[0]['chosen'][:100]}...")
    print(f"  Rejected (first 100 chars): {dataset[0]['rejected'][:100]}...")

    # ==========================================
    # 3. Load model and tokenizer
    # ==========================================
    print("\n[3/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Model size: {model.num_parameters():,} parameters")

    # ==========================================
    # 4. Setup LoRA for efficient training
    # ==========================================
    print("\n[4/5] Setting up LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # beta=BETA,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # from peft import get_peft_model
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # ==========================================
    # 5. DPO Training
    # ==========================================
    print("\n[5/5] Setting up DPO trainer...")

    training_args = DPOConfig(
        output_dir=BASE_OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        logging_steps=50,
        logging_dir=f"{BASE_OUTPUT_DIR}/logs",
        save_steps=200,
        save_total_limit=3,
        weight_decay=0.01,
        bf16=True,
        beta=BETA,
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model=model, 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    dpo_trainer.train()

    # ==========================================
    # Save final model
    # ==========================================
    print("\nSaving final model...")
    dpo_trainer.save_model(os.path.join(BASE_OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(BASE_OUTPUT_DIR, "final_model"))

    print("\n" + "="*80)
    print("DPO Training Complete")
    print("="*80)
    print(f"Final model saved to: {os.path.join(BASE_OUTPUT_DIR, 'final_model')}")

if __name__ == "__main__":
    main()
