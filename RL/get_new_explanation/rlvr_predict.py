"""
Inference script: Use the trained RL policy to predict AI vs. Human authorship.

Loads the trained Qwen3.6-27B policy (LoRA adapter) and generates a prediction
for each text in the input JSON file. Saves results with both original labels and the new predictions.

Input: /gpfs/projects/p32143/RL_human_decision/RL/get_new_explanation/inference_humanRL.json
Output: /gpfs/projects/p32143/RL_human_decision/RL/get_new_explanation/rlvr_predictions.json
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path("/gpfs/projects/p32143/RL_human_decision")
INPUT_FILE = PROJECT_ROOT / "RL/get_new_explanation/inference_humanRL.json"
OUTPUT_FILE = PROJECT_ROOT / "RL/get_new_explanation/rlvr_predictions.json"  # Updated output filename

POLICY_MODEL_NAME = "Qwen/Qwen3.6-27B"
POLICY_CACHE_DIR = "/projects/p32143/cache/huggingface/qwen36_27b"
# LORA_ADAPTER_PATH = "/projects/p32143/RL_human_decision/baseline/rlvr/checkpoints/final_adapter"
LORA_ADAPTER_PATH = ""

# Generation parameters (Adjusted for strict classification)
MAX_NEW_TOKENS = 10         # Reduced since we only want a single word
DO_SAMPLE = False           # Greedy decoding for deterministic classification
TEMPERATURE = 1.0           # Ignored when DO_SAMPLE is False
TOP_P = 1.0                 # Ignored when DO_SAMPLE is False
REPETITION_PENALTY = 1.0    # Set to 1.0 since it's a short token prediction

# Policy prompt template (Modified for binary prediction)
POLICY_INSTRUCTION = """Task: Analyze the provided text and predict whether it was written by an AI or a Human.

Constraints:
1. Output EXACTLY and ONLY the word "AI" or "Human".
2. Do NOT provide an explanation, introduction, or final verdict.
3. Do NOT use any punctuation.

Text:
{text}

Prediction:
"""


# ============================================================================
# Functions
# ============================================================================
def load_input_data(filepath: Path) -> List[Dict[str, Any]]:
    """Load the input JSON file."""
    print(f"[Data] Loading input from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)[:300]
    print(f"[Data] Loaded {len(data)} records")
    return data


def load_policy_model_and_tokenizer():
    """
    Load the trained policy model (Qwen3.6-27B + LoRA adapter).

    Returns:
        (policy_model, policy_tokenizer)
    """
    print(f"\n[Model] Loading tokenizer from {POLICY_MODEL_NAME}")
    policy_tokenizer = AutoTokenizer.from_pretrained(
        POLICY_MODEL_NAME,
        cache_dir=POLICY_CACHE_DIR,
        trust_remote_code=True,
    )
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    print("[Model] Tokenizer loaded")

    print(f"[Model] Loading base model {POLICY_MODEL_NAME}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=POLICY_CACHE_DIR,
        trust_remote_code=True,
    )
    print(f"[Model] Base model loaded: {policy_model.num_parameters():,} parameters")

    if LORA_ADAPTER_PATH:
        print(f"[Model] Loading LoRA adapter from {LORA_ADAPTER_PATH}")
        policy_model = PeftModel.from_pretrained(policy_model, str(LORA_ADAPTER_PATH))
        print("[Model] LoRA adapter loaded")
    else:
        print("[Model] No LORA_ADAPTER_PATH provided. Using base model only.")

    policy_model.eval()
    for param in policy_model.parameters():
        param.requires_grad = False

    return policy_model, policy_tokenizer


def build_policy_prompt(text: str, tokenizer) -> str:
    """Build the policy prompt (chat-templated) and disable thinking."""
    messages = [{"role": "user", "content": POLICY_INSTRUCTION.format(text=text)}]
    
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for some transformer versions
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )


@torch.no_grad()
def generate_prediction(
    policy_model,
    policy_tokenizer,
    text: str,
    device: str = "cuda",
) -> str:
    """
    Generate a prediction for the given text using the trained policy.

    Args:
        policy_model: The trained policy model
        policy_tokenizer: The tokenizer
        text: The input text to analyze
        device: Device to run on

    Returns:
        Generated prediction string (stripped of whitespace)
    """
    # Build the prompt
    prompt = build_policy_prompt(text, policy_tokenizer)

    # Tokenize
    inputs = policy_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=5096,
    ).to(device)

    # Generate
    output_ids = policy_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE if DO_SAMPLE else None,
        top_p=TOP_P if DO_SAMPLE else None,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=DO_SAMPLE,
        pad_token_id=policy_tokenizer.eos_token_id,
    )

    # Decode: extract only the generated part (remove prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    prediction = policy_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return prediction


def process_records(
    records: List[Dict[str, Any]],
    policy_model,
    policy_tokenizer,
) -> List[Dict[str, Any]]:
    """
    Process all records and generate new predictions.

    Args:
        records: List of input records
        policy_model: The trained policy model
        policy_tokenizer: The tokenizer

    Returns:
        List of records with predictions added
    """
    results = []
    device = next(policy_model.parameters()).device

    print(f"\n[Inference] Generating predictions for {len(records)} texts...")
    for i, record in enumerate(tqdm(records, desc="Generating predictions")):
        try:
            text = record["text"]
            prediction = generate_prediction(policy_model, policy_tokenizer, text, device=str(device))

            # Add prediction to record
            result = record.copy()
            result["model_prediction"] = prediction
            results.append(result)

        except Exception as e:
            print(f"\n[Warning] Error processing record {i} (text_id={record.get('text_id')}): {e}")
            # Include record even if generation fails, with error placeholder
            result = record.copy()
            result["model_prediction"] = "[FAILED]"
            results.append(result)

    return results


def save_output(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Save the results to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Output] Saving results to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Output] Saved {len(results)} records")


def main():
    print("\n" + "="*70)
    print("       INFERENCE: Generate Predictions with Trained RL Policy")
    print("="*70)

    # Load input data
    records = load_input_data(INPUT_FILE)

    # Load trained policy model
    policy_model, policy_tokenizer = load_policy_model_and_tokenizer()

    # Generate new predictions
    results = process_records(records, policy_model, policy_tokenizer)

    # Save results
    save_output(results, OUTPUT_FILE)

    print("\n" + "="*70)
    print(f"Inference complete! Results saved to {OUTPUT_FILE}")
    print("="*70 + "\n")

    # Print sample results
    print("Sample output (first 5 records):")
    for i, record in enumerate(results[:5]):
        print(f"\n[Record {i+1}] text_id={record['text_id']}")
        print(f"  True Label:       {record['label']}")
        print(f"  Model Prediction: {record['model_prediction']}")


if __name__ == "__main__":
    main()