"""
Inference script: Load a trained checkpoint and generate explanations for test texts.
Stores results (text + explanation) in JSON format.
"""

import os
import json
import glob
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# Configuration
# ==========================================
BASE_DIR = "/projects/p32143/RL_human_decision/text_data/social"
CACHE_DIR = "/projects/p32143/cache/huggingface"
OUTPUT_DIR = "/projects/p32143/cache/rl_dpo_qwen34b/final_model"
TEST_CSV = "/projects/p32143/RL_human_decision/simulator/test_simulator.csv"
RESULTS_DIR = "/projects/p32143/RL_human_decision/baseline/preference_RL/inference_results"

POLICY_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Policy instruction (MUST match exactly what rl.py used)
POLICY_INSTRUCTION = """Task: Analyze the provided Social Media post for linguistic markers of AI or human authorship.

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

Explanation:
"""

# MAX_TEXT_CHARS = 2000


def find_latest_checkpoint(output_dir: str) -> str:
    """Find the latest checkpoint by parsing checkpoint-* dirs and extracting step number."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")

    # Extract step number from checkpoint path
    checkpoints_with_steps = []
    for ckpt in checkpoints:
        try:
            step = int(os.path.basename(ckpt).split("-")[1])
            checkpoints_with_steps.append((step, ckpt))
        except (IndexError, ValueError):
            continue

    if not checkpoints_with_steps:
        raise FileNotFoundError(f"No valid checkpoints found in {output_dir}")

    # Return the checkpoint with the highest step
    latest_step, latest_ckpt = max(checkpoints_with_steps, key=lambda x: x[0])
    print(f"[Checkpoint] Found latest checkpoint at step {latest_step}: {latest_ckpt}")
    return latest_ckpt


def build_policy_prompt(text: str, tokenizer: AutoTokenizer) -> str:
    """Build policy prompt in Qwen instruct format."""
    # text = text[:MAX_TEXT_CHARS]
    messages = [{"role": "user", "content": POLICY_INSTRUCTION.format(text=text)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def read_text_file(path: str) -> str:
    """Read text file from the social media directory."""
    try:
        full_path = os.path.join(BASE_DIR, path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return ""


@torch.no_grad()
def generate_explanations(
    policy_model: PeftModel,
    policy_tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int = 160,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """Generate explanations for a batch of texts."""
    prompts = [build_policy_prompt(t, policy_tokenizer) for t in texts]

    # Tokenize
    enc = policy_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(policy_model.device) for k, v in enc.items()}

    # Generate
    output_ids = policy_model.generate(
        **enc,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=policy_tokenizer.eos_token_id,
    )

    # Decode: extract only the generated part (remove prompt)
    completions = []
    prompt_lens = [len(enc["input_ids"][i]) for i in range(len(enc["input_ids"]))]
    for i, output_id in enumerate(output_ids):
        generated_ids = output_id[prompt_lens[i]:]
        completion = policy_tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion.strip())

    return completions


def main():
    print("\n" + "=" * 70)
    print("RL Inference: Generate explanations using trained policy")
    print("=" * 70)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find latest checkpoint
    # checkpoint_path = find_latest_checkpoint(OUTPUT_DIR)
    checkpoint_path = OUTPUT_DIR

    # Load policy tokenizer and model
    print(f"\n[Model] Loading tokenizer from {POLICY_MODEL}...")
    policy_tokenizer = AutoTokenizer.from_pretrained(
        POLICY_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    policy_tokenizer.padding_side = "left"
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    print(f"[Model] Loading base model {POLICY_MODEL}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    print(f"[Model] Loading LoRA adapter from {checkpoint_path}...")
    policy_model = PeftModel.from_pretrained(policy_model, checkpoint_path)
    policy_model.eval()

    for p in policy_model.parameters():
        p.requires_grad_(False)

    print("[Model] Ready for inference.\n")

    # Load test CSV
    print(f"[Data] Loading test CSV from {TEST_CSV}...")
    df = pd.read_csv(TEST_CSV)
    print(f"[Data] Loaded {len(df)} test samples")

    # Read texts
    print("[Data] Reading text files...")
    texts = []
    text_ids = []
    labels = []
    for idx, row in df.iterrows():
        text = read_text_file(row["text_path"])
        if text:
            texts.append(text)
            text_ids.append(row["text_id"])
            labels.append(row["label"])

    print(f"[Data] Successfully loaded {len(texts)} texts\n")

    # Generate explanations in batches
    batch_size = 8
    all_explanations = []

    print("[Inference] Generating explanations...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = text_ids[i:i+batch_size]

        print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...", end=" ", flush=True)
        explanations = generate_explanations(
            policy_model,
            policy_tokenizer,
            batch_texts,
            max_length=1024,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.1,
        )
        all_explanations.extend(explanations)
        print("done")

    # Build results
    print("\n[Results] Building results JSON...")
    results = []
    for text_id, text, label, explanation in zip(text_ids, texts, labels, all_explanations):
        results.append({
            "text_id": text_id,
            "label": label,
            "text": text,
            "explanation": explanation,
        })

    # Save to JSON
    results_file = os.path.join(RESULTS_DIR, f"inference_{Path(checkpoint_path).name}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Results] Saved {len(results)} results to {results_file}")

    return results


if __name__ == "__main__":
    main()
