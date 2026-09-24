"""
Dataset builder: reads Task 1 output (undecided text pools) and converts to
HuggingFace datasets.Dataset format compatible with GRPOTrainer.

Each sample includes:
  - "prompt": the policy's generation prompt (already chat-templated, ready for generation)
  - "text_id": for debugging/tracking
  - "label_truth": "AI" or "Human" (ground truth)
  - "label_truth_numeric": 1 or 0 (ground truth in numeric form, used by reward function)
  - "text_content": the original text (used by reward function to build eval prompt)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import datasets

# Handle relative imports when run from different contexts
try:
    from . import config, prompt_templates
except ImportError:
    import config
    import prompt_templates


def load_pool(jsonl_path: Path) -> List[Dict]:
    """
    Load a JSONL file (output from Task 1) into a list of dicts.

    Each line is a JSON record with keys: text_id, label_truth, label_truth_numeric, text_path, source_dir.
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    return records


def build_hf_dataset(
    pool_records: List[Dict],
    policy_tokenizer,
) -> datasets.Dataset:
    """
    Convert a pool of records into a HuggingFace Dataset.

    For each record:
      1. Read text_content from disk (text_path)
      2. Build policy_prompt via build_policy_prompt
      3. Emit a row with: prompt, text_id, label_truth, label_truth_numeric, text_content

    Args:
        pool_records: List of dicts from load_pool()
        policy_tokenizer: Qwen tokenizer with apply_chat_template method

    Returns:
        datasets.Dataset with columns: prompt, text_id, label_truth, label_truth_numeric, text_content
    """
    rows = []

    for record in pool_records:
        text_path = record["text_path"]

        # Read text content from disk
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text_content = f.read().strip()
        except FileNotFoundError:
            print(f"[Warning] Text file not found: {text_path}, skipping record")
            continue

        # Build the policy prompt (chat-templated, ready for generation)
        prompt = prompt_templates.build_policy_prompt(text_content, policy_tokenizer)

        # Emit row
        row = {
            "prompt": prompt,
            "text_id": record["text_id"],
            "label_truth": record["label_truth"],
            "label_truth_numeric": record["label_truth_numeric"],
            "text_content": text_content,
        }
        rows.append(row)

    # Convert to HuggingFace Dataset
    dataset = datasets.Dataset.from_list(rows)
    print(f"[Dataset] Loaded {len(dataset)} samples from {len(pool_records)} pool records")
    return dataset


def get_train_dataset(policy_tokenizer) -> datasets.Dataset:
    """
    Load and build the training dataset from Task 1's train split.
    """
    print(f"[Dataset] Loading training pool from {config.TRAIN_POOL_PATH}...")
    pool_records = load_pool(config.TRAIN_POOL_PATH)
    dataset = build_hf_dataset(pool_records, policy_tokenizer)
    print(f"[Dataset] Training dataset size: {len(dataset)}")
    return dataset


def get_eval_dataset(policy_tokenizer) -> datasets.Dataset:
    """
    Load and build the evaluation dataset from Task 1's test split.
    """
    print(f"[Dataset] Loading evaluation pool from {config.TEST_POOL_PATH}...")
    pool_records = load_pool(config.TEST_POOL_PATH)
    dataset = build_hf_dataset(pool_records, policy_tokenizer)
    print(f"[Dataset] Evaluation dataset size: {len(dataset)}")
    return dataset
