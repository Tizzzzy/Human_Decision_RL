"""
Dataset loading from text_guess table and text files.

Loads unique (text_id, label_truth) pairs from the database,
reads text content, and builds HF Dataset for GRPO training.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

try:
    from . import config, prompt_templates
except ImportError:
    import config
    import prompt_templates


def load_text_from_file(text_path: Path) -> str:
    """
    Load text content from a file.

    Args:
        text_path (Path): Path to text file

    Returns:
        str: Text content
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {text_path}: {e}")
        return ""


def get_text_path(text_id: str, label_truth: str) -> Path:
    """
    Resolve the path to a text file based on text_id and label_truth.

    Args:
        text_id (str): Text identifier
        label_truth (str): Ground truth label ("AI" or "Human")

    Returns:
        Path: Full path to the text file
    """
    if label_truth == "AI":
        return config.TEXT_DATA_ROOT / "SocialMedia_rewrite" / f"{text_id}.txt"
    elif label_truth == "Human":
        return config.TEXT_DATA_ROOT / "SocialMedia_Reddit" / f"{text_id}.txt"
    else:
        raise ValueError(f"Unknown label_truth: {label_truth}")


def load_distinct_texts_from_db() -> List[Dict]:
    """
    Load all unique (text_id, label_truth) pairs from the text_guess table.

    Returns:
        list[dict]: List of dicts with keys:
            - text_id (str)
            - label_truth (str): "AI" or "Human"
            - label_truth_numeric (int): 1 for AI, 0 for Human
            - text_path (Path)
            - text_content (str)
    """
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()

    # Get all distinct (text_id, label_truth) pairs
    query = """
        SELECT DISTINCT text_id, label_truth
        FROM text_guess
        WHERE text_id IS NOT NULL
        ORDER BY text_id, label_truth
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    records = []
    for text_id, label_truth in rows:
        # Resolve text path
        text_path = get_text_path(text_id, label_truth)

        # Skip if file doesn't exist
        if not text_path.exists():
            print(f"Warning: Text file not found: {text_path}")
            continue

        # Load text content
        text_content = load_text_from_file(text_path)
        if not text_content:
            print(f"Warning: Empty text content from {text_path}")
            continue

        # Numeric label: 1 for AI, 0 for Human
        label_numeric = 1 if label_truth == "AI" else 0

        records.append({
            "text_id": text_id,
            "label_truth": label_truth,
            "label_truth_numeric": label_numeric,
            "text_path": str(text_path),
            "text_content": text_content,
        })

    print(f"Loaded {len(records)} unique (text_id, label_truth) pairs from DB")
    return records


def build_hf_dataset(records: List[Dict], tokenizer) -> HFDataset:
    """
    Convert records to HuggingFace Dataset with prompts.

    Args:
        records (list[dict]): Records from load_distinct_texts_from_db
        tokenizer: HF tokenizer

    Returns:
        datasets.Dataset: HF dataset with columns:
            - prompt (str): Formatted prompt
            - text_id (str)
            - label_truth (str)
            - label_truth_numeric (int)
            - text_content (str)
    """
    dataset_records = []

    for record in records:
        text_content = record["text_content"]

        # Build policy prompt
        prompt = prompt_templates.build_policy_prompt(text_content, tokenizer)

        dataset_records.append({
            "prompt": prompt,
            "text_id": record["text_id"],
            "label_truth": record["label_truth"],
            "label_truth_numeric": record["label_truth_numeric"],
            "text_content": text_content,
        })

    return HFDataset.from_list(dataset_records)


def get_train_test_datasets(tokenizer) -> Tuple[HFDataset, HFDataset]:
    """
    Load and split datasets into train and test.

    Uses stratified split by label to maintain class balance.

    Args:
        tokenizer: HF tokenizer

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Load all records from DB
    records = load_distinct_texts_from_db()

    # Optionally limit for debugging
    if config.MAX_TEXTS is not None:
        records = records[:config.MAX_TEXTS]
        print(f"Limited to {len(records)} records for debugging")

    # Stratified split
    train_records, test_records = train_test_split(
        records,
        test_size=1.0 - config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=[r["label_truth_numeric"] for r in records]
    )

    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # Build HF datasets
    train_dataset = build_hf_dataset(train_records, tokenizer)
    test_dataset = build_hf_dataset(test_records, tokenizer)

    return train_dataset, test_dataset


def get_train_dataset(tokenizer) -> HFDataset:
    """Load train dataset."""
    train_dataset, _ = get_train_test_datasets(tokenizer)
    return train_dataset


def get_eval_dataset(tokenizer) -> HFDataset:
    """Load eval dataset."""
    _, test_dataset = get_train_test_datasets(tokenizer)
    return test_dataset
