"""
Task 1: Identify all (text_id, label_truth) pairs on disk with NO human decisions yet.

Inputs:
  - statistic/text_2026-09-02.db (table text_guess)
  - webscrape/social/SocialMedia_Reddit/ (2260 human-written texts)
  - webscrape/social/SocialMedia_rewrite/ (2260 AI-generated texts)

Outputs:
  - undecided_pool.jsonl (full pool before split)
  - train_undecided.jsonl (85% train)
  - test_undecided.jsonl (15% test, for eval-only)
  - summary.json (count breakdown for sanity checks)

The undecided pool is used as the RL training/testing set to avoid data contamination
(never train on texts where humans already provided ground truth).
"""

import json
import sqlite3
from pathlib import Path
from typing import Set, Tuple, List, Dict
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np


# ============================================================================
# Configuration (hard-coded paths matching plan)
# ============================================================================
PROJECT_ROOT = Path("/gpfs/projects/p32143/RL_human_decision")
DB_PATH = PROJECT_ROOT / "statistic/text_2026-09-02.db"
WEBSCRAPE_REDDIT_DIR = PROJECT_ROOT / "webscrape/social/SocialMedia_Reddit"
WEBSCRAPE_REWRITE_DIR = PROJECT_ROOT / "webscrape/social/SocialMedia_rewrite"
TEXT_DATA_ROOT = PROJECT_ROOT / "text_data/social"
OUTPUT_DIR = PROJECT_ROOT / "RL/get_undecide_text"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Functions
# ============================================================================
def load_decided_keys(db_path: Path) -> Set[Tuple[str, str]]:
    """
    Load all (text_id, label_truth) pairs that have human decisions in the DB.

    Returns:
        set of tuples: {(text_id, label_truth), ...}
        e.g., {('reddit_2459', 'AI'), ('reddit_1578', 'Human'), ...}
    """
    decided = set()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT text_id, label_truth FROM text_guess WHERE text_id IS NOT NULL"
    )

    for row in cursor.fetchall():
        text_id, label_truth = row
        # Preserve case: DB has 'AI' / 'Human' (capitalized)
        decided.add((text_id, label_truth))

    conn.close()
    print(f"[DB] Loaded {len(decided)} distinct (text_id, label_truth) pairs with decisions")
    return decided


def enumerate_disk_pool(
    reddit_dir: Path,
    rewrite_dir: Path,
    text_data_root: Path,
) -> List[Dict]:
    """
    Enumerate all (text, label) pairs on disk.

    Args:
        reddit_dir: webscrape/social/SocialMedia_Reddit (human texts)
        rewrite_dir: webscrape/social/SocialMedia_rewrite (AI texts)
        text_data_root: text_data/social (where downstream code reads from)

    Returns:
        List of dicts: {text_id, label_truth, label_truth_numeric, text_path, source_dir}
        text_path points to text_data/social/ for consistency with existing scripts.
    """
    pool = []

    # Human-written texts (from SocialMedia_Reddit)
    reddit_files = sorted(glob(str(reddit_dir / "*.txt")))
    for file_path in reddit_files:
        text_id = Path(file_path).stem  # e.g., 'reddit_1001'
        label_truth = "Human"
        label_truth_numeric = 0
        text_path = str(text_data_root / "SocialMedia_Reddit" / f"{text_id}.txt")
        source_dir = "SocialMedia_Reddit"

        pool.append({
            "text_id": text_id,
            "label_truth": label_truth,
            "label_truth_numeric": label_truth_numeric,
            "text_path": text_path,
            "source_dir": source_dir,
        })

    print(f"[Disk] Found {len(reddit_files)} human-written texts (SocialMedia_Reddit)")

    # AI-generated texts (from SocialMedia_rewrite)
    rewrite_files = sorted(glob(str(rewrite_dir / "*.txt")))
    for file_path in rewrite_files:
        text_id = Path(file_path).stem  # e.g., 'reddit_1001'
        label_truth = "AI"
        label_truth_numeric = 1
        text_path = str(text_data_root / "SocialMedia_rewrite" / f"{text_id}.txt")
        source_dir = "SocialMedia_rewrite"

        pool.append({
            "text_id": text_id,
            "label_truth": label_truth,
            "label_truth_numeric": label_truth_numeric,
            "text_path": text_path,
            "source_dir": source_dir,
        })

    print(f"[Disk] Found {len(rewrite_files)} AI-generated texts (SocialMedia_rewrite)")
    print(f"[Disk] Total disk pool: {len(pool)} (text_id, label_truth) pairs")
    return pool


def compute_undecided(disk_pool: List[Dict], decided_keys: Set[Tuple[str, str]]) -> List[Dict]:
    """
    Filter disk pool to only include (text_id, label_truth) pairs NOT in the decided set.
    """
    undecided = []
    for record in disk_pool:
        key = (record["text_id"], record["label_truth"])
        if key not in decided_keys:
            undecided.append(record)

    print(f"[Filter] Undecided pool size: {len(undecided)}")
    return undecided


def split_and_write(
    undecided: List[Dict],
    output_dir: Path,
    test_size: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Split undecided pool into train/test, stratified by label_truth_numeric, and write to JSONL.
    Also write the full undecided pool and a summary JSON.
    """
    # Stratify by label to ensure balanced AI/Human split
    label_numeric = [r["label_truth_numeric"] for r in undecided]

    train_records, test_records = train_test_split(
        undecided,
        test_size=test_size,
        random_state=seed,
        stratify=label_numeric,
    )

    # Write full undecided pool
    undecided_pool_file = output_dir / "undecided_pool.jsonl"
    with open(undecided_pool_file, "w") as f:
        for record in undecided:
            f.write(json.dumps(record) + "\n")
    print(f"[Write] Wrote full undecided pool to {undecided_pool_file}")

    # Write train split
    train_file = output_dir / "train_undecided.jsonl"
    with open(train_file, "w") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")
    print(f"[Write] Wrote train split ({len(train_records)} records) to {train_file}")

    # Write test split
    test_file = output_dir / "test_undecided.jsonl"
    with open(test_file, "w") as f:
        for record in test_records:
            f.write(json.dumps(record) + "\n")
    print(f"[Write] Wrote test split ({len(test_records)} records) to {test_file}")

    # Compute summary stats
    disk_pool_ai = sum(1 for r in undecided if r["label_truth"] == "AI")
    disk_pool_human = sum(1 for r in undecided if r["label_truth"] == "Human")
    train_ai = sum(1 for r in train_records if r["label_truth"] == "AI")
    train_human = sum(1 for r in train_records if r["label_truth"] == "Human")
    test_ai = sum(1 for r in test_records if r["label_truth"] == "AI")
    test_human = sum(1 for r in test_records if r["label_truth"] == "Human")

    summary = {
        "disk_pool_total": len(undecided),
        "disk_pool_ai": disk_pool_ai,
        "disk_pool_human": disk_pool_human,
        "train_count": len(train_records),
        "train_ai": train_ai,
        "train_human": train_human,
        "test_count": len(test_records),
        "test_ai": test_ai,
        "test_human": test_human,
        "split_ratio": (1 - test_size),
        "random_seed": seed,
    }

    # Write summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Write] Wrote summary to {summary_file}")

    # Print summary to console
    print("\n" + "="*70)
    print("                   UNDECIDED POOL SUMMARY                        ")
    print("="*70)
    print(f"Total disk pool:     {summary['disk_pool_total']} ({summary['disk_pool_ai']} AI, {summary['disk_pool_human']} Human)")
    print(f"Train split:         {summary['train_count']} ({summary['train_ai']} AI, {summary['train_human']} Human)")
    print(f"Test split:          {summary['test_count']} ({summary['test_ai']} AI, {summary['test_human']} Human)")
    print(f"Split ratio:         {summary['split_ratio']:.0%} train / {test_size:.0%} test")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("               BUILDING UNDECIDED TEXT POOL (Task 1)             ")
    print("="*70 + "\n")

    # Step 1: Load decided keys from DB
    print("[Step 1/4] Loading decided (text_id, label_truth) pairs from DB...")
    decided_keys = load_decided_keys(DB_PATH)

    # Step 2: Enumerate disk pool
    print("\n[Step 2/4] Enumerating disk pool...")
    disk_pool = enumerate_disk_pool(
        WEBSCRAPE_REDDIT_DIR,
        WEBSCRAPE_REWRITE_DIR,
        TEXT_DATA_ROOT,
    )

    # Step 3: Compute undecided
    print("\n[Step 3/4] Computing undecided pool (set difference)...")
    undecided = compute_undecided(disk_pool, decided_keys)

    # Step 4: Split and write
    print("\n[Step 4/4] Splitting and writing outputs...")
    split_and_write(undecided, OUTPUT_DIR, test_size=0.15, seed=42)

    print("[Done] Task 1 complete!")


if __name__ == "__main__":
    main()
