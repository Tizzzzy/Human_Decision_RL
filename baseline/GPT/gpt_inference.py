"""
GPT Inference Script: Generate explanations for test texts using OpenAI's GPT models.

Loads test texts from simulator CSV and generates explanations using GPT-4 (or GPT-3.5-turbo).
Output format matches preference_RL/inference.py for consistency.

Stores results (text + explanation) in JSON format.
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import time
from datetime import datetime

import openai
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================
BASE_DIR = "/projects/p32143/RL_human_decision/text_data/social"
TEST_CSV = "/projects/p32143/RL_human_decision/simulator/test_simulator.csv"
RESULTS_DIR = "/projects/p32143/RL_human_decision/baseline/GPT/inference_results"

# OpenAI Configuration
GPT_MODEL = "gpt-5.6-luna"
API_KEY = os.getenv("OPENAI_API_KEY")  # Set via environment variable
max_completion_tokens = 1048
TOP_P = 1.0

# Rate limiting (to avoid hitting API limits)
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# Policy instruction (with ground truth - for informed explanations)
# NOTE: Ground truth is provided to the model, but should NOT be revealed in the explanation itself
POLICY_INSTRUCTION_WITH_TRUTH = """Task: Analyze the provided Social Media post for linguistic markers of AI or human authorship.

IMPORTANT INFORMATION (DO NOT REVEAL IN YOUR EXPLANATION):
Ground truth: This post was written by a {label}.

Your task: Generate linguistic markers that would help someone detect that this post was written by a {label}, WITHOUT explicitly saying the answer. Focus on identifying the characteristic patterns and markers.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list.
2. Do NOT provide an introduction, a final verdict, or a summary.
3. Do NOT mention the ground truth or the actual category (AI/Human) in your explanation.
4. Do NOT group the bullet points into sub-headings (keep it a flat list).
5. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category or type of phrasing]."
    - "Look for [linguistic category or type of phrasing] like '[short example]'."
6. When providing examples, you may list multiple words separated by commas (e.g., like "word 1", "word 2").
7. Do NOT use bolding, italics, or any special Markdown formatting.
8. Limit the output to exactly 2 to 6 bullet points.

Post:
```
{text}
```

Explanation:"""

# Original prompt (without ground truth - for comparison)
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

Explanation:"""

# Toggle between with/without ground truth
USE_GROUND_TRUTH = True  # Set to False to use original prompt without ground truth


def read_text_file(path: str) -> str:
    """Read text file from the social media directory."""
    try:
        full_path = os.path.join(BASE_DIR, path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return ""


def generate_explanation_gpt(
    client: OpenAI,
    text: str,
    label: str = None,
    model: str = GPT_MODEL,
    max_completion_tokens: int = max_completion_tokens,
    top_p: float = TOP_P,
) -> str:
    """
    Generate explanation for a single text using OpenAI's GPT API.

    Args:
        client: OpenAI client instance
        text: The social media text to analyze
        label: Ground truth label ("Human" or "AI") for informed explanations
               If provided with USE_GROUND_TRUTH=True, model knows the answer
               but is instructed not to reveal it
        model: GPT model to use
        max_completion_tokens: Maximum tokens for the response
        top_p: Nucleus sampling parameter

    Returns:
        Generated explanation text, or empty string on error
    """
    try:
        # Build the prompt based on USE_GROUND_TRUTH setting
        if USE_GROUND_TRUTH and label:
            prompt = POLICY_INSTRUCTION_WITH_TRUTH.format(text=text, label=label)
        else:
            prompt = POLICY_INSTRUCTION.format(text=text)

        # Call GPT API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
        )

        # Extract and return the generated text
        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        print(f"[Error] Failed to generate explanation: {e}")
        return ""


def generate_explanations_batch(
    client: OpenAI,
    texts: List[str],
    text_ids: List[str],
    labels: List[str] = None,
    model: str = GPT_MODEL,
    max_completion_tokens: int = max_completion_tokens,
    top_p: float = TOP_P,
) -> List[str]:
    """
    Generate explanations for a batch of texts.

    Args:
        client: OpenAI client instance
        texts: List of texts to analyze
        text_ids: List of text IDs (for logging)
        labels: List of ground truth labels ("Human" or "AI")
                Optional, used for informed explanations
        model: GPT model to use
        max_completion_tokens: Maximum tokens per response
        top_p: Nucleus sampling parameter

    Returns:
        List of generated explanations
    """
    explanations = []

    # Use ground truth if available and enabled
    truth_mode = "with ground truth" if (USE_GROUND_TRUTH and labels) else "without ground truth"
    print(f"\n[Inference] Generating explanations using {model} ({truth_mode})...")

    for i, (text, text_id) in enumerate(zip(texts, text_ids)):
        label = labels[i] if labels else None
        print(f"  [{i+1}/{len(texts)}] Generating explanation for {text_id} ({label})...", end=" ", flush=True)

        explanation = generate_explanation_gpt(
            client,
            text,
            label=label,
            model=model,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
        )

        explanations.append(explanation)
        print("done")

        # Rate limiting to avoid API throttling
        if i < len(texts) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    return explanations


def main():
    print("\n" + "=" * 70)
    print(f"GPT Inference: Generate explanations using {GPT_MODEL}")
    truth_mode = "WITH ground truth" if USE_GROUND_TRUTH else "WITHOUT ground truth"
    print(f"Mode: {truth_mode}")
    if USE_GROUND_TRUTH:
        print("Note: Model knows ground truth but will NOT reveal it in explanations")
    print("=" * 70)

    # Verify API key
    if not API_KEY:
        print("[Error] OPENAI_API_KEY environment variable not set!")
        print("[Help] Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize OpenAI client
    print(f"\n[Setup] Initializing OpenAI client...")
    client = OpenAI(api_key=API_KEY)

    # Load test CSV
    print(f"[Data] Loading test CSV from {TEST_CSV}...")
    try:
        df = pd.read_csv(TEST_CSV)
        print(f"[Data] Loaded {len(df)} test samples")
    except Exception as e:
        print(f"[Error] Failed to load CSV: {e}")
        return

    # Read texts
    print("[Data] Reading text files...")
    texts = []
    text_ids = []
    labels = []
    text_paths = []

    for idx, row in df.iterrows():
        text = read_text_file(row["text_path"])
        if text:
            texts.append(text)
            text_ids.append(row["text_id"])
            labels.append(row["label"])
            text_paths.append(row["text_path"])

    print(f"[Data] Successfully loaded {len(texts)} texts")

    if not texts:
        print("[Error] No texts loaded. Exiting.")
        return

    # Generate explanations
    all_explanations = generate_explanations_batch(
        client,
        texts,
        text_ids,
        labels=labels,  # Pass ground truth labels if USE_GROUND_TRUTH is enabled
        model=GPT_MODEL,
        max_completion_tokens=max_completion_tokens,
        top_p=TOP_P,
    )

    # Build results
    print("\n[Results] Building results JSON...")
    results = []
    for text_id, text, label, text_path, explanation in zip(
        text_ids, texts, labels, text_paths, all_explanations
    ):
        results.append({
            "text_id": text_id,
            "label": label,
            "text": text,
            "text_path": text_path,
            "explanation": explanation,
        })

    # Save to JSON with ground truth indicator in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    truth_indicator = "with_truth" if USE_GROUND_TRUTH else "without_truth"
    results_file = os.path.join(RESULTS_DIR, f"inference_gpt_{GPT_MODEL.replace('/', '_')}_{truth_indicator}_{timestamp}.json")

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Results] Saved {len(results)} results to {results_file}")
    except Exception as e:
        print(f"[Error] Failed to save results: {e}")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total texts processed: {len(results)}")
    print(f"Model used: {GPT_MODEL}")
    print(f"Max tokens: {max_completion_tokens}")
    print(f"Results file: {results_file}")

    # Count labels
    label_counts = {}
    for result in results:
        label = result["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    return results


if __name__ == "__main__":
    main()
