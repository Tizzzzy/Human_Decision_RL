"""
Inference script: Use Anthropic Claude API to predict AI vs. Human authorship.

Loads the input JSON file and generates a prediction for each text using Claude.
Saves results with both original labels and the new predictions.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
import anthropic

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path("/gpfs/projects/p32143/RL_human_decision")
INPUT_FILE = PROJECT_ROOT / "RL/get_new_explanation/inference_humanRL.json"
OUTPUT_FILE = PROJECT_ROOT / "RL/get_new_explanation/claude_predictions.json"

# Claude API Configuration
CLAUDE_MODEL_NAME = "claude-opus-5" 
MAX_NEW_TOKENS = 10         
TEMPERATURE = 0.0           # Greedy decoding equivalent for deterministic classification

# System instructions are strictly enforced by Claude
POLICY_INSTRUCTION = """Task: Analyze the provided text and predict whether it was written by an AI or a Human.

Constraints:
1. Output EXACTLY and ONLY the word "AI" or "Human".
2. Do NOT provide an explanation, introduction, or final verdict.
3. Do NOT use any punctuation."""

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

def generate_prediction(
    client: anthropic.Anthropic,
    text: str,
    max_retries: int = 3
) -> str:
    """Generate a prediction using the Claude API with basic retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL_NAME,
                max_tokens=MAX_NEW_TOKENS,
                # temperature=TEMPERATURE,
                system=POLICY_INSTRUCTION,
                messages=[
                    {
                        "role": "user",
                        "content": f"Text:\n{text}\n\nPrediction:"
                    }
                ]
            )
            # Extract and clean the generated token
            return response.content[0].text.strip()
            
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"[Warning] Rate limit hit. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"[Error] Rate limit exceeded after {max_retries} attempts.")
                return "[FAILED - RATE LIMIT]"
        except Exception as e:
            print(f"[Error] API Call failed: {e}")
            return "[FAILED]"

def process_records(
    records: List[Dict[str, Any]],
    client: anthropic.Anthropic,
) -> List[Dict[str, Any]]:
    """Process all records and generate new predictions."""
    results = []

    print(f"\n[Inference] Generating predictions for {len(records)} texts via {CLAUDE_MODEL_NAME}...")
    for i, record in enumerate(tqdm(records, desc="Generating predictions")):
        text = record["text"]
        prediction = generate_prediction(client, text)
        print(f"[Record {i+1}] text_id={record.get('text_id', 'N/A')} | Prediction: {prediction}")

        # Add prediction to record
        result = record.copy()
        result["model_prediction"] = prediction
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
    print("       INFERENCE: Generate Predictions with Anthropic Claude API")
    print("="*70)

    # Initialize Anthropic client (automatically picks up ANTHROPIC_API_KEY from env)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[Error] ANTHROPIC_API_KEY environment variable not set. Please run `export ANTHROPIC_API_KEY='your-key'`")
        return
        
    client = anthropic.Anthropic()

    # Load input data
    records = load_input_data(INPUT_FILE)

    # Generate new predictions
    results = process_records(records, client)

    # Save results
    save_output(results, OUTPUT_FILE)

    print("\n" + "="*70)
    print(f"Inference complete! Results saved to {OUTPUT_FILE}")
    print("="*70 + "\n")

    # Print sample results
    print("Sample output (first 5 records):")
    for i, record in enumerate(results[:5]):
        print(f"\n[Record {i+1}] text_id={record.get('text_id', 'N/A')}")
        print(f"  True Label:       {record.get('label', 'N/A')}")
        print(f"  Model Prediction: {record.get('model_prediction', 'N/A')}")

if __name__ == "__main__":
    main()