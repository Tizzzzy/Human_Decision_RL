#!/usr/bin/env python3
"""
Compare explanations generated WITH vs WITHOUT ground truth.

This script loads both versions of explanations and shows:
- Specific samples side-by-side
- Statistical differences
- Quality metrics for each mode
"""

import json
import glob
import os
from typing import Dict, List, Tuple

# ==========================================
# Configuration
# ==========================================
RESULTS_DIR = "/projects/p32143/RL_human_decision/baseline/GPT/inference_results"


def find_latest_files(pattern: str) -> List[str]:
    """Find latest files matching pattern, grouped by mode."""
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    return sorted(files)


def load_results(filepath: str) -> List[Dict]:
    """Load results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def count_bullets(explanation: str) -> int:
    """Count bullet points in explanation."""
    lines = explanation.strip().split('\n')
    bullets = [l for l in lines if l.strip().startswith(('- ', '• ', '* '))]
    return len(bullets)


def extract_keywords(explanation: str) -> List[str]:
    """Extract potential keywords from explanation."""
    # Simple extraction: words after "Look for"
    keywords = []
    for line in explanation.split('\n'):
        if 'look for' in line.lower():
            # Extract text after "look for"
            parts = line.split('look for')
            if len(parts) > 1:
                text = parts[1].strip()
                # Remove period and quotes
                text = text.rstrip('.').strip('"\'')
                if text:
                    keywords.append(text[:50])  # First 50 chars
    return keywords


def analyze_explanation(explanation: str) -> Dict:
    """Analyze a single explanation."""
    return {
        'length': len(explanation),
        'bullet_count': count_bullets(explanation),
        'keywords': extract_keywords(explanation),
        'has_like': 'like' in explanation.lower(),
        'has_example': 'like' in explanation.lower(),
    }


def main():
    print("\n" + "=" * 80)
    print("Ground Truth Mode Comparison: WITH vs WITHOUT")
    print("=" * 80)

    # Find latest files
    with_truth_files = find_latest_files("*_with_truth_*.json")
    without_truth_files = find_latest_files("*_without_truth_*.json")

    if not with_truth_files:
        print("\n❌ No 'with_truth' results found")
        print("Run: python gpt_inference.py  (with USE_GROUND_TRUTH = True)")
        return

    if not without_truth_files:
        print("\n⚠️  No 'without_truth' results found")
        print("To compare, also run: python gpt_inference.py  (with USE_GROUND_TRUTH = False)")
        print("\nProceeding with WITH truth only...")

    # Load data
    latest_with = with_truth_files[-1]
    print(f"\n✓ Loaded with truth: {os.path.basename(latest_with)}")

    with_data = load_results(latest_with)
    without_data = load_results(without_truth_files[-1]) if without_truth_files else None

    if not with_data:
        print("❌ Failed to load data")
        return

    # Create index
    with_by_id = {r['text_id']: r for r in with_data}
    without_by_id = {r['text_id']: r for r in without_data} if without_data else {}

    common_ids = set(with_by_id.keys()) & set(without_by_id.keys()) if without_data else []

    # Statistics
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    print(f"\nWith Ground Truth:")
    print(f"  Total explanations: {len(with_data)}")

    if without_data:
        print(f"\nWithout Ground Truth:")
        print(f"  Total explanations: {len(without_data)}")
        print(f"  Common texts: {len(common_ids)}")

    # Analyze all explanations
    with_analyses = {
        text_id: analyze_explanation(r['explanation'])
        for text_id, r in with_by_id.items()
    }

    without_analyses = {
        text_id: analyze_explanation(r['explanation'])
        for text_id, r in without_by_id.items()
    } if without_data else {}

    # Aggregate metrics
    def aggregate_metrics(analyses: Dict) -> Dict:
        if not analyses:
            return {}
        lengths = [a['length'] for a in analyses.values()]
        bullets = [a['bullet_count'] for a in analyses.values()]
        return {
            'avg_length': sum(lengths) / len(lengths),
            'avg_bullets': sum(bullets) / len(bullets),
            'min_bullets': min(bullets),
            'max_bullets': max(bullets),
            'with_examples': sum(1 for a in analyses.values() if a['has_example']),
        }

    with_metrics = aggregate_metrics(with_analyses)
    without_metrics = aggregate_metrics(without_analyses)

    print(f"\nWith Ground Truth Metrics:")
    print(f"  Avg length: {with_metrics.get('avg_length', 0):.0f} chars")
    print(f"  Avg bullets: {with_metrics.get('avg_bullets', 0):.1f}")
    print(f"  Bullet range: {with_metrics.get('min_bullets', 0)}-{with_metrics.get('max_bullets', 0)}")
    print(f"  With examples: {with_metrics.get('with_examples', 0)}/{len(with_analyses)}")

    if without_metrics:
        print(f"\nWithout Ground Truth Metrics:")
        print(f"  Avg length: {without_metrics.get('avg_length', 0):.0f} chars")
        print(f"  Avg bullets: {without_metrics.get('avg_bullets', 0):.1f}")
        print(f"  Bullet range: {without_metrics.get('min_bullets', 0)}-{without_metrics.get('max_bullets', 0)}")
        print(f"  With examples: {without_metrics.get('with_examples', 0)}/{len(without_analyses)}")

        # Comparison
        print(f"\nDifferences:")
        length_diff = with_metrics.get('avg_length', 0) - without_metrics.get('avg_length', 0)
        bullet_diff = with_metrics.get('avg_bullets', 0) - without_metrics.get('avg_bullets', 0)
        print(f"  Length: {length_diff:+.0f} chars")
        print(f"  Bullets: {bullet_diff:+.1f} avg")

    # By Label Analysis
    print(f"\n" + "=" * 80)
    print("Analysis by Label")
    print("=" * 80)

    for label in ["AI", "Human"]:
        with_label = [r for r in with_data if r['label'] == label]
        without_label = [r for r in without_data if r['label'] == label] if without_data else []

        with_label_analyses = {
            r['text_id']: analyze_explanation(r['explanation'])
            for r in with_label
        }
        without_label_analyses = {
            r['text_id']: analyze_explanation(r['explanation'])
            for r in without_label
        } if without_label else {}

        with_label_metrics = aggregate_metrics(with_label_analyses)
        without_label_metrics = aggregate_metrics(without_label_analyses)

        print(f"\n{label} Texts:")
        print(f"  With truth: {len(with_label)} texts, "
              f"avg {with_label_metrics.get('avg_bullets', 0):.1f} bullets")
        if without_label:
            print(f"  Without truth: {len(without_label)} texts, "
                  f"avg {without_label_metrics.get('avg_bullets', 0):.1f} bullets")

    # Sample Comparisons
    print("\n" + "=" * 80)
    print("Sample Explanations")
    print("=" * 80)

    # Show samples for both AI and Human
    for label in ["AI", "Human"]:
        sample_ids = [r['text_id'] for r in with_data if r['label'] == label][:2]

        for text_id in sample_ids:
            with_r = with_by_id[text_id]
            without_r = without_by_id.get(text_id) if without_data else None

            print(f"\n{'─' * 80}")
            print(f"Text ID: {text_id} ({label})")
            print(f"{'─' * 80}")

            print(f"\nWITH Ground Truth:")
            print(with_r['explanation'][:400])
            if len(with_r['explanation']) > 400:
                print("...")

            if without_r:
                print(f"\nWITHOUT Ground Truth:")
                print(without_r['explanation'][:400])
                if len(without_r['explanation']) > 400:
                    print("...")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Ground truth mode allows the model to generate more targeted explanations.
Key observations:
- Check if explanations are more specific with ground truth
- Look for category-specific language and markers
- Compare explanation quality between modes
- Verify model doesn't reveal the actual label

To generate both versions:
  1. Run with: USE_GROUND_TRUTH = True  (default)
  2. Run with: USE_GROUND_TRUTH = False
  3. Compare outputs with this script
""")


if __name__ == "__main__":
    main()
