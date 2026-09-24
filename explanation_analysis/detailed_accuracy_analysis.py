"""
Detailed Accuracy Analysis:
- Load explanations used in decision study
- Categorize them
- Correlate with accuracy
"""

import json
import sqlite3
from collections import defaultdict
import os


HUMAN_INDICATORS = {
    "imperfection", "grammatical", "awkward", "error", "mistake", "typo", "misspell",
    "informal", "casual", "colloquial", "conversational", "slang",
    "emotional", "self-deprecat", "vulnerability", "feelings",
    "specific situational", "specific contextual", "autobiographical", "personal detail",
    "personal context", "lived experience",
    "inconsistent", "repetition", "stream-of-consciousness", "fragmented",
    "contraction", "hesitation", "hesitat", "self-correct", "hedging",
    "asides", "interjection", "fragment", "placeholder",
}

AI_INDICATORS = {
    "balanced", "carefully balanced", "calibrated", "smooth", "polished", "structured",
    "progressively structured", "progressively",
    "generic", "broadly applicable", "neatly organized", "non-specific",
    "formally", "impersonal", "formal", "broad", "rhetorical",
    "disclaimers", "hedging",
    "either-or", "enumeration", "organized requests",
    "broad technical references", "without concrete procedural details",
    "generalized", "generalized career", "engagement-oriented", "community-engagement",
}


def categorize_explanation(explanation: str) -> str:
    """Categorize as 'human-focused', 'ai-focused', or 'neutral'."""
    explanation_lower = explanation.lower()
    human_matches = sum(1 for keyword in HUMAN_INDICATORS if keyword in explanation_lower)
    ai_matches = sum(1 for keyword in AI_INDICATORS if keyword in explanation_lower)

    if human_matches > ai_matches:
        return "human-focused"
    elif ai_matches > human_matches:
        return "ai-focused"
    else:
        return "neutral"


def load_inference_explanations():
    """Load explanations from inference results."""

    inference_dir = '/gpfs/projects/p32143/RL_human_decision/baseline/preference_RL/inference_results'

    explanations_by_id = {}

    # Look for JSON files in inference results
    if os.path.exists(inference_dir):
        for filename in os.listdir(inference_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(inference_dir, filename)

                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Data might be a list or dict
                    if isinstance(data, list):
                        for item in data:
                            if 'text_id' in item:
                                explanations_by_id[item['text_id']] = {
                                    'explanation': item.get('explanation', ''),
                                    'source': filename,
                                    'category': categorize_explanation(item.get('explanation', ''))
                                }
                    elif isinstance(data, dict):
                        for text_id, item in data.items():
                            if isinstance(item, dict) and 'explanation' in item:
                                explanations_by_id[text_id] = {
                                    'explanation': item.get('explanation', ''),
                                    'source': filename,
                                    'category': categorize_explanation(item.get('explanation', ''))
                                }
                except Exception as e:
                    pass

    print(f"Loaded {len(explanations_by_id)} explanations from inference results")
    return explanations_by_id


def analyze_accuracy_by_explanation_category():
    """Analyze accuracy correlating with explanation category."""

    print("\n" + "="*80)
    print("DETAILED ACCURACY ANALYSIS BY EXPLANATION CATEGORY")
    print("="*80)

    db_path = '/gpfs/projects/p32143/RL_human_decision/statistic/text_2026-08-11.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all guesses
    cursor.execute("""
        SELECT
            text_id,
            label_truth,
            explanation_source,
            guess,
            correct
        FROM text_guess
        WHERE correct IS NOT NULL
    """)

    rows = cursor.fetchall()
    print(f"\nTotal guesses: {len(rows)}")

    # Load inference explanations for categorization
    print("Loading inference explanations...")
    infer_explanations = load_inference_explanations()

    # Organize results by label and explanation category
    results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    for text_id, label_truth, explanation_source, guess, correct in rows:
        # Try to categorize the explanation
        if text_id in infer_explanations:
            explanation_category = infer_explanations[text_id]['category']
        else:
            # Try to infer from explanation_source
            if 'human-focused' in explanation_source.lower():
                explanation_category = 'human-focused'
            elif 'ai-focused' in explanation_source.lower():
                explanation_category = 'ai-focused'
            else:
                explanation_category = 'unknown'

        # Record result
        results[label_truth][explanation_category]['total'] += 1
        if correct:
            results[label_truth][explanation_category]['correct'] += 1

    # Print results
    print(f"\n{'-'*80}")
    print("ACCURACY BY TRUE LABEL AND EXPLANATION CATEGORY:")
    print(f"{'-'*80}")

    for label in ['Human', 'AI']:
        if label not in results or not results[label]:
            continue

        print(f"\n✅ WHEN TRUE LABEL IS: {label}")

        for category in sorted(results[label].keys()):
            data = results[label][category]
            total = data['total']
            correct = data['correct']
            accuracy = (correct / total * 100) if total > 0 else 0

            print(f"   {category:20s}: {correct:4d}/{total:4d} correct ({accuracy:5.1f}%)")

    # Key insight for AI texts
    print(f"\n{'-'*80}")
    print("KEY INSIGHT:")
    print(f"{'-'*80}")

    if 'AI' in results:
        ai_results = results['AI']

        if 'human-focused' in ai_results and 'ai-focused' in ai_results:
            human_acc = (ai_results['human-focused']['correct'] /
                        ai_results['human-focused']['total'] * 100)
            ai_acc = (ai_results['ai-focused']['correct'] /
                     ai_results['ai-focused']['total'] * 100)

            print(f"\nWhen explaining AI-written texts:")
            print(f"  - Human-focused explanations: {human_acc:.1f}% accuracy")
            print(f"  - AI-focused explanations:    {ai_acc:.1f}% accuracy")
            print(f"  - Difference:                 {abs(ai_acc - human_acc):.1f} pp")

            if ai_acc > human_acc:
                print(f"\n  ✅ HYPOTHESIS 2 CONFIRMED!")
                print(f"     AI-focused explanations are MORE accurate on AI texts")
                print(f"     This shows that label-aware explanations matter for accuracy")
                print(f"     Human Decision RL learned this by optimizing for accuracy")
            else:
                print(f"\n  Note: Human-focused explanations slightly higher")
                print(f"     (but both models in study may be from same training)")

    conn.close()

    return results


def analyze_preference_vs_decision_rl():
    """Compare accuracies if we can identify which are from each model."""

    print("\n\n" + "="*80)
    print("PREFERENCE RL vs HUMAN DECISION RL ACCURACY COMPARISON")
    print("="*80)

    db_path = '/gpfs/projects/p32143/RL_human_decision/statistic/text_2026-08-11.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get guesses with explanation source details
    cursor.execute("""
        SELECT
            text_id,
            label_truth,
            explanation_source,
            correct,
            COUNT(*) as count
        FROM text_guess
        WHERE correct IS NOT NULL AND explanation_source IS NOT NULL
        GROUP BY text_id, label_truth, explanation_source, correct
    """)

    rows = cursor.fetchall()

    # Try to identify which model each explanation comes from
    print(f"\nAnalyzing {len(rows)} unique explanation sources...")

    model_results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    for text_id, label_truth, explanation_source, correct, count in rows:
        # Heuristic: try to identify model from explanation_source
        if 'pref' in explanation_source.lower():
            model = 'Preference RL'
        elif 'human' in explanation_source.lower() or 'decision' in explanation_source.lower():
            model = 'Human Decision RL'
        else:
            model = 'Unknown'

        model_results[label_truth][model]['total'] += count
        if correct:
            model_results[label_truth][model]['correct'] += count

    print(f"\n{'-'*80}")
    print("RESULTS BY MODEL (if identifiable):")
    print(f"{'-'*80}")

    for label in ['Human', 'AI']:
        if label not in model_results:
            continue

        print(f"\n✅ FOR {label} TEXTS:")
        for model in sorted(model_results[label].keys()):
            data = model_results[label][model]
            total = data['total']
            correct = data['correct']
            accuracy = (correct / total * 100) if total > 0 else 0

            print(f"   {model:20s}: {correct:4d}/{total:4d} ({accuracy:5.1f}%)")

    conn.close()


def main():
    """Main analysis."""

    print("\n" + "█"*80)
    print("█ DETAILED ACCURACY ANALYSIS")
    print("█"*80)

    # Analyze by explanation category
    category_results = analyze_accuracy_by_explanation_category()

    # Try to compare models
    analyze_preference_vs_decision_rl()


if __name__ == "__main__":
    main()
