"""
Calculate human-preferred explanations based on decision accuracy from database.

For each text and its two explanations:
1. Load accuracy data from human decisions
2. Compare which explanation leads to higher accuracy
3. Handle edge case: if only one explanation has data, use it if accuracy >= 0.5, else skip

Output format:
{
  "text_id_human": {
    "text": "the original text",
    "label": "Human",
    "chosen_explanation": "the best explanation",
    "chosen_source": "source of best explanation",
    "chosen_accuracy": 0.75,
    "chosen_decisions_count": 4,
    "rejected_source": "other source",
    "rejected_accuracy": 0.25,
    "rejected_decisions_count": 4,
    "decision_method": "both_have_data" | "one_has_data" | ...
  },
  ...
}
"""

import json
import csv
import sqlite3
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
CSV_FILE = "/projects/p32143/RL_human_decision/text_data/social/master_stimuli_database_socialmedia.csv"
DATABASE_FILE = "/projects/p32143/RL_human_decision/statistic/text_2026-08-11.db"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
OUTPUT_FILE = "human_preferred_explanations.json"

# ==========================================
# Helper Functions
# ==========================================
def load_explanations_from_csv(csv_file):
    """
    Load explanations from CSV file.

    Returns:
        dict mapping (text_id_label) -> {label, text_content, explanations: [{source, path}, ...]}
    """
    explanations_data = defaultdict(lambda: {'explanations': []})
    text_ids_to_texts = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_id = row['text_id']
                label = row['label']
                explanation_path = row['explanation_path']
                text_path = row['text_path']

                composite_key = f"{text_id}_{label.lower()}"

                # Extract source from explanation_path (directory name)
                source = explanation_path.split('/')[0]

                explanations_data[composite_key]['explanations'].append({
                    'source': source,
                    'path': explanation_path
                })
                explanations_data[composite_key]['label'] = label

                # Load text content if not already loaded
                if composite_key not in text_ids_to_texts:
                    full_text_path = f"{TEXT_DATA_DIR}{text_path}"
                    try:
                        with open(full_text_path, 'r', encoding='utf-8') as tf:
                            text_ids_to_texts[composite_key] = tf.read().strip()
                    except Exception as e:
                        print(f"Error loading text {composite_key} from {full_text_path}: {e}")

    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return {}, {}

    return dict(explanations_data), text_ids_to_texts


def load_accuracy_from_database(db_file):
    """
    Load accuracy data from database.

    Returns:
        dict mapping (text_id, explanation_source) -> {accuracy, correct_count, total_count}
    """
    accuracy_data = {}

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Query: for each text_id and explanation_source, calculate accuracy
        query = """
        SELECT
            text_id,
            explanation_source,
            COUNT(*) as total_count,
            SUM(CAST(correct AS INTEGER)) as correct_count,
            ROUND(100.0 * SUM(CAST(correct AS INTEGER)) / COUNT(*), 2) as accuracy
        FROM text_guess
        GROUP BY text_id, explanation_source
        """

        cursor.execute(query)
        for row in cursor.fetchall():
            text_id, explanation_source, total_count, correct_count, accuracy = row
            key = (text_id, explanation_source)
            accuracy_data[key] = {
                'accuracy': accuracy / 100.0,  # Convert to 0-1 scale
                'accuracy_percent': accuracy,
                'correct_count': correct_count,
                'total_count': total_count
            }

        conn.close()

    except Exception as e:
        print(f"Error loading from database {db_file}: {e}")
        return {}

    return accuracy_data


def get_human_preferred(explanations, accuracy_data, composite_key):
    """
    Determine which explanation is preferred by humans based on accuracy.

    Args:
        explanations: list of 2 explanation dicts with 'source' key
        accuracy_data: dict mapping (text_id, source) -> accuracy info
        composite_key: text_id_label

    Returns:
        dict with chosen explanation or None if no valid choice
    """
    text_id = composite_key.rsplit('_', 1)[0]  # Extract text_id from composite key

    if len(explanations) < 2:
        print(f"⚠️  {composite_key}: Only {len(explanations)} explanation(s) found")
        return None

    exp_0 = explanations[0]
    exp_1 = explanations[1]

    source_0 = exp_0['source']
    source_1 = exp_1['source']

    # Get accuracy data for both explanations
    acc_0 = accuracy_data.get((text_id, source_0))
    acc_1 = accuracy_data.get((text_id, source_1))

    # Decision logic based on available data
    if acc_0 and acc_1:
        # Both have data: pick the one with higher accuracy
        if acc_0['accuracy'] > acc_1['accuracy']:
            chosen_idx = 0
            decision_method = "both_have_data_higher_accuracy"
        elif acc_1['accuracy'] > acc_0['accuracy']:
            chosen_idx = 1
            decision_method = "both_have_data_higher_accuracy"
        else:
            # Tie: pick the one with more decisions (higher confidence)
            if acc_0['total_count'] >= acc_1['total_count']:
                chosen_idx = 0
                decision_method = "both_have_data_tied_more_samples"
            else:
                chosen_idx = 1
                decision_method = "both_have_data_tied_more_samples"

    elif acc_0 and not acc_1:
        # Only first explanation has data
        if acc_0['accuracy'] >= 0.5:
            chosen_idx = 0
            decision_method = "only_first_has_data_above_threshold"
        else:
            return None  # Skip: accuracy below threshold

    elif acc_1 and not acc_0:
        # Only second explanation has data
        if acc_1['accuracy'] >= 0.5:
            chosen_idx = 1
            decision_method = "only_second_has_data_above_threshold"
        else:
            return None  # Skip: accuracy below threshold

    else:
        # Neither has data
        return None

    # Build result
    chosen = explanations[chosen_idx]
    rejected = explanations[1 - chosen_idx]

    chosen_acc = accuracy_data.get((text_id, chosen['source']))
    rejected_acc = accuracy_data.get((text_id, rejected['source']))

    result = {
        'chosen_idx': chosen_idx,
        'chosen_source': chosen['source'],
        'chosen_accuracy': chosen_acc['accuracy'] if chosen_acc else None,
        'chosen_accuracy_percent': chosen_acc['accuracy_percent'] if chosen_acc else None,
        'chosen_decisions_count': chosen_acc['total_count'] if chosen_acc else 0,
        'rejected_source': rejected['source'],
        'rejected_accuracy': rejected_acc['accuracy'] if rejected_acc else None,
        'rejected_accuracy_percent': rejected_acc['accuracy_percent'] if rejected_acc else None,
        'rejected_decisions_count': rejected_acc['total_count'] if rejected_acc else 0,
        'decision_method': decision_method
    }

    return result


def create_human_preferences(explanations_data, accuracy_data, text_ids_to_texts):
    """
    Create human preference pairs based on accuracy data.

    Returns:
        human preferences dataset
    """
    human_preferences = {}

    for composite_key, data in explanations_data.items():
        text_content = text_ids_to_texts.get(composite_key)
        label = data.get('label', '')
        explanations = data.get('explanations', [])

        if not text_content:
            continue

        result = get_human_preferred(explanations, accuracy_data, composite_key)

        if result is None:
            continue

        chosen_idx = result['chosen_idx']
        chosen = explanations[chosen_idx]
        rejected = explanations[1 - chosen_idx]

        human_preferences[composite_key] = {
            'text': text_content,
            'label': label,
            'chosen_explanation_source': chosen['source'],
            'chosen_accuracy': result['chosen_accuracy'],
            'chosen_accuracy_percent': result['chosen_accuracy_percent'],
            'chosen_decisions_count': result['chosen_decisions_count'],
            'rejected_explanation_source': rejected['source'],
            'rejected_accuracy': result['rejected_accuracy'],
            'rejected_accuracy_percent': result['rejected_accuracy_percent'],
            'rejected_decisions_count': result['rejected_decisions_count'],
            'decision_method': result['decision_method']
        }

    return human_preferences


def main():
    print("="*80)
    print("Ranking Explanations by Human Decision Accuracy")
    print("="*80)

    # ==========================================
    # 1. Load explanations from CSV
    # ==========================================
    print("\n[1/3] Loading explanations from CSV...")
    explanations_data, text_ids_to_texts = load_explanations_from_csv(CSV_FILE)

    print(f"  Texts with explanations: {len(explanations_data)}")

    if not explanations_data:
        print("❌ No explanations found. Exiting.")
        return

    # ==========================================
    # 2. Load accuracy data from database
    # ==========================================
    print("\n[2/3] Loading accuracy data from database...")
    accuracy_data = load_accuracy_from_database(DATABASE_FILE)

    print(f"  Explanation accuracy records: {len(accuracy_data)}")

    if not accuracy_data:
        print("❌ No accuracy data found. Exiting.")
        return

    # ==========================================
    # 3. Calculate human preferences
    # ==========================================
    print("\n[3/3] Calculating human preferences...")

    human_preferences = create_human_preferences(explanations_data, accuracy_data, text_ids_to_texts)

    # Save preferences
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(human_preferences, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(human_preferences)} human preference pairs to {OUTPUT_FILE}")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total human preference pairs: {len(human_preferences)}")

    # Count by label
    human_count = sum(1 for item in human_preferences.values() if item['label'] == 'Human')
    ai_count = sum(1 for item in human_preferences.values() if item['label'] == 'AI')
    print(f"  Human texts: {human_count}")
    print(f"  AI texts: {ai_count}")

    # Count by decision method
    print("\nDecision methods:")
    decision_methods = defaultdict(int)
    for item in human_preferences.values():
        decision_methods[item['decision_method']] += 1

    for method in sorted(decision_methods.keys()):
        print(f"  {method}: {decision_methods[method]}")

    # Statistics by explanation source
    print("\nChosen explanation sources (Human texts):")
    human_sources = {}
    for item in human_preferences.values():
        if item['label'] == 'Human':
            source = item['chosen_explanation_source']
            human_sources[source] = human_sources.get(source, 0) + 1
    for source in sorted(human_sources.keys()):
        print(f"  {source}: {human_sources[source]}")

    print("\nChosen explanation sources (AI texts):")
    ai_sources = {}
    for item in human_preferences.values():
        if item['label'] == 'AI':
            source = item['chosen_explanation_source']
            ai_sources[source] = ai_sources.get(source, 0) + 1
    for source in sorted(ai_sources.keys()):
        print(f"  {source}: {ai_sources[source]}")

    # Accuracy statistics
    print("\nAccuracy Statistics:")
    chosen_accuracies = [item['chosen_accuracy'] for item in human_preferences.values() if item['chosen_accuracy'] is not None]
    if chosen_accuracies:
        print(f"  Chosen explanation - Mean accuracy: {sum(chosen_accuracies) / len(chosen_accuracies):.2%}")
        print(f"  Chosen explanation - Min accuracy: {min(chosen_accuracies):.2%}")
        print(f"  Chosen explanation - Max accuracy: {max(chosen_accuracies):.2%}")

    rejected_accuracies = [item['rejected_accuracy'] for item in human_preferences.values() if item['rejected_accuracy'] is not None]
    if rejected_accuracies:
        print(f"  Rejected explanation - Mean accuracy: {sum(rejected_accuracies) / len(rejected_accuracies):.2%}")
        print(f"  Rejected explanation - Min accuracy: {min(rejected_accuracies):.2%}")
        print(f"  Rejected explanation - Max accuracy: {max(rejected_accuracies):.2%}")


if __name__ == "__main__":
    main()
