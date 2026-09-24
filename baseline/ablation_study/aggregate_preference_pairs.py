"""
Aggregate preference pairs from three different models (Qwen, Gemma, Llama)
using majority voting to determine the best explanation.
"""

import json
from collections import Counter

# File paths
QWEN_FILE = "preference_pairs.json"
GEMMA_FILE = "preference_pairs_gemma.json"
LLAMA_FILE = "preference_pairs_llama.json"
OUTPUT_FILE = "preference_pairs_aggregated.json"

def load_json(filepath):
    """Load JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def aggregate_preferences():
    """
    Aggregate preference pairs from three models using majority voting.

    For each text (composite key), vote on which explanation source wins.
    The explanation source chosen by the most models becomes the final choice.
    """
    print("="*80)
    print("Aggregating Preference Pairs from Three Models")
    print("="*80)

    # Load all preference pairs
    print("\n[1/3] Loading preference pairs from three models...")
    qwen_prefs = load_json(QWEN_FILE)
    gemma_prefs = load_json(GEMMA_FILE)
    llama_prefs = load_json(LLAMA_FILE)

    print(f"  Qwen: {len(qwen_prefs)} entries")
    print(f"  Gemma: {len(gemma_prefs)} entries")
    print(f"  Llama: {len(llama_prefs)} entries")

    # Collect all unique composite keys
    all_keys = set(qwen_prefs.keys()) | set(gemma_prefs.keys()) | set(llama_prefs.keys())
    print(f"  Total unique keys: {len(all_keys)}")

    # ==========================================
    # [2/3] Vote on best explanation
    # ==========================================
    print("\n[2/3] Voting on best explanation for each text...")

    aggregated = {}
    no_consensus_count = 0
    split_votes_count = 0

    for composite_key in all_keys:
        qwen_entry = qwen_prefs.get(composite_key)
        gemma_entry = gemma_prefs.get(composite_key)
        llama_entry = llama_prefs.get(composite_key)

        # Skip if entry doesn't exist in any model
        if not qwen_entry and not gemma_entry and not llama_entry:
            continue

        # Get text content and label from any available entry
        text_content = None
        label = None
        if qwen_entry:
            text_content = qwen_entry.get('text')
            label = qwen_entry.get('label')
        elif gemma_entry:
            text_content = gemma_entry.get('text')
            label = gemma_entry.get('label')
        elif llama_entry:
            text_content = llama_entry.get('text')
            label = llama_entry.get('label')

        # Collect votes for each explanation source
        votes = []
        model_votes = {}

        if qwen_entry:
            chosen_source = qwen_entry.get('chosen_source')
            votes.append(chosen_source)
            model_votes['qwen'] = chosen_source

        if gemma_entry:
            chosen_source = gemma_entry.get('chosen_source')
            votes.append(chosen_source)
            model_votes['gemma'] = chosen_source

        if llama_entry:
            chosen_source = llama_entry.get('chosen_source')
            votes.append(chosen_source)
            model_votes['llama'] = chosen_source

        # Count votes
        vote_counts = Counter(votes)
        winning_source = vote_counts.most_common(1)[0][0]
        winning_count = vote_counts.most_common(1)[0][1]

        # Track voting statistics
        if winning_count < len(votes):
            split_votes_count += 1
        if winning_count < 2:
            no_consensus_count += 1

        # Build aggregated entry
        # Find the explanation text for the winning source
        winning_explanation_text = None
        if qwen_entry and qwen_entry.get('chosen_source') == winning_source:
            winning_explanation_text = qwen_entry.get('chosen_explanation')
        elif gemma_entry and gemma_entry.get('chosen_source') == winning_source:
            winning_explanation_text = gemma_entry.get('chosen_explanation')
        elif llama_entry and llama_entry.get('chosen_source') == winning_source:
            winning_explanation_text = llama_entry.get('chosen_explanation')

        # Get rejected explanations from first available entry
        rejected_explanations = None
        if qwen_entry:
            rejected_explanations = qwen_entry.get('rejected_explanations')
        elif gemma_entry:
            rejected_explanations = gemma_entry.get('rejected_explanations')
        elif llama_entry:
            rejected_explanations = llama_entry.get('rejected_explanations')

        aggregated[composite_key] = {
            'text': text_content,
            'label': label,
            'chosen_explanation': winning_explanation_text,
            'chosen_source': winning_source,
            'winning_votes': winning_count,
            'total_votes': len(votes),
            'model_votes': model_votes,
            'rejected_explanations': rejected_explanations
        }

    # ==========================================
    # [3/3] Save aggregated preferences
    # ==========================================
    print("\n[3/3] Saving aggregated preferences...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(aggregated)} aggregated preference pairs to {OUTPUT_FILE}")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total aggregated entries: {len(aggregated)}")

    # Count by label
    human_count = sum(1 for item in aggregated.values() if item['label'] == 'Human')
    ai_count = sum(1 for item in aggregated.values() if item['label'] == 'AI')
    print(f"  Human texts: {human_count}")
    print(f"  AI texts: {ai_count}")

    # Voting statistics
    unanimous_count = sum(1 for item in aggregated.values() if item['winning_votes'] == 3)
    majority_count = sum(1 for item in aggregated.values() if item['winning_votes'] == 2)
    print(f"\nVoting Statistics:")
    print(f"  Unanimous (3/3 models agree): {unanimous_count}")
    print(f"  Majority (2/3 models agree): {majority_count}")
    print(f"  No consensus (split votes): {no_consensus_count}")

    # Count chosen explanation sources by label
    print("\nChosen explanation sources (Human texts):")
    human_sources = {}
    for item in aggregated.values():
        if item['label'] == 'Human':
            source = item['chosen_source']
            human_sources[source] = human_sources.get(source, 0) + 1
    for source in sorted(human_sources.keys()):
        print(f"  {source}: {human_sources[source]}")

    print("\nChosen explanation sources (AI texts):")
    ai_sources = {}
    for item in aggregated.values():
        if item['label'] == 'AI':
            source = item['chosen_source']
            ai_sources[source] = ai_sources.get(source, 0) + 1
    for source in sorted(ai_sources.keys()):
        print(f"  {source}: {ai_sources[source]}")


if __name__ == "__main__":
    aggregate_preferences()
