import os
import csv
import json
from pathlib import Path

# Import the BERTScorer class
from bert_score import BERTScorer

def get_similarity(text1, text2, scorer):
    """
    Returns a BERTScore F1 similarity score between the two texts.
    Lower score = more semantically different.
    """
    P, R, F1 = scorer.score([text1], [text2])
    return F1.item()

def load_jsonl_to_dict(file_path):
    """Loads a JSONL file into a dictionary keyed by the speech text."""
    data_map = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found -> {file_path}")
        return data_map

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                speech_text = obj.get("text")
                if speech_text:
                    data_map[speech_text] = {
                        "explanation": obj.get("explanation", "").strip(),
                        "label": obj.get("label", "")
                    }
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON on line {idx + 1} in {file_path}")
    return data_map

def find_most_different_pairs_jsonl(base_jsonl, claude_jsonl, gemini_jsonl, output_csv):
    print("Loading JSONL files into memory...")
    base_data = load_jsonl_to_dict(base_jsonl)
    claude_data = load_jsonl_to_dict(claude_jsonl)
    gemini_data = load_jsonl_to_dict(gemini_jsonl)

    print("Loading BERT model (this may take a moment)...")
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Writing comprehensive headers
        csv_writer.writerow([
            'Speech_Text', 
            'Label', 
            'Most_Different_Source_1', 
            'Most_Different_Source_2', 
            'Explanation_1', 
            'Explanation_2', 
            'BERT_F1_Score'
        ])

        processed_count = 0

        # Match items using the keys from the base dataset
        for speech_text, base_item in base_data.items():
            # Check if this exact speech text exists in all three datasets
            if speech_text in claude_data and speech_text in gemini_data:
                label = base_item["label"]
                base_exp = base_item["explanation"]
                claude_exp = claude_data[speech_text]["explanation"]
                gemini_exp = gemini_data[speech_text]["explanation"]

                # Group explanations with their source labels
                explanation_data = [
                    (base_exp, "Base"),
                    (claude_exp, "Claude"),
                    (gemini_exp, "Gemini")
                ]

                # Filter out empty or extremely short explanations (less than 10 characters)
                valid_explanations = [data for data in explanation_data if len(data[0]) > 10]
                empty_count = 3 - len(valid_explanations)

                if empty_count >= 2:
                    # Cannot perform pairwise comparison with 0 or 1 valid text
                    continue
                    
                elif empty_count == 1:
                    # Exactly one file missing or empty; compare the remaining two directly
                    exp1, src1 = valid_explanations[0]
                    exp2, src2 = valid_explanations[1]
                    
                    sim_score = get_similarity(exp1, exp2, scorer)
                    most_different_pair = (src1, src2, exp1, exp2, sim_score)
                    
                else:
                    # No empty explanations. Compute scores for all 3 pairs.
                    sim_base_claude = get_similarity(base_exp, claude_exp, scorer)
                    sim_base_gemini = get_similarity(base_exp, gemini_exp, scorer)
                    sim_claude_gemini = get_similarity(claude_exp, gemini_exp, scorer)

                    pairs = [
                        ("Base", "Claude", base_exp, claude_exp, sim_base_claude),
                        ("Base", "Gemini", base_exp, gemini_exp, sim_base_gemini),
                        ("Claude", "Gemini", claude_exp, gemini_exp, sim_claude_gemini)
                    ]

                    # Select the pair with the MINIMUM similarity score (the most different)
                    most_different_pair = min(pairs, key=lambda x: x[4])

                # Write record details into rows
                csv_writer.writerow([
                    speech_text,
                    label,
                    most_different_pair[0],  # Source 1 Name
                    most_different_pair[1],  # Source 2 Name
                    most_different_pair[2],  # Explanation 1 Text
                    most_different_pair[3],  # Explanation 2 Text
                    round(most_different_pair[4], 4)  # Rounded BERTScore
                ])
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} matching records...")

    print(f"Done! Evaluated {processed_count} rows. Results saved to {output_csv}")

# --- Configuration ---
category = "speeches_sampled"
BASE_JSONL = "speeches_sampled_explanation_gpt.jsonl"
CLAUDE_JSONL = "speeches_sampled_explanation_claude.jsonl"
GEMINI_JSONL = "speeches_sampled_explanation_gemini.jsonl"
OUTPUT_FILE = f'most_different_explanations_{category}.csv'

if __name__ == '__main__':
    find_most_different_pairs_jsonl(BASE_JSONL, CLAUDE_JSONL, GEMINI_JSONL, OUTPUT_FILE)