# import os
# import csv
# from pathlib import Path

# # Import the BERTScorer class
# from bert_score import BERTScorer

# def get_similarity(text1, text2, scorer):
#     """
#     Returns a BERTScore F1 similarity score between the two texts.
#     Lower score = more semantically different.
#     """
#     # scorer.score takes lists of candidate and reference texts.
#     P, R, F1 = scorer.score([text1], [text2])
#     return F1.item()

# def compare_base_and_claude(base_dir, claude_dir, output_csv):
#     base_path = Path(base_dir)
#     claude_path = Path(claude_dir)

#     print(f"Scanning directories starting from: {base_path}")
#     print("Loading BERT model (this may take a moment)...")
    
#     # Initialize the scorer ONCE.
#     scorer = BERTScorer(lang="en", rescale_with_baseline=True)

#     # Open the CSV file for writing
#     with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(['File_1', 'File_2', 'BERT_F1_Score'])

#         # Walk through the base directory
#         for root, dirs, files in os.walk(base_path):
#             for file in files:
#                 if not file.endswith('.txt'):
#                     continue

#                 # Get the relative path
#                 current_file_path = Path(root) / file
#                 rel_path = current_file_path.relative_to(base_path)

#                 # Construct expected path for Claude
#                 claude_file_path = claude_path / rel_path

#                 # Check if the file exists in both directories
#                 if claude_file_path.exists():
#                     try:
#                         with open(current_file_path, 'r', encoding='utf-8') as f:
#                             base_text = f.read().strip()
#                         with open(claude_file_path, 'r', encoding='utf-8') as f:
#                             claude_text = f.read().strip()
                        
#                         # Only process if both files have actual content
#                         if len(base_text) > 10 and len(claude_text) > 10:
#                             sim_score = get_similarity(base_text, claude_text, scorer)
                            
#                             csv_writer.writerow([
#                                 str(current_file_path), 
#                                 str(claude_file_path), 
#                                 round(sim_score, 4)
#                             ])
#                             print(f"Processed: {rel_path}")
#                         else:
#                             print(f"Skipped (empty file): {rel_path}")

#                     except Exception as e:
#                         print(f"Error reading files for {rel_path}: {e}")
#                 else:
#                     print(f"Claude file missing for: {rel_path}")

#     print(f"Done! Results saved to {output_csv}")

# # --- Configuration ---
# category = "SocialMedia_rewrite" 
# BASE_DIRECTORY = f'{category}_explanation'
# CLAUDE_DIRECTORY = f'{category}_explanation_claude'
# OUTPUT_FILE = f'base_vs_claude_{category.lower()}.csv'

# if __name__ == '__main__':
#     compare_base_and_claude(BASE_DIRECTORY, CLAUDE_DIRECTORY, OUTPUT_FILE)

import os
import csv
from pathlib import Path

# Import the BERTScorer class
from bert_score import BERTScorer

def get_similarity(text1, text2, scorer):
    """
    Returns a BERTScore F1 similarity score between the two texts.
    Lower score = more semantically different.
    """
    # scorer.score takes lists of candidate and reference texts.
    # We pass single-item lists since we are comparing one pair at a time.
    P, R, F1 = scorer.score([text1], [text2])
    
    # Extract the float value from the PyTorch tensor
    return F1.item()

def find_most_different_pairs(base_dir, claude_dir, gemini_dir, output_csv):
    base_path = Path(base_dir)
    claude_path = Path(claude_dir)
    gemini_path = Path(gemini_dir)

    print(f"Scanning directories starting from: {base_path}")
    print("Loading BERT model (this may take a moment)...")
    
    # Initialize the scorer ONCE before the loop for performance.
    # rescale_with_baseline=True stretches the scores out to make differences more obvious.
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Writing headers: The two files that are most different, and their similarity score
        csv_writer.writerow(['File_1', 'File_2', 'BERT_F1_Score'])

        # Walk through the base directory
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if not file.endswith('.txt'):
                    continue

                # Get the relative path (e.g., Dec_2021/abs_5501.txt)
                current_file_path = Path(root) / file
                rel_path = current_file_path.relative_to(base_path)

                # Construct expected paths for Claude and Gemini
                claude_file_path = claude_path / rel_path
                gemini_file_path = gemini_path / rel_path

                # Check if the file exists in ALL THREE directories
                if claude_file_path.exists() and gemini_file_path.exists():
                    try:
                        # Read the contents of all three files
                        with open(current_file_path, 'r', encoding='utf-8') as f:
                            base_text = f.read().strip()
                        with open(claude_file_path, 'r', encoding='utf-8') as f:
                            claude_text = f.read().strip()
                        with open(gemini_file_path, 'r', encoding='utf-8') as f:
                            gemini_text = f.read().strip()
                    except Exception as e:
                        print(f"Error reading files for {rel_path}: {e}")
                        continue

                    # Group texts with their respective paths
                    file_data = [
                        (base_text, str(current_file_path)),
                        (claude_text, str(claude_file_path)),
                        (gemini_text, str(gemini_file_path))
                    ]

                    # Filter out empty files
                    non_empty_files = [data for data in file_data if len(data[0]) > 10]
                    empty_count = 3 - len(non_empty_files)

                    # Logic based on the number of empty files
                    if empty_count >= 2:
                        # Cannot compare if 2 or 3 files are empty (1 or 0 files remaining)
                        print(f"Skipped: {rel_path} ({empty_count} empty files)")
                        continue
                        
                    elif empty_count == 1:
                        # Exactly one file is empty. Compare the remaining two.
                        text1, path1 = non_empty_files[0]
                        text2, path2 = non_empty_files[1]
                        
                        sim_score = get_similarity(text1, text2, scorer)
                        most_different_pair = (path1, path2, sim_score)
                        
                    else:
                        # No empty files. Compare all three combinations.
                        sim_base_claude = get_similarity(base_text, claude_text, scorer)
                        sim_base_gemini = get_similarity(base_text, gemini_text, scorer)
                        sim_claude_gemini = get_similarity(claude_text, gemini_text, scorer)

                        pairs = [
                            (str(current_file_path), str(claude_file_path), sim_base_claude),
                            (str(current_file_path), str(gemini_file_path), sim_base_gemini),
                            (str(claude_file_path), str(gemini_file_path), sim_claude_gemini)
                        ]

                        # Find the pair with the MINIMUM similarity score (the most different)
                        most_different_pair = min(pairs, key=lambda x: x[2])

                    # Write the winning (most different) pair to the CSV
                    csv_writer.writerow([
                        most_different_pair[0], 
                        most_different_pair[1], 
                        round(most_different_pair[2], 4) # Rounding score for readability
                    ])
                    print(f"Processed: {rel_path}")

    print(f"Done! Results saved to {output_csv}")

# --- Configuration ---
# Update these paths if your script is not in the same folder as these directories
category = "SocialMedia_rewrite" # Change to "News" or "Books" as needed
BASE_DIRECTORY = f'{category}_explanation'
CLAUDE_DIRECTORY = f'{category}_explanation_claude'
GEMINI_DIRECTORY = f'{category}_explanation_gemini'
OUTPUT_FILE = f'most_different_files_{category.lower()}.csv'

# Run the function
if __name__ == '__main__':
    find_most_different_pairs(BASE_DIRECTORY, CLAUDE_DIRECTORY, GEMINI_DIRECTORY, OUTPUT_FILE)