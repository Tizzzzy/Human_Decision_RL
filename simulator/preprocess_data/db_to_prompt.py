import sqlite3
import json
from collections import defaultdict

# Connect to your SQLite database
db_path = '/projects/p32143/RL_human_decision/statistic/text_2026-08-21.db'  # Replace with your actual database file path
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query the necessary columns
# We group by text_id, label_truth, and explanation_source to handle duplicate text_ids
cursor.execute("""
    SELECT text_id, label_truth, explanation_source, guess 
    FROM text_guess
    WHERE text_id IS NOT NULL
""")
rows = cursor.fetchall()

# Group guesses by unique prompt configuration
grouped_data = defaultdict(list)
for text_id, label_truth, explanation_source, guess in rows:
    key = (text_id, label_truth, explanation_source)
    grouped_data[key].append(guess)

prompts_dataset = []

# Process each unique prompt group
for (text_id, label_truth, explanation_source), guesses in grouped_data.items():
    
    # 1. Determine the text file path based on label_truth
    if label_truth.lower() == 'ai':
        text_path = f"/projects/p32143/RL_human_decision/text_data/social/SocialMedia_rewrite/{text_id}.txt"
    else:
        text_path = f"/projects/p32143/RL_human_decision/text_data/social/SocialMedia_Reddit/{text_id}.txt"
        
    # 2. Determine the explanation file path
    explanation_path = f"/projects/p32143/RL_human_decision/text_data/social/{explanation_source}/{text_id}.txt"
    
    # 3. Read the file contents (with fallback if the file is missing)
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
    except FileNotFoundError:
        text_content = "[TEXT FILE NOT FOUND]"
        
    try:
        with open(explanation_path, 'r', encoding='utf-8') as f:
            explanation_content = f.read().strip()
    except FileNotFoundError:
        explanation_content = "[EXPLANATION FILE NOT FOUND]"
        
    # 4. Construct the final prompt
    prompt = f"""You are tasked with detecting whether a given text is AI-generated or human-written.
Below is the text, followed by an explanation hint that may help in your prediction.

Text:
{text_content}

Explanation Hint:
{explanation_content}

Based on the text and the explanation hint, predict whether the text is AI-generated or human-written. 
Output 1 for AI-generated or 0 for human-written. Output only the number 0 or 1."""

    # 5. Process and aggregate the guesses (AI = 1, Human = 0)
    raw_guesses_num = [1 if str(g).lower() == 'ai' else 0 for g in guesses if g]
    prob_ai = sum(raw_guesses_num) / len(raw_guesses_num) if raw_guesses_num else 0.0
    
    # 6. Store the compiled data
    prompts_dataset.append({
        "text_id": text_id,
        "label_truth": label_truth,
        "label_truth_numeric": 1 if label_truth.lower() == 'ai' else 0,
        "explanation_source": explanation_source,
        "prompt": prompt,
        "raw_guesses_text": guesses,
        "raw_guesses_numeric": raw_guesses_num,
        "probability_label_1": prob_ai,
        "total_guesses": len(raw_guesses_num)
    })

# Export the final compiled dataset to JSON
output_file = '/projects/p32143/RL_human_decision/simulator/data/prompts_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(prompts_dataset, f, indent=4)

print(f"Successfully processed {len(prompts_dataset)} unique prompts and saved to {output_file}.")