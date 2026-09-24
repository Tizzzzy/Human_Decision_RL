import pandas as pd
import json
import os

df = pd.read_csv("master_stimuli_database_books.csv")
human_sample = df[df['label'] == 'Human'].sample(n=5, random_state=42)
ai_sample = df[df['label'] == 'AI'].sample(n=5, random_state=42)
sampled_df = pd.concat([human_sample, ai_sample])

result = {}
for _, row in sampled_df.iterrows():
    text_path = row['text_path']
    explanation_path = row['explanation_path']
    label = row['label']

    # Try to read text
    text_content = ""
    if pd.notna(text_path) and os.path.exists(str(text_path)):
        with open(str(text_path), 'r', encoding='utf-8') as f:
            text_content = f.read()
    else:
        text_content = f"<File {text_path} not found in environment>"

    # Try to read explanation
    explanation_content = ""
    if pd.notna(explanation_path) and os.path.exists(str(explanation_path)):
        with open(str(explanation_path), 'r', encoding='utf-8') as f:
            explanation_content = f.read()
    else:
        explanation_content = f"<File {explanation_path} not found in environment>"

    result[str(text_path)] = {
        "text": text_content,
        "explanation": explanation_content,
        "label": label
    }

with open("example_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print("Here is the resulting JSON structure:")
print(json.dumps(result, indent=4))