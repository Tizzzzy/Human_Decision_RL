import os
from google import genai
import time
import json
import random

# 1. Define your file paths
INPUT_FILE = "substantive_speeches_v2.jsonl"
OUTPUT_FILE = "speeches_sampled_explanation_gemini.jsonl"
RANDOM_SEED = 42  # Change this number if you want a different random selection
TARGET_PER_PARTY = 1000 # Number of speeches to sample per party

print("Model loaded successfully!\n" + "="*50)

def explanation(text):
    """Passes the text to the model and returns the explanation, with a 3-try retry mechanism."""
    prompt = f"""Task: Analyze the provided political speech text for linguistic markers, policy focus, or rhetorical devices that indicate whether the speaker belongs to the Democratic or Republican party.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list. 
2. Do NOT provide an introduction, a final verdict, or a summary.
3. Do NOT group the bullet points into sub-headings (keep it a flat list).
4. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category, policy focus, or type of phrasing]."
    - "Look for [linguistic category, policy focus, or type of phrasing] like '[short example]'."
5. When providing examples, you may list multiple words separated by commas (e.g., like "word 1", "word 2").
6. Do NOT use bolding, italics, or any special Markdown formatting.
7. Limit the output to exactly 3 bullet points.

Example Speech Text:
I am frank to say from a humane standpoint that I am interested In them. and yet. at the same time. we have a great many people who have their money invested. to the extent of millions of dollars. in brick manufacturing plants. and even though we say that this brick will be used only in connection with Government buildings. nevertheless these private concerns have been selling to the Government

Explanation:
- Look for policy focus on protecting private enterprise and capital investments.
- Look for phrasing balancing social welfare with business interests like 'humane standpoint'.
- Look for arguments against government competition with private industry.
- Look for explicit mentions of financial stakes like 'money invested', 'millions of dollars'.

Speech Text:
{text}

Explanation:
"""
    
    client = genai.Client(api_key="AIzaSyCpLt9fECSRhiH3c3k-XxarU5HL94yuja8")

    # Retry mechanism: try up to 3 times
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print(f"    -> API Error (Attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(60)  # Short delay before retrying
                
    # If all 3 attempts fail, return None so it writes 'null' to the JSON
    print("    -> Failed 3 times. Returning null.")
    return None

def sample_and_process(input_file, output_file, seed):
    """Samples speeches, handles resuming from previous runs, and saves outputs."""
    dem_pool = []
    rep_pool = []

    # 2. Parse the entire file into memory to separate by party
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                label = data.get("label")
                
                if label == "D":
                    dem_pool.append(data)
                elif label == "R":
                    rep_pool.append(data)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON on line {line_idx + 1}")

    print(f"Loaded {len(dem_pool)} Democratic and {len(rep_pool)} Republican speeches.")

    # 3. Apply the random seed and sample
    random.seed(seed)
    
    # Safely sample up to TARGET_PER_PARTY, or all available if fewer exist
    sampled_dems = random.sample(dem_pool, min(len(dem_pool), TARGET_PER_PARTY))
    sampled_reps = random.sample(rep_pool, min(len(rep_pool), TARGET_PER_PARTY))
        
    # Combine and shuffle them so they aren't grouped sequentially in the output
    final_samples = sampled_dems + sampled_reps
    random.shuffle(final_samples)
    total_samples = len(final_samples)

    # 4. Resume Mechanism
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            # Count only non-empty lines to be safe
            processed_count = sum(1 for line in f if line.strip())
        print(f"Found existing output. Resuming from item {processed_count + 1}...")

    # Slice the list to skip the ones we've already done
    remaining_samples = final_samples[processed_count:]

    if not remaining_samples:
        print("All items have already been processed!")
        return

    print(f"Successfully selected {total_samples} balanced speeches using seed {seed}.")
    print(f"Beginning inference on the remaining {len(remaining_samples)} items...")

    # 5. Process the selected subset using 'a' (append) mode
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for idx, data in enumerate(remaining_samples):
            text_content = data.get("text", "")
            party = data.get("label", "Unknown")
            
            # Calculate the actual index out of the total target for logging
            actual_idx = processed_count + idx + 1
            print(f"Processing item {actual_idx}/{total_samples} (Party: {party})...")
            
            # Generate explanation via API (includes built-in retries)
            explanation_content = explanation(text_content)
            data["explanation"] = explanation_content
            
            # Save item to output file immediately
            out_f.write(json.dumps(data) + "\n")

    print(f"\n--- Processing complete! Outputs saved to {output_file} ---")

if __name__ == "__main__":
    sample_and_process(INPUT_FILE, OUTPUT_FILE, RANDOM_SEED)
