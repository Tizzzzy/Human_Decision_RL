import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import re

# --- 1. Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
INPUT_FILE = "substantive_speeches.jsonl"
OUTPUT_FILE = "substantive_speeches_v2.jsonl"
PROGRESS_FILE = "progress.txt" # NEW: File to track our line number

# --- 2. Load Tokenizer and Model ---
print("Loading tokenizer and model into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="/projects/p32143/cache/huggingface"  # Optional: specify a cache directory
)
print("Model loaded successfully!\n" + "="*50)


def filter_text(text):
    # Added the 'f' prefix to properly inject the 'text' variable into the string
    prompt = f"""You are an expert political analyst. Your task is to analyze a provided excerpt of a speech and determine whether there is enough ideological, historical, rhetorical, or policy-based evidence to identify the speaker's political party (e.g., Democrat, Republican, etc.).

Analyze this excerpt:
{text}

Output your response strictly in the following format:

Rationale = [[ 1-2 sentences rationale ]]
Can_Identify = [[ true or false ]]
"""
    
    
# f"""You are a political analyst helping clean a dataset of U.S. Congressional speeches. Your task is to determine if a speech excerpt contains substantive political content.

# Definitions:
# True (Signal): The text contains actual political debate, policy positions, ideological rhetoric, or discussions about real-world issues. A reader could reasonably attempt to guess the speaker's political leaning based on the content.
# False (Noise): The text focuses on the administrative mechanics of Congress. It contains no political opinions or policy arguments.

# Examples:
# "I thank the Senator for his correction. It was a slip of the tongue if I said that. One thing more. and this. I know. will appeal to the sense of fairness of every Senator here. Last summer. after much litigation over the continuance or noneontinuance of the special committee of investigation. the Federal courts in Pennsylvania held that the committee had ceased to exist. and that only the Senate could revive it."
# Output: False

# "The proposed tax cuts heavily favor massive corporations over the working families in my district. We cannot afford to gut our social safety nets."
# Output: True

# Speech Excerpt:
# {text}

# Output Format:
# Output exactly one word: True or False. Do not include any other text, punctuation, or explanation."""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
    
    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=248, # Slight buffer to ensure the full word is generated
        temperature=0.1,
        do_sample=True,    # Required when using temperature > 0 in most HF configs
        pad_token_id=tokenizer.eos_token_id
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    rewritten_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    return rewritten_text


def process_dataset():
    print(f"Reading from {INPUT_FILE}...")
    
    party_counts = Counter()
    total_processed = 0
    total_substantive = 0
    
    # --- NEW: Restore stats from the existing output file if it exists ---
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing {OUTPUT_FILE}. Restoring previous statistics...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    party_counts[record.get("label", "")] += 1
                    total_substantive += 1
                except json.JSONDecodeError:
                    continue

    # --- NEW: Restore the line progress from the tracking file ---
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                total_processed = int(f.read().strip())
                print(f"Resuming from line {total_processed}...")
            except ValueError:
                total_processed = 0

    print(f"Reading from {INPUT_FILE}...")
    
    # --- CHANGED: Open outfile in 'a' (append) mode instead of 'w' ---
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            # --- NEW: Skip lines we've already processed ---
            if i < total_processed:
                continue
                
            try:
                record = json.loads(line)
                text = record.get("text", "")
                label = record.get("label", "")
                
                if text:
                    classification = filter_text(text)

                    classification_lower = classification.lower()
                    print(classification_lower)

                    if "can_identify" in classification_lower:
                        result_part = classification_lower.split("can_identify")[-1]
                        if "true" in result_part:
                            outfile.write(json.dumps(record) + '\n')
                            outfile.flush() 
                            
                            total_substantive += 1
                            party_counts[label] += 1
                        
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"Error processing record {total_processed + 1}: {e}")
                
            # Increment the processed counter
            total_processed += 1
            
            # --- NEW: Save progress to disk every 10 records ---
            if total_processed % 10 == 0:
                with open(PROGRESS_FILE, 'w') as pf:
                    pf.write(str(total_processed))
            
            if total_processed % 100 == 0:
                print(f"Processed {total_processed} records. Kept {total_substantive} Substantive so far...")
                
        # Save final progress when loop finishes naturally
        with open(PROGRESS_FILE, 'w') as pf:
            pf.write(str(total_processed))
            
    # --- Final Output ---
    print("\n" + "="*45)
    print("LLM FILTERING COMPLETE")
    print("="*45)
    print(f"Total Records Evaluated: {total_processed:,}")
    print(f"Total Substantive Kept:  {total_substantive:,}")
    
    if total_substantive > 0:
        print("\n--- Final Substantive Label Distribution ---")
        for party, count in party_counts.most_common():
            pct = (count / total_substantive) * 100
            party_name = 'Republican (R)' if party == 'R' else 'Democrat (D)' if party == 'D' else party
            print(f"{party_name}: {count:,} ({pct:.2f}%)")
    print("="*45)

if __name__ == "__main__":
    process_dataset()