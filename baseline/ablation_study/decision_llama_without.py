

import json
import torch
import csv
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
from collections import defaultdict

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
CSV_FILE = "/projects/p32143/RL_human_decision/text_data/social/master_stimuli_database_socialmedia.csv"
OUTPUT_PREFERENCE = "decision_llama_without.json"

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.5,
    "top_p": 0.95,
    "do_sample": True,
}

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of AI vs. Human text detection heuristics.

I will show you a social media text and you need to determine whether the text was written by a human or an AI.

Text:
{text}

First, provide a brief 1-2 sentence analytical justification noting specific linguistic markers (e.g., formatting, tone, phrasing). 
Then, on a new line, provide your final verdict wrapped in label tags.

Format your response EXACTLY like this:
[1-2 sentence justification]
<label>AI</label> OR <label>HUMAN</label>"""

# ==========================================
# Helper Functions
# ==========================================
def load_text_from_csv(csv_file, text_data_dir):
    """
    Load explanations from CSV file and corresponding text files.
    """
    text_ids_to_texts = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_id = row['text_id']
                label = row['label']
                text_path = row['text_path']

                # Create composite key to avoid overwrites (text_id_human or text_id_ai)
                composite_key = f"{text_id}_{label.lower()}"

                # Load text content
                full_text_path = f"{text_data_dir}{text_path}"
                try:
                    with open(full_text_path, 'r', encoding='utf-8') as tf:
                        text_ids_to_texts[composite_key] = tf.read().strip()
                except Exception as e:
                    print(f"Error loading text {composite_key} from {full_text_path}: {e}")

    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return {}

    return text_ids_to_texts


def judge_text(tokenizer, model, text_id, text_content):

    # Format the judge prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        text=text_content
    )

    try:
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Apply chat template and get inputs ready for the model
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG
            )

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"📝  {text_id}: Judge response: {response}")

        # Parse response to extract the predicted label using regex
        # This safely grabs whatever is inside the <label> tags
        match = re.search(r'<label>(.*?)</label>', response, re.IGNORECASE)
        prediction = match.group(1).strip().upper() if match else "UNKNOWN"

        return {
            'prediction': prediction,
            'judge_response': response
        }

    except Exception as e:
        print(f"⚠️  {text_id}: Error during judgment: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("Evaluating Texts with Gemma Judge")
    print("="*80)

    # ==========================================
    # 1. Load data from CSV
    # ==========================================
    print("\n[1/4] Loading text data...")
    text_ids_to_texts = load_text_from_csv(CSV_FILE, TEXT_DATA_DIR)

    if not text_ids_to_texts:
        print("❌ No texts found. Exiting.")
        return
        
    print(f"✓ Loaded {len(text_ids_to_texts)} texts to evaluate.")

    # ==========================================
    # 2. Load Llama-3.1-8B-Instruct judge
    # ==========================================
    print("\n[2/4] Loading Llama-3.1-8B-Instruct judge...")
    try:
        print("🚀 Initializing Llama model (this may take a moment)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Judge loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load judge: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 3. Judge texts
    # ==========================================
    print("\n[3/4] Judging texts...")
    judge_results = {}
    
    # Process items with a progress bar
    for composite_key, text_content in tqdm(text_ids_to_texts.items(), desc="Evaluating"):
        result = judge_text(tokenizer, model, composite_key, text_content)
        
        if result:
            # Extract true label from the end of the composite key
            true_label = composite_key.split('_')[-1].upper()
            
            judge_results[composite_key] = {
                'true_label': true_label,
                'prediction': result['prediction'],
                'raw_response': result['judge_response'],
                'text_content': text_content
            }

    # ==========================================
    # 4. Save and Summarize Results
    # ==========================================
    print(f"\n[4/4] Saving results to {OUTPUT_PREFERENCE}...")
    try:
        with open(OUTPUT_PREFERENCE, 'w', encoding='utf-8') as f:
            json.dump(judge_results, f, indent=4)
        print("✓ Results saved successfully.")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        
    # Calculate and display basic accuracy
    correct = sum(1 for res in judge_results.values() if res['true_label'] == res['prediction'])
    total = len(judge_results)
    
    if total > 0:
        print("\n" + "="*40)
        print("🎯 FINAL RESULTS SUMMARY")
        print("="*40)
        print(f"Total Evaluated: {total}")
        print(f"Correct Guesses: {correct}")
        print(f"Overall Accuracy: {(correct / total) * 100:.2f}%")
        print("="*40)

if __name__ == "__main__":
    main()