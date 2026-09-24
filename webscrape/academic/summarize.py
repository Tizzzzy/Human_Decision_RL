import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define your directories
INPUT_DIR = "Academic"
OUTPUT_DIR = "Academic_summary"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# 2. Load Tokenizer and Model (Outside the loop!)
print("Loading tokenizer and model into memory. This might take a moment...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="./hf_cache"  # Optional: specify a cache directory for faster subsequent loads
)
print("Model loaded successfully!\n" + "="*50)

def summarize_text(text):
    """Passes the text to Qwen3 and returns the summary."""
    # Craft a precise prompt for the model
    prompt = f"""Summarize the following academic text in 3 sentences or fewer. Start the response immediately with the summary content. Strictly omit all introductions, meta-commentary, and conversational fillers.    

{text}
"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
    
    # Generate the summary
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512, # Reduced from 16384. Summaries rarely need more than 500 tokens.
        temperature=0.1     # Lower temperature keeps the summary factual and focused
    )
    
    # Extract just the new generated tokens, ignoring the prompt
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return summary

def process_academic_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. Walk through the Academic directory
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".txt"):
                input_file_path = os.path.join(root, file)
                
                # Figure out the relative path (e.g., 'Apr_2000') to recreate the folder
                relative_path = os.path.relpath(root, INPUT_DIR)
                output_folder_path = os.path.join(OUTPUT_DIR, relative_path)
                
                # Create the specific month/year folder in the summary directory if needed
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                    
                output_file_path = os.path.join(output_folder_path, file)
                
                # 4. Checkpointing: Skip if we already summarized this file
                if os.path.exists(output_file_path):
                    continue
                
                print(f"Processing: {input_file_path}...")
                
                try:
                    # Read the original text
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                        
                    if not text_content:
                        print(f"  -> Skipping empty file.")
                        continue
                        
                    # Generate the summary
                    summary_content = summarize_text(text_content)
                    
                    # Write the summary to the new mirrored directory
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(summary_content)
                        
                except Exception as e:
                    print(f"  -> Error processing {input_file_path}: {e}")

    print("\nBatch summarization complete! Check your 'Academic_summary' folder.")

if __name__ == "__main__":
    process_academic_directory()