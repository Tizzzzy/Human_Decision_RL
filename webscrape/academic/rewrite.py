import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define your three directories
ORIGINAL_DIR = "Academic"
SUMMARY_DIR = "Academic_summary"
REWRITE_DIR = "Academic_rewrite"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# 2. Load Tokenizer and Model
print("Loading tokenizer and model into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="./hf_cache"  # Optional: specify a cache directory for faster subsequent loads
)
print("Model loaded successfully!\n" + "="*50)

def rewrite_text(summary, target_word_count):
    """Passes the summary to Qwen3 and asks it to expand to the target length."""
    
    # LLMs struggle with exact word counts, so we use strong instructional phrasing
    prompt = (
        f"You are an academic. I will provide you with a summary of an academic abstract. "
        f"Your task is to expand this summary and rewrite it into a full, detailed academic abstract. "
        f"It is CRITICAL that your rewritten text is approximately {target_word_count} words long. "
        f"After the opening sentence, use first-person narrative 'we' when detailing the methodology, findings, and conclusions.\n\n"
        f"Summary:\n{summary}"
    )
            # f"Do not use introductory filler phrases like 'This paper presents,' 'This work discusses,' or 'In this study.' "

    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
    
    # Generate the rewrite. We set max_new_tokens high enough to allow for the expansion.
    # A safe rule of thumb: 1 word is roughly 1.5 to 2 tokens.
    max_tokens_needed = int(target_word_count * 2.5) + 200 
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens_needed,
        temperature=0.7 # Slightly higher temperature here to allow for creative expansion
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    rewritten_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return rewritten_text

def process_rewrite_directory():
    if not os.path.exists(REWRITE_DIR):
        os.makedirs(REWRITE_DIR)

    # 3. Walk through the Academic_summary directory
    for root, dirs, files in os.walk(SUMMARY_DIR):
        for file in files:
            if file.endswith(".txt"):
                summary_file_path = os.path.join(root, file)
                
                # Figure out the relative path (e.g., 'Dec_2021')
                relative_path = os.path.relpath(root, SUMMARY_DIR)
                
                # Construct paths for the Original and Rewrite files
                original_file_path = os.path.join(ORIGINAL_DIR, relative_path, file)
                output_folder_path = os.path.join(REWRITE_DIR, relative_path)
                output_file_path = os.path.join(output_folder_path, file)
                
                # Create the specific month/year folder in the rewrite directory if needed
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                    
                # 4. Checkpointing: Skip if we already rewrote this file
                if os.path.exists(output_file_path):
                    continue
                    
                # Failsafe: Ensure the original file exists to calculate word count
                if not os.path.exists(original_file_path):
                    print(f"  -> Skipping: Original file missing for {file}")
                    continue
                
                print(f"Processing: {file}...")
                
                try:
                    # Read the summary
                    with open(summary_file_path, 'r', encoding='utf-8') as f:
                        summary_content = f.read().strip()
                        
                    # Read original to get target word count
                    with open(original_file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read().strip()
                        
                    target_words = len(original_content.split())
                    
                    if not summary_content or target_words == 0:
                        print(f"  -> Skipping empty file.")
                        continue
                        
                    # Generate the rewritten text
                    rewritten_content = rewrite_text(summary_content, target_words)
                    
                    # Write to the new mirrored directory
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(rewritten_content)
                        
                except Exception as e:
                    print(f"  -> Error processing {file}: {e}")

    print("\nBatch rewriting complete! Check your 'Academic_rewrite' folder.")

if __name__ == "__main__":
    process_rewrite_directory()