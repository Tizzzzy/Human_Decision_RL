import os
import torch
from openai import OpenAI

# 1. Define your three directories
ORIGINAL_DIR = "SocialMedia_Reddit"
SUMMARY_DIR = "SocialMedia_Summary"
REWRITE_DIR = "SocialMedia_rewrite"

# Initialize your specific OpenAI client
# (Ensure your actual API key is set safely, e.g., via environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_text(summary, target_word_count):
    """Passes the summary to the GPT model and asks it to expand to the target length."""
    
    # Advanced Human-Like Prompt Engineering
    prompt = (
        f"You are a highly active, authentic Reddit user tasked with rewriting a summary into a casual social media text post. "
        f"CRITICAL REQUIREMENT: Your rewrite must be approximately {target_word_count} words long.\n\n"
        f"To ensure the text reads like a real human posting on a forum and not an AI generation, strictly follow these rules:\n"
        f"1. Dive straight into the topic like you are speaking to a friend or dropping into an ongoing community discussion.\n"
        f"2. Do NOT use bullet points, numbered lists, emojis, generic language, or an overuse of em-dashes. Do Not list words in groups of three.\n\n"
        f"Summary to rewrite:\n{summary}"
    )
    
    try:
        # Using the specific generation snippet provided
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt,
            reasoning={
                "effort": "none"
            }
        )
        return response.output_text
        
    except Exception as e:
        print(f"API Error during generation: {e}")
        return ""

def process_rewrite_directory():
    if not os.path.exists(REWRITE_DIR):
        os.makedirs(REWRITE_DIR)
    
    count = 0

    # 3. Walk through the summary directory
    for root, dirs, files in os.walk(SUMMARY_DIR):
        for file in files:
            if file.endswith(".txt"):

                # if count >= 2:
                #     print("\nReached processing limit of 10 files. Stopping.")
                #     return
                
                summary_file_path = os.path.join(root, file)
                
                # Figure out the relative path
                relative_path = os.path.relpath(root, SUMMARY_DIR)
                
                # Construct paths for the Original and Rewrite files
                original_file_path = os.path.join(ORIGINAL_DIR, relative_path, file)
                output_folder_path = os.path.join(REWRITE_DIR, relative_path)
                output_file_path = os.path.join(output_folder_path, file)
                
                # Create the specific sub-folder in the rewrite directory if needed
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                    
                # 4. Checkpointing: Skip if we already rewrote this file
                if os.path.exists(output_file_path):
                    continue
                    
                # Failsafe: Ensure the original file exists to calculate word count
                if not os.path.exists(original_file_path):
                    print(f"  -> Skipping: Original file missing for {file}")
                    continue
                
                print(f"Processing: {file}... ({count + 1})")
                
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
                    
                    if rewritten_content:
                        count += 1
                        # Write to the new mirrored directory
                        with open(output_file_path, 'w', encoding='utf-8') as f:
                            f.write(rewritten_content)
                    else:
                        print(f"  -> Skipping save for {file} due to empty API response.")
                        
                except Exception as e:
                    print(f"  -> Error processing {file}: {e}")

    print("\nBatch rewriting complete! Check your rewrite folder.")

if __name__ == "__main__":
    process_rewrite_directory()