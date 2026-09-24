import os
import torch
import anthropic


# 1. Define your directories
INPUT_DIR_ORIGINAL = "Poetry"
OUTPUT_DIR_ORIGINAL = f"{INPUT_DIR_ORIGINAL}_explanation_claude"

INPUT_DIR_REWRITE = "Poetry_rewrite"
OUTPUT_DIR_REWRITE = f"{INPUT_DIR_REWRITE}_explanation_claude"

print("Model loaded successfully!\n" + "="*50)

def explanation(text):
    """Passes the text to Qwen3 and returns the explaination."""
    # Craft a precise prompt for the model
    prompt = f"""Task: Analyze the provided poem for linguistic markers of AI or human authorship.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list. 
2. Do NOT provide an introduction, a final verdict, or a summary.
3. Do NOT group the bullet points into sub-headings (keep it a flat list).
4. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category or type of phrasing]."
    - "Look for [linguistic category or type of phrasing] like '[short example]'."
5. When providing examples, you may list multiple words separated by commas (e.g., like "word 1", "word 2").
6. Do NOT use bolding, italics, or any special Markdown formatting.
7. Limit the output to exactly 2 to 6 bullet points.

Examples:

Poem:
```
She stirs where words fall short, a quiet flame,  
Not born of thought, but deep within the bone.  
The ego clings to form, to name, to claim—  
She moves the stillness, breaks the self's false throne.  
No gaze upon the self, but self upon her face.  
Awakening blooms not in the mind, but in the breath.  
She is the unseen, the first, the true—  
The seer is her echo, the seen, her light.
```

Explanation:
- Loook for rhyming and a consistent rhythm. 
- Look for consistency in the length of lines and of words used.

Poem:
```
The lightest realizations arrive in restraint—
so the old masters tell us.

Not unlike the tug at the end of a line.

We have language for what is within reach
but not the mutable form behind it.

Or else, why write.

I'm sick of peering at the ego.
No, my ego's tired of peering at
me
—

It's she who awakens me into being.

So it goes: the seer mistaken for the seen.
```

Explanation:
- Look for inconsistent line lengths.
- Look for personal phrasing/first person tense.
- Look for endings that feel abrupt rather than neatly wrapped up. 
- Look for incomplete sentences and phrases.

Poem:
```
{text}
```

Explanation:
"""
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=512,
    )
    
    # print(response.content[0].text)
    return response.content[0].text

def process_directory(input_dir, output_dir):
    """Walks through a specified input directory and saves model outputs to a specified output directory."""
    print(f"\n--- Starting processing for: {input_dir} ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0

    # 3. Walk through the specified input directory

    all_txt_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                # Store the full path so we can sort properly
                all_txt_files.append(os.path.join(root, file))
    
    all_txt_files.sort()
    files_to_process = all_txt_files[:3000]

    for input_file_path in files_to_process:
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_dir)
        file_name = os.path.basename(input_file_path)
        output_folder_path = os.path.join(output_dir, relative_path)
        output_file_path = os.path.join(output_folder_path, file_name)

        os.makedirs(output_folder_path, exist_ok=True)
                                
        # 4. Checkpointing: Skip if we already processed this file
        if os.path.exists(output_file_path):
            continue
                
        print(f"Processing: {input_file_path}...")
        
        try:
            # Read the text
            with open(input_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
                
            if not text_content:
                print(f"  -> Skipping empty file.")
                continue
                
            # Generate the explanation
            explanation_content = explanation(text_content)
            
            # Write the explanation to the new mirrored directory
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(explanation_content)
                
        except Exception as e:
            print(f"  -> Error processing {input_file_path}: {e}")

    print(f"--- Finished processing {input_dir}! Outputs saved to {output_dir} ---")

if __name__ == "__main__":
    # Process the original files first
    process_directory(INPUT_DIR_ORIGINAL, OUTPUT_DIR_ORIGINAL)
    
    # Then process the AI rewritten files
    process_directory(INPUT_DIR_REWRITE, OUTPUT_DIR_REWRITE)
    
    print("\nAll batches complete!")
