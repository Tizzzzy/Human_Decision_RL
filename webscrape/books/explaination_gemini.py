import os
import torch
from google import genai
import time

# 1. Define your directories
INPUT_DIR_ORIGINAL = "books_paragraph"
OUTPUT_DIR_ORIGINAL = f"{INPUT_DIR_ORIGINAL}_explanation_gemini"

INPUT_DIR_REWRITE = "books_rewrite"
OUTPUT_DIR_REWRITE = f"{INPUT_DIR_REWRITE}_explanation_gemini"

print("Model loaded successfully!\n" + "="*50)

def explanation(text):
    """Passes the text to Qwen3 and returns the explaination."""
    # Craft a precise prompt for the model
    prompt = f"""Task: Analyze the provided fiction paragraph for linguistic markers of AI or human authorship.

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

Paragraph:
Liverpool, a bustling hub nestled between the Atlantic currents, functions as a vital stopover for cargo and passengers ferrying between New York and South American capitals. Delegates from nations across the continent arrive in the U.S. via this route, their eyes fixed on the promise of mutual prosperity. Though formal trade and diplomatic channels remain absent, their presence signals a growing demand for structured cooperation. The port becomes more than a transit point—it becomes a catalyst, urging the U.S. to forge meaningful, equitable relationships with South American nations, laying the foundation for lasting economic and political synergy.

Explanation:
- Look for authoritative sounding summaries.
- Look for em dashes.
- Look for multiple adjectives of similar length used together.

Paragraph:
The present situation is such that travelers and merchandise find Liverpool often a necessary intermediate port between New York and some of the South American capitals. The fact that some of the delegates from South American States to the conference of American nations now in session at Washington reached our shores by reversing that line of travel is very conclusive of the need of such a conference and very suggestive as to the first and most necessary step in the direction of fuller and more beneficial intercourse with nations that are now our neighbors upon the lines of latitude, but not upon the lines of established commercial intercourse.

Explanation:
- Look for awkward sounding word placements.
- Look for use of non-descript adjectives like "that" and "this."
- Look for unnecessarily formal language like "upon" instead of "on."

Paragraph:
{text}

Explanation:
"""
    
    client = genai.Client(api_key="AIzaSyCpLt9fECSRhiH3c3k-XxarU5HL94yuja8")

    response = client.models.generate_content(
        # model="gemini-3-flash-preview", 
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
    )
    # print(response.text)


    return response.text

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
    files_to_process = all_txt_files

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
        
        success = False

        while not success:
            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                if not text_content:
                    print(f"  -> Skipping empty file.")
                    success = True # Mark as "done" so we don't retry an empty file
                    continue
                
                # Generate the explanation
                explanation_content = explanation(text_content)
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(explanation_content)
                
                success = True # Exit the while loop

            except Exception as e:
                time.sleep(60)
                print(f"  -> Error processing {input_file_path}: {e}. Retrying...")

    print(f"--- Finished processing {input_dir}! Outputs saved to {output_dir} ---")


if __name__ == "__main__":
    # Process the original files first
    # process_directory(INPUT_DIR_ORIGINAL, OUTPUT_DIR_ORIGINAL)
    
    # Then process the AI rewritten files
    process_directory(INPUT_DIR_REWRITE, OUTPUT_DIR_REWRITE)
    
    print("\nAll batches complete!")
