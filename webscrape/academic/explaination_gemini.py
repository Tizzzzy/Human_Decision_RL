import os
import torch
from google import genai
import time

# 1. Define your directories
INPUT_DIR_ORIGINAL = "Academic"
OUTPUT_DIR_ORIGINAL = "Academic_explanation_gemini"

INPUT_DIR_REWRITE = "Academic_rewrite"
OUTPUT_DIR_REWRITE = "Academic_rewrite_explanation_gemini"

print("Model loaded successfully!\n" + "="*50)

def explanation(text):
    """Passes the text to Qwen3 and returns the explaination."""
    # Craft a precise prompt for the model
    prompt = f"""Task: Analyze the provided academic abstract for linguistic markers of AI or human authorship.

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

Abstract:
We introduce TyDiP, a novel multilingual dataset comprising 4.5K three-way politeness annotations across nine typologically diverse languages, designed to capture the nuanced interplay between formality, politeness, and cultural context. Using this dataset, we evaluate the performance of multilingual models in zero-shot identification of politeness levels, revealing robust cross-linguistic generalization despite notable gaps compared to human annotators. Our analysis highlights consistent patterns in English politeness strategies across languages, suggesting shared pragmatic structures. However, we observe significant variation in how formality and politeness are encoded, indicating language-specific pragmatic norms. These findings underscore the challenges in achieving human-level accuracy in cross-linguistic politeness recognition and emphasize the need for richer, culturally grounded annotation frameworks in future multilingual NLP research.

Explanation:
- Look for compound nouns with multiple adjectives.
- Look for sentence structures where the initial clause is followed by a comma then a supporting clause.
- Look for overly formal word choices like "observe" instead of "see", "utilize" instead of "use."

Abstract:
We study politeness phenomena in nine typologically diverse languages. Politeness is an important facet of communication and is sometimes argued to be cultural-specific, yet existing computational linguistic study is limited to English. We create TyDiP, a dataset containing three-way politeness annotations for 500 examples in each language, totaling 4.5K examples. We evaluate how well multilingual models can identify politeness levels -- they show a fairly robust zero-shot transfer ability, yet fall short of estimated human accuracy significantly. We further study mapping the English politeness strategy lexicon into nine languages via automatic translation and lexicon induction, analyzing whether each strategy's impact stays consistent across languages. Lastly, we empirically study the complicated relationship between formality and politeness through transfer experiments. We hope our dataset will support various research questions and applications, from evaluating multilingual models to constructing polite multilingual agents.

Explanation:
- Look for generic words like important, various, multiple.
- Look for hedging words like sometimes, may, can
- Look for awkward phrasing
- Look for references to hoping, believing, wanting
- Look for application of pronouns like they to refer to abstract entities

Abstract:
{text}

Explanation:
"""
    
    client = genai.Client(api_key="AIzaSyCpLt9fECSRhiH3c3k-XxarU5HL94yuja8")

    response = client.models.generate_content(
        # model="gemini-3.1-pro-preview", 
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
    process_directory(INPUT_DIR_ORIGINAL, OUTPUT_DIR_ORIGINAL)
    
    # Then process the AI rewritten files
    process_directory(INPUT_DIR_REWRITE, OUTPUT_DIR_REWRITE)
    
    print("\nAll batches complete!")
