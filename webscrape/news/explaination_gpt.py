import os
import torch
from openai import OpenAI


# 1. Define your directories
INPUT_DIR_ORIGINAL = "News"
OUTPUT_DIR_ORIGINAL = f"{INPUT_DIR_ORIGINAL}_explanation"

INPUT_DIR_REWRITE = "News_rewrite"
OUTPUT_DIR_REWRITE = f"{INPUT_DIR_REWRITE}_explanation"

print("Model loaded successfully!\n" + "="*50)

def explanation(text):
    """Passes the text to Qwen3 and returns the explaination."""
    # Craft a precise prompt for the model
    prompt = f"""Task: Analyze the provided News article for linguistic markers of AI or human authorship.

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

Article:
```
Health and Human Services' acting secretary has appointed Dr. Richard Besser as the interim director for the Centers for Disease Control and Prevention. William Gimson will return to his position as the CDC's chief operating officer. He replaces William Gimson, who took over as interim CDC director at noon on January 20. Gimson notified CDC employees that HHS acting secretary Charles E. Johnson had announced the appointment. Gimson replaced Dr. Julie Gerberding, who was the head of the CDC from 2002 until two days ago. Gerberding, along with other senior officials, also resigned on January 20, when Barack Obama and his administration took over. Past HHS secretary Michael Leavitt said that the interim directors would take over until the next HHS nominee -- former Sen. Tom Daschle -- is confirmed and makes the permanent appointments. Gimson told employees he's returning to his post as the CDC's chief operating officer. The CDC usually has a physician as its director, which Gimson is not. According to the biography posted on the CDC Web site, Besser's last position at the CDC was as the director of the Coordinating Office for Terrorism Preparedness and Emergency Response, where he was responsible for public health emergency preparedness and emergency response activities. According to CDC sources, Besser was seeing patients when he learned of his new position. In addition to heading the CDC bioterrorism preparedness division, he is a practicing pediatrician.
```

Explanation:
- Look for heavy use of acronyms.
- Look for dashes with space between them and the words (i.e., dashes that are not em dashes).
- Look for redundancy / repetition of the same word several times.

Article:
```
Dr. Richard Besser has been named interim director of the Centers for Disease Control and Prevention (CDC), succeeding William Gimson, who has returned to his position as chief operating officer. Besser, a board-certified pediatrician with extensive public health experience, previously led the CDC’s bioterrorism preparedness division and has long been recognized for his expertise in emergency response and infectious disease control. His appointment follows the recent resignations of Dr. Julie Gerberding and several other senior officials, raising concerns about leadership stability during a critical period for public health. The CDC faces mounting pressure to strengthen its response capabilities amid rising threats from emerging infectious diseases and ongoing public health challenges. As interim director, Besser will oversee day-to-day operations and ensure continuity in public health initiatives, including outbreak surveillance, vaccine distribution, and health promotion. His appointment underscores the administration’s effort to stabilize the agency amid internal transitions. The interim leadership will remain in place until Sen. Tom Daschle, the next nominee from the U.S. Department of Health and Human Services, is confirmed by Congress. Officials emphasize that Besser’s background in both clinical medicine and public health preparedness makes him well-suited to guide the CDC through this transitional phase.
```

Explanation:
- Look for similar sentence lengths.
- Look for complex sentence construction that reads well.
- Look for use of words like "highlight", "emphasize", "suggest."

Article:
```
{text}
```

Explanation:
"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        reasoning={
            "effort": "none"
        }
    )
    
    return response.output_text

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
