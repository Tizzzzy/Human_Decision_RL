import os
import torch
import anthropic


# 1. Define your directories
# INPUT_DIR_ORIGINAL = "SocialMedia_Reddit"
# OUTPUT_DIR_ORIGINAL = f"{INPUT_DIR_ORIGINAL}_explanation_claude"

INPUT_DIR_REWRITE = "SocialMedia_rewrite"
OUTPUT_DIR_REWRITE = f"{INPUT_DIR_REWRITE}_explanation_claude"

print("Model loaded successfully!\n" + "="*50)

def explanation(text):
    """Passes the text to Qwen3 and returns the explaination."""
    # Craft a precise prompt for the model
    prompt = f"""Task: Analyze the provided Social Media post for linguistic markers of AI or human authorship.

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

Post:
```
I've always felt a deep connection to Taurus—something quiet, steady, and truly special. I used to think the zodiac signs were all about flashy personalities or dramatic emotions, but Taurus? They're the ones who show up without fanfare. I've watched friends go through tough times, and every time, a Taurus friend has been there—calm, grounded, offering a warm cup of tea or just sitting in silence with you, listening. They don't need to fix everything. They just show up with presence. That's rare.  
Taurus isn't loud or flashy. They don't chase attention or try to be the center of the room. Instead, they build real, lasting connections. They're loyal—not just in words, but in action. When you're struggling, they don't offer quick fixes. They offer stability. They offer comfort. And that kind of emotional support? It's not just helpful—it's healing.  
I've seen how Taurus people nurture their loved ones with patience and care. They take time to understand what others need, not what they want. They're dependable, grounded, and deeply kind. In a world that often values speed and intensity, Taurus brings balance. They remind us that strength isn't always loud—it can be quiet, consistent, and deeply human.  
I've realized something: Taurus isn't just underrated. They're essential. They're the quiet pillars in our lives—steady, patient, and full of genuine warmth. I don't just appreciate them. I value them. And if you've ever had a Taurus friend, you know—this sign isn't just a zodiac sign. It's a gift. I'm so proud to stand beside them.
```

Explanation:
- Look for em dashes.
- Look for descriptions followed by elaboration, setting a familiar rhythm to the sentences. 
- Look for reasonable length paragraphs that tend to be consistent in length.

Post:
```
Taurus season started yesterday, and so I wanted to make a post dedicated to  appreciating Taurus since I feel like Taurus is never really talked about and is very underrated. I find that I come across a lot of Taureans (probably cause I'm a Taurus descendant) and almost all of them I meet I find them to be nothing short of amazing. I believe that Taurus is the kindest and loyalest sign out of the zodiac, and they're very caring as well. Taurus offers a stable support system that you can certainly rely on, they always offer a hand to those they care about. They're great at advice, and can help you see the beauty in life's most difficult situations. They will always be there to encourage you to keep pushing forward even when your life seems to be getting off track, and they do this in a very gentle, caring and nurturing way. If you're lucky enough to be friends with them you'll know that they offer a ride or die type friendship, and they will always be thoughtful of you. You will also be aware that the stereotype of them being boring is far from true, they will surely have you entertained even in the most mundane situations. They are as loyal as they come and will not only fend for you in your presence but in your absence as well. They offer genuineness, stability, loyalty and beauty to your life and I very much appreciate all the Taureans I've had the pleasure to befriend.
```

Explanation:
- Look for run on sentences and use of because to string together clauses.
- Look for colloquial expressions like "nothing short of X". 
- Look for non-descript positive words like "great" and "amazing."
- Look for overly long paragraphs or blocks of text.

Post:
```
{text}
```

You know that this post is AI-generated; can you provide an explanation that tells the reader what to focus on, without explicitly stating that it is AI-generated?

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
    files_to_process = all_txt_files

    for input_file_path in files_to_process:
        # if count >= 5:
        #     break
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
                count += 1
                f.write(explanation_content)
                
        except Exception as e:
            print(f"  -> Error processing {input_file_path}: {e}")

    print(f"--- Finished processing {input_dir}! Outputs saved to {output_dir} ---")


if __name__ == "__main__":
    # Process the original files first
    # process_directory(INPUT_DIR_ORIGINAL, OUTPUT_DIR_ORIGINAL)
    
    # Then process the AI rewritten files
    process_directory(INPUT_DIR_REWRITE, OUTPUT_DIR_REWRITE)
    
    print("\nAll batches complete!")
