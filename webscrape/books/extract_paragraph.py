import os
import re
import random

INPUT_DIR = "Books"
OUTPUT_DIR = "books_paragraph"
PARAGRAPHS_PER_BOOK = 10

def extract_random_paragraphs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # The regex pattern to isolate the core text
    pattern = re.compile(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*(.*?)\*\*\* END OF THE PROJECT GUTENBERG EBOOK", 
        re.DOTALL
    )

    books_processed = 0
    global_paragraph_counter = 1  # This will count from 1 to 10,000+

    print(f"Scanning '{INPUT_DIR}' for books...\n")

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".txt"):
            continue
            
        input_path = os.path.join(INPUT_DIR, filename)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. Extract the core book text
            match = pattern.search(content)
            if not match:
                continue

            core_text = match.group(1)

            # 2. Split into paragraphs by double newline
            raw_paragraphs = core_text.split('\n\n')

            # 3. Clean and filter the paragraphs
            valid_paragraphs = []
            for p in raw_paragraphs:
                clean_p = " ".join(p.replace('\n', ' ').split())
                if len(clean_p.split()) > 100:
                    valid_paragraphs.append(clean_p)

            # 4. Randomly sample 10 paragraphs
            sample_size = min(PARAGRAPHS_PER_BOOK, len(valid_paragraphs))
            if sample_size == 0:
                continue
                
            sampled_paragraphs = random.sample(valid_paragraphs, sample_size)

            # 5. Save EACH paragraph as its own individual text file
            for paragraph in sampled_paragraphs:
                output_path = os.path.join(OUTPUT_DIR, f"paragraph_{global_paragraph_counter}.txt")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(paragraph)
                    
                # Increment the counter so the next file is paragraph_2.txt, etc.
                global_paragraph_counter += 1

            books_processed += 1
            
            # Print an update every 100 books to show both books and total paragraphs
            if books_processed % 100 == 0:
                print(f"Processed {books_processed} books... (Total paragraphs saved: {global_paragraph_counter - 1})")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "="*50)
    print(f"Extraction complete! Successfully processed {books_processed} books.")
    print(f"Generated {global_paragraph_counter - 1} individual paragraph files in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    extract_random_paragraphs()