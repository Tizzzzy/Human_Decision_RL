import os
import csv
import sys
import kagglehub

# 1. Force kagglehub to download into your specific project path
os.environ["KAGGLEHUB_CACHE"] = "/projects/p32143/RL_human_decision/webscrape/social"

# Increase CSV field size limit (Reddit posts can occasionally be massive and crash the parser)
csv.field_size_limit(sys.maxsize)

TOTAL_POSTS = 10000
OUTPUT_DIR = "SocialMedia_Reddit"

def format_text(text):
    """Cleans up the text by removing newlines and extra spaces."""
    if not text:
        return ""
    return text

def extract_reddit_kaggle():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Downloading dataset via kagglehub...\n")
    
    # This will download the files into the path you specified above
    dataset_path = kagglehub.dataset_download("prakharrathi25/reddit-data-huge")
    print(f"Dataset ready at: {dataset_path}\n")
    print("=" * 60)

    posts_processed = 0

    # 2. Loop through all the downloaded CSV files
    for filename in os.listdir(dataset_path):
        if posts_processed >= TOTAL_POSTS:
            break
            
        # Skip the subreddits.txt file or any non-CSVs
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(dataset_path, filename)
        print(f"Scanning {filename}...")

        try:
            # Using csv.DictReader is much safer for your RAM than loading the whole CSV at once
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Failsafe: Ensure the file actually has a 'Body' column
                if not reader.fieldnames or 'Body' not in reader.fieldnames:
                    print(f"  -> Skipping {filename}: No 'Body' column found.")
                    continue
                
                for row in reader:
                    if posts_processed >= TOTAL_POSTS:
                        break
                        
                    raw_body = row.get('Body', '')
                    
                    # 3. Skip empty fields and Reddit moderation placeholders
                    if not raw_body or str(raw_body).strip() == "":
                        continue
                        
                    clean_body = format_text(str(raw_body))
                    
                    if clean_body in ["[removed]", "[deleted]", "nan"]:
                        continue
                        
                    # Skip extremely short posts (like "ok" or "thanks") to keep data quality high
                    if len(clean_body) < 100:
                        continue

                    posts_processed += 1
                    
                    # 4. Save to text file
                    out_filename = os.path.join(OUTPUT_DIR, f"reddit_{posts_processed}.txt")
                    with open(out_filename, 'w', encoding='utf-8') as out_f:
                        out_f.write(clean_body)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "=" * 60)
    print(f"Successfully extracted {posts_processed} Reddit posts to the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    extract_reddit_kaggle()