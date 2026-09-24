import os
from datasets import load_dataset

TOTAL_ARTICLES = 2000
BASE_DIR = "News"

def extract_first_paragraph(text):
    """
    Extracts the first paragraph, removes datelines, and checks word count.
    Returns the cleaned text if it passes the constraints, otherwise returns None.
    """
    if not text:
        return None
    
    # Remove dateline (e.g., "LONDON, England (CNN) -- ")
    parts = text.split(" -- ", 1)
    text = parts[1] if len(parts) > 1 else text
    
    # Split the text by newlines to isolate paragraphs
    # We use strip() to ignore empty strings/blank lines
    paragraphs = [p.strip() for p in text.strip().split('\n') if p.strip()]
    
    if not paragraphs:
        return None
        
    # Take only the first paragraph (or the whole thing if it's just one)
    first_paragraph = paragraphs[0]
    
    # Clean up any weird extra spaces inside the paragraph
    clean_text = " ".join(first_paragraph.split())
    
    # Calculate word count
    word_count = len(clean_text.split())
    
    # Apply length constraints: greater than 250 or less than 20 words
    if word_count < 20 or word_count > 250:
        return None
        
    return clean_text

def extract_news_dataset():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    print("Connecting to Hugging Face to stream CNN/DailyMail dataset...\n")
    
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
        
        articles_processed = 0
        
        print(f"Target: {TOTAL_ARTICLES} news articles...\n")

        for entry in dataset:
            if articles_processed >= TOTAL_ARTICLES:
                break
                
            raw_article = entry.get('article', '')
            
            # Process the text with the new paragraph and word count logic
            clean_article = extract_first_paragraph(raw_article)
            
            # If the article failed the word count check or is empty, skip to the next one
            if not clean_article:
                continue
                
            articles_processed += 1
            
            # Save the file
            file_path = os.path.join(BASE_DIR, f"news_{articles_processed}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_article)
                
            # Print an update every 1,000 files
            if articles_processed % 1000 == 0:
                print(f"Successfully saved {articles_processed}/{TOTAL_ARTICLES} articles...")

        print(f"\nNews extraction complete! Check your '{BASE_DIR}' folder.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_news_dataset()