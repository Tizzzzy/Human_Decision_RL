import os
import time
import json
import random
import urllib.request
import urllib.error

TOTAL_BOOKS = 1000
BASE_DIR = "Books"
STATE_FILE = "gutenberg_random_state.json"

def load_state():
    """Loads the last processed count and the list of IDs already attempted."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            books_processed = state.get("books_processed", 0)
            # Convert list back to a set for lightning-fast duplicate checking
            tried_ids = set(state.get("tried_ids", [])) 
            print(f"Resuming from checkpoint: {books_processed} books processed.")
            return books_processed, tried_ids
    return 0, set()

def save_state(books_processed, tried_ids):
    """Saves progress after every book attempt."""
    state = {
        "books_processed": books_processed,
        "tried_ids": list(tried_ids)
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def scrape_1000_random_books():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    books_processed, tried_ids = load_state()
    
    print(f"Target: {TOTAL_BOOKS} random books...\n")

    while books_processed < TOTAL_BOOKS:
        # 1. Generate a random ID between 1300 and 15000
        book_id = random.randint(1300, 15000)
        
        # 2. Duplicate Check: If we already tried this ID, skip and pick a new one
        if book_id in tried_ids:
            continue
            
        tried_ids.add(book_id)
        
        # Construct the URL based on the random ID
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req).read().decode('utf-8')
            
            # Save the file using the book ID in the name for traceability
            file_path = os.path.join(BASE_DIR, f"book_{book_id}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response)
                
            books_processed += 1
            print(f"Saved {books_processed}/{TOTAL_BOOKS}: Book ID {book_id}")
            
            # Save state and wait politely
            save_state(books_processed, tried_ids)
            time.sleep(2)
            
        except urllib.error.HTTPError as e:
            # Handle missing books (404) by saving state and moving on
            if e.code == 404:
                print(f"Book ID {book_id} not found (404). Skipping...")
                save_state(books_processed, tried_ids)
                time.sleep(1) 
                
            # Handle getting temporarily blocked (403) by Gutenberg
            elif e.code == 403:
                print(f"HTTP 403 Forbidden. Gutenberg might be rate-limiting you.")
                print("Sleeping for 5 minutes to let the server cool down...")
                save_state(books_processed, tried_ids)
                time.sleep(300) 
            else:
                print(f"HTTP Error {e.code} for ID {book_id}. Sleeping 60s...")
                save_state(books_processed, tried_ids)
                time.sleep(60)
                
        except Exception as e:
            print(f"An unexpected error occurred at ID {book_id}: {e}")
            save_state(books_processed, tried_ids)
            print("Sleeping 60 seconds before retrying...")
            time.sleep(60)

    print("\nRandom Fiction scraping complete! Check your 'Fiction' folder.")

if __name__ == "__main__":
    scrape_1000_random_books()