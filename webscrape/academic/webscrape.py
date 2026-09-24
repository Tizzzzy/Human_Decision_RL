import os
import time
import urllib.request
import urllib.error  # Added to handle specific HTTP errors
import feedparser
import json
import calendar
from datetime import datetime
import random

TOTAL_PAPERS = 10000
BATCH_SIZE = 500  
BASE_DIR = "Academic"
STATE_FILE = "scraping_state.json"

def format_text(text):
    if not text:
        return ""
    return " ".join(text.replace('\n', ' ').split())

def generate_date_ranges():
    ranges = []
    for year in range(2021, 1999, -1):
        start_month = 11 if year == 2022 else 12
        for month in range(start_month, 0, -1):
            last_day = calendar.monthrange(year, month)[1]
            start_str = f"{year}{month:02d}010000"
            end_str = f"{year}{month:02d}{last_day}2359"
            ranges.append(f"[{start_str}+TO+{end_str}]")

    return ranges

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            seen_ids = set(state.get("seen_ids", []))
            papers_processed = state.get("papers_processed", 0)
            start_index = state.get("start_index", 0)
            date_index = state.get("date_index", 0)
                
            print(f"Resuming from checkpoint: {papers_processed} papers processed.")
            return seen_ids, papers_processed, start_index, date_index
            
    return set(), 0, 0, 0

def save_state(seen_ids, papers_processed, start_index, date_index):
    state = {
        "seen_ids": list(seen_ids), 
        "papers_processed": papers_processed,
        "start_index": start_index,
        "date_index": date_index
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def scrape_strict_10k_abstracts():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    base_url = 'http://export.arxiv.org/api/query?'
    seen_paper_ids, papers_processed, start_index, date_index = load_state()
    date_ranges = generate_date_ranges()

    print(f"Target: {TOTAL_PAPERS} unique abstracts...")

    while papers_processed < TOTAL_PAPERS:
        if date_index >= len(date_ranges):
            print("Exhausted all date ranges!")
            break

        current_date_range = date_ranges[date_index]
        # search_query = f'all:the+AND+submittedDate:{current_date_range}'
        search_query = f'cat:cs*+AND+submittedDate:{current_date_range}' # <-- NEW VERSION
        
        query = f"search_query={search_query}&sortBy=submittedDate&sortOrder=descending&start={start_index}&max_results={BATCH_SIZE}"
        url = base_url + query
        
        readable_month = f"{current_date_range[5:7]}/{current_date_range[1:5]}"
        print(f"Fetching papers from {readable_month} at API index {start_index}...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req).read()
            feed = feedparser.parse(response)
            
            if not feed.entries:
                print(f"No more entries for {readable_month}. Moving to previous month.")
                date_index += 1
                start_index = 0
                save_state(seen_paper_ids, papers_processed, start_index, date_index)
                continue

            for entry in feed.entries:
                if papers_processed >= TOTAL_PAPERS:
                    break
                
                paper_id = entry.id
                if paper_id in seen_paper_ids:
                    continue
                
                if 'summary_detail' not in entry or 'value' not in entry.summary_detail:
                    continue
                
                raw_abstract = entry.summary_detail['value']
                seen_paper_ids.add(paper_id)
                papers_processed += 1
                
                published_parsed = entry.published_parsed
                pub_date = datetime.fromtimestamp(time.mktime(published_parsed))
                folder_name = pub_date.strftime("%b_%Y")
                abstract_text = format_text(raw_abstract)

                if len(abstract_text.split()) < 100:  # Skip very short abstracts
                    continue

                folder_path = os.path.join(BASE_DIR, folder_name)
                
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                abs_filename = os.path.join(folder_path, f"abs_{papers_processed}.txt")
                
                with open(abs_filename, 'w', encoding='utf-8') as f:
                    f.write(abstract_text)

            print(f"Successfully saved {papers_processed}/{TOTAL_PAPERS} unique abstracts.")
            
            start_index += BATCH_SIZE
            save_state(seen_paper_ids, papers_processed, start_index, date_index)
            
            # Increased standard sleep to 5 seconds to be a bit gentler on the API
            time.sleep(5) 

        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("\nHTTP 429: Too Many Requests. arXiv is rate-limiting us.")
                print("Sleeping for 60 seconds to let the server cool down...")
                time.sleep(60)
                # We use 'continue' to restart the loop WITHOUT incrementing the start_index.
                # This ensures we retry the exact batch that failed.
                continue
            else:
                print(f"\nAn HTTP error occurred: {e}")
                save_state(seen_paper_ids, papers_processed, start_index, date_index)
                break
                
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            save_state(seen_paper_ids, papers_processed, start_index, date_index)
            break

    print("\nScraping script finished.")

if __name__ == "__main__":
    scrape_strict_10k_abstracts()