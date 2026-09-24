import os
import time
import requests
from bs4 import BeautifulSoup
import json

TOTAL_POEMS = 10000
BASE_DIR = "Poetry"
STATE_FILE = "poetry_scraping_state.json"
BASE_URL = "https://poets.org"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            seen_urls = set(state.get("seen_urls", []))
            poems_processed = state.get("poems_processed", 0)
            page_index = state.get("page_index", 0)
                
            print(f"Resuming from checkpoint: {poems_processed} poems processed.")
            return seen_urls, poems_processed, page_index
            
    return set(), 0, 0

def save_state(seen_urls, poems_processed, page_index):
    state = {
        "seen_urls": list(seen_urls), 
        "poems_processed": poems_processed,
        "page_index": page_index
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def scrape_poets_org():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    seen_urls, poems_processed, page_index = load_state()
    
    # Using a session keeps the connection alive and is polite to the server
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    print(f"Target: {TOTAL_POEMS} pre-2023 poems...\n")

    while poems_processed < TOTAL_POEMS:
        # Load the directory page
        list_url = f"{BASE_URL}/poems?page={page_index}"
        print(f"Scanning directory page {page_index}...")
        
        try:
            response = session.get(list_url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to load page {page_index}. Status: {response.status_code}")
                time.sleep(10)
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('tr')
            
            # If the table is empty or missing, we have reached the end of the site
            if not rows or len(rows) < 2:
                print("No more poems found. Reached the end of the directory.")
                break
            
            for row in rows:
                if poems_processed >= TOTAL_POEMS:
                    break
                    
                title_cell = row.find('td', class_='views-field-title')
                date_cell = row.find('td', class_='views-field-field-date-published')

                if title_cell and date_cell:
                    link_tag = title_cell.find('a', href=True)
                    time_tag = date_cell.find('time')
                    
                    if link_tag and time_tag:
                        poem_title = link_tag.text.strip()
                        poem_url = BASE_URL + link_tag['href']
                        
                        # 1. Duplicate check
                        if poem_url in seen_urls:
                            continue
                            
                        # 2. Extract and check the year
                        year_str = time_tag.text.strip()
                        try:
                            year = int(year_str)
                        except ValueError:
                            continue

                        # 3. Filter rule: strictly before 2023
                        if year < 2023:
                            try:
                                poem_resp = session.get(poem_url, timeout=10)
                                poem_soup = BeautifulSoup(poem_resp.text, 'html.parser')

                                poem_body = poem_soup.find('div', class_='field field--body')

                                if not poem_body:
                                    # Mark as seen so we don't retry broken pages
                                    seen_urls.add(poem_url)
                                    continue

                                # 4. Extract stanzas and clean lines
                                full_poem = []
                                stanzas = poem_body.find_all('p')

                                for stanza in stanzas:
                                    raw_stanza = stanza.get_text(separator='\n')
                                    clean_lines = [line.strip() for line in raw_stanza.split('\n') if line.strip()]
                                    clean_stanza = "\n".join(clean_lines)
                                    
                                    if clean_stanza:
                                        full_poem.append(clean_stanza)

                                final_poem_text = "\n\n".join(full_poem)

                                # 5. Save the file
                                if final_poem_text:
                                    poems_processed += 1
                                    seen_urls.add(poem_url)
                                    
                                    file_path = os.path.join(BASE_DIR, f"poem_{poems_processed}.txt")
                                    with open(file_path, 'w', encoding='utf-8') as f:
                                        f.write(final_poem_text)
                                        
                                    print(f"Saved {poems_processed}/{TOTAL_POEMS}: '{poem_title}' ({year})")
                                    
                                # Polite delay between individual poems
                                time.sleep(1)

                            except Exception as e:
                                print(f"Error scraping individual poem {poem_url}: {e}")
                                seen_urls.add(poem_url)
                                continue

            # Move to the next directory page and save state
            page_index += 1
            save_state(seen_urls, poems_processed, page_index)
            
            # Polite delay between directory pages
            time.sleep(3)

        except Exception as e:
            print(f"\nAn error occurred on directory page {page_index}: {e}")
            save_state(seen_urls, poems_processed, page_index)
            print("Sleeping for 60 seconds before retrying...")
            time.sleep(60)

    print("\nPoetry scraping finished. Check your 'Poetry' folder!")

if __name__ == "__main__":
    scrape_poets_org()