import pandas as pd
import glob
import os
import json

def merge_speeches_and_parties(input_dir, output_file):
    speech_files = glob.glob(os.path.join(input_dir, 'speeches_*.txt'))
    
    # Trackers for our final distribution printout
    total_saved = 0
    party_counts = {'D': 0, 'R': 0}
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for speech_file in speech_files:
            filename = os.path.basename(speech_file)
            session = filename.split('_')[1].split('.')[0]
            map_file = os.path.join(input_dir, f'{session}_SpeakerMap.txt')
            
            if not os.path.exists(map_file):
                print(f"Skipping {session}: SpeakerMap not found.")
                continue
                
            print(f"Processing session {session}...")
            
            try:
                # Load with encoding_errors='replace' to handle OCR artifacts
                df_speeches = pd.read_csv(speech_file, sep='|', encoding='utf-8', on_bad_lines='skip', dtype=str, encoding_errors='replace')
                df_map = pd.read_csv(map_file, sep='|', encoding='utf-8', on_bad_lines='skip', dtype=str, encoding_errors='replace')
                
                if 'speech_id' not in df_speeches.columns or 'speech_id' not in df_map.columns:
                    print(f"Skipping {session}: Missing 'speech_id' column.")
                    continue
                    
                # Subset to only the columns we care about
                df_speeches = df_speeches[['speech_id', 'speech']]
                df_map = df_map[['speech_id', 'party']]
                
                # Pre-filter to keep ONLY Democrats ('D') and Republicans ('R')
                df_map = df_map[df_map['party'].isin(['D', 'R'])]
                
                # Merge the dataframes
                merged_df = pd.merge(df_speeches, df_map, on='speech_id', how='inner')
                merged_df = merged_df.dropna(subset=['speech', 'party'])
                
                # Iterate and write to JSONL
                for record in merged_df.to_dict(orient='records'):
                    text = str(record['speech'])
                    label = record['party']
                    
                    # Calculate word count based on whitespace
                    word_count = len(text.split())
                    
                    # Apply length filters: Between 20 and 250 words
                    if 50 <= word_count <= 60:
                        json_record = {
                            "text": text,
                            "label": label
                        }
                        out_f.write(json.dumps(json_record) + '\n')
                        
                        # Update our trackers
                        total_saved += 1
                        party_counts[label] += 1
                        
            except Exception as e:
                print(f"Error processing session {session}: {e}")

    # Print the final data count and distribution
    print("\n" + "="*45)
    print("PROCESSING COMPLETE")
    print("="*45)
    print(f"Total Valid Speeches Saved: {total_saved:,}")
    if total_saved > 0:
        pct_d = (party_counts['D'] / total_saved) * 100
        pct_r = (party_counts['R'] / total_saved) * 100
        print(f"Democrats (D):   {party_counts['D']:,} ({pct_d:.2f}%)")
        print(f"Republicans (R): {party_counts['R']:,} ({pct_r:.2f}%)")
    print("="*45)

# --- Execution ---
INPUT_DIRECTORY = 'hein-bound'
OUTPUT_FILE = 'filtered_speeches.jsonl'

merge_speeches_and_parties(INPUT_DIRECTORY, OUTPUT_FILE)