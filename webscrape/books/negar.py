import pandas as pd
import uuid
import os

def process_stimuli_data(input_csv_path, label, text_base_folder):
    """
    Reads a CSV containing File_1 and File_2 pairs, extracts metadata, 
    and formats them into individual rows for D1 and D2 sets.
    """
    # Load the CSV
    df = pd.read_csv(input_csv_path)
    
    formatted_rows = []
    
    for _, row in df.iterrows():
        # Your CSV has columns 'File_1', 'File_2'
        file_1_path = row['File_1']
        file_2_path = row['File_2']
        
        # Extract the text_id (e.g., from 'books_paragraph_explanation/paragraph_3600.txt' -> 'paragraph_3600')
        filename = os.path.basename(file_1_path)
        text_id = os.path.splitext(filename)[0]
        
        # Construct the text_path pointing back to the original text folder
        text_path = f"{text_base_folder}/{filename}"
        
        # Create row for D1 (using File 1)
        formatted_rows.append({
            'stimulus_id': uuid.uuid4().hex,  # Generates a random unique hash
            'text_id': text_id,
            'text_path': text_path,
            'explanation_path': file_1_path,
            'label': label,
            'dataset_split': 'D1'
        })
        
        # Create row for D2 (using File 2)
        formatted_rows.append({
            'stimulus_id': uuid.uuid4().hex,
            'text_id': text_id,
            'text_path': text_path,
            'explanation_path': file_2_path,
            'label': label,
            'dataset_split': 'D2'
        })
        
    return pd.DataFrame(formatted_rows)

if __name__ == "__main__":
    # 1. Process the Human-written paragraphs
    # (Make sure the CSV filename matches your actual file)
    df_human = process_stimuli_data(
        input_csv_path='most_different_files_books_paragraph.csv', 
        label='Human', 
        text_base_folder='books_paragraph'
    )
    
    # 2. Process the AI-written rewrites 
    # (Replace with the actual name of your AI csv file)
    df_ai = process_stimuli_data(
        input_csv_path='most_different_files_books_rewrite.csv', 
        label='AI', 
        text_base_folder='books_rewrite'
    )
    
    # 3. Combine them into one master dataset
    master_df = pd.concat([df_human, df_ai], ignore_index=True)
    
    # 4. Save to a new CSV for your coworker
    output_filename = 'master_stimuli_database.csv'
    master_df.to_csv(output_filename, index=False)
    
    print(f"Success! Master CSV saved as: {output_filename}")
    print(f"Total stimuli generated: {len(master_df)}")
    print("\nPreview of the data:")
    print(master_df.head())