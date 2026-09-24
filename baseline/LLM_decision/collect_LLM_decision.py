import sqlite3
import os
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================
SOURCE_DB_PATH = '/projects/p32143/RL_human_decision/statistic/text_2026-09-02.db'
DEST_DB_PATH = 'gpt56_baseline.db'

# Initialize OpenAI Client (assumes OPENAI_API_KEY is set in environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt56_guess(text_content, explanation_content):
    """Passes the text and explanation to GPT-5.6 and forces a binary guess."""
    prompt = f"""You are evaluating a text to determine if it was written by an AI or a Human.

Text:
{text_content}

Explanation hint:
{explanation_content}

Based on the text and the explanation, make a decision. 
Respond with EXACTLY one word: either "AI" or "Human".
"""

    # print(f"Sending prompt to GPT-5.6 for evaluation:\n{prompt}\n")
    
    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={"effort": "low"},
            input=[
                {"role": "developer", "content": "You are a binary classification assistant. Output only 'AI' or 'Human'."},
                {"role": "user", "content": prompt},
            ],
        )
        
        # Clean the response to ensure it matches the binary format
        guess = response.output_text.strip().capitalize()
        # Remove any trailing punctuation the model might add
        guess = ''.join(e for e in guess if e.isalnum())
        
        # if guess not in ['AI', 'Human']:
        #     guess = 'Unknown'

        # print(f"GPT-5.6 Guess: {guess}")
            
        return guess
    
    except Exception as e:
        print(f"API Call Failed: {e}")
        return "Error"

def main():
    # 1. Connect to Source and Destination Databases
    src_conn = sqlite3.connect(SOURCE_DB_PATH)
    src_cursor = src_conn.cursor()
    
    dest_conn = sqlite3.connect(DEST_DB_PATH)
    dest_cursor = dest_conn.cursor()
    
    # 2. Create Destination Table mimicking the exact original schema[cite: 1]
    dest_cursor.execute('''
    CREATE TABLE IF NOT EXISTS text_guess (
        id INTEGER NOT NULL, 
        user_id VARCHAR(150), 
        dataset VARCHAR(8), 
        trial_num INTEGER, 
        stimulus_index INTEGER, 
        stimulus_id VARCHAR(64), 
        text_id VARCHAR(50), 
        label_truth VARCHAR(10), 
        explanation_source VARCHAR(80), 
        guess VARCHAR(10), 
        correct BOOLEAN, 
        ip VARCHAR(150), 
        user_platform VARCHAR(150), 
        user_browser VARCHAR(150), 
        user_version VARCHAR(150), 
        user_language VARCHAR(150), 
        timestamp DATETIME, 
        PRIMARY KEY (id)
    )
    ''')
    
    # 3. Fetch all rows from the source database
    src_cursor.execute("SELECT * FROM text_guess")
    rows = src_cursor.fetchall()
    columns = [desc[0] for desc in src_cursor.description]
    
    # 4. Iterate through rows
    for row in rows:
        row_dict = dict(zip(columns, row))
        text_id = row_dict['text_id']
        label_truth = row_dict['label_truth']
        explanation_source = row_dict['explanation_source']
        
        # Determine Text Path based on label_truth
        if label_truth == 'AI':
            text_path = f"/projects/p32143/RL_human_decision/text_data/social/SocialMedia_rewrite/{text_id}.txt"
        elif label_truth == 'Human':
            text_path = f"/projects/p32143/RL_human_decision/text_data/social/SocialMedia_Reddit/{text_id}.txt"
        else:
            continue
            
        # Determine Explanation Path
        expl_path = f"/projects/p32143/RL_human_decision/text_data/social/{explanation_source}/{text_id}.txt"
        
        # 5. Read file contents
        try:
            with open(text_path, 'r', encoding='utf-8') as tf:
                text_content = tf.read()
            with open(expl_path, 'r', encoding='utf-8') as ef:
                explanation_content = ef.read()
        except FileNotFoundError as e:
            print(f"File missing for text_id {text_id}: {e}")
            continue
            
        # 6. Get GPT-5.6 Guess
        llm_guess = get_gpt56_guess(text_content, explanation_content)
        
        if llm_guess == "Error":
            continue
        
        # Determine if the LLM was correct
        is_correct = True if llm_guess == label_truth else False
        
        # 7. Insert the new row into the destination database
        # We override the human attributes with the LLM's attributes while keeping stimulus data identical.
        dest_cursor.execute('''
            INSERT INTO text_guess (
                id, user_id, dataset, trial_num, stimulus_index, stimulus_id, 
                text_id, label_truth, explanation_source, guess, correct, 
                ip, user_platform, user_browser, user_version, user_language, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            row_dict['id'], 
            'gpt-5.6-luna',            # Overridden user_id
            row_dict['dataset'], 
            row_dict['trial_num'], 
            row_dict['stimulus_index'], 
            row_dict['stimulus_id'], 
            text_id, 
            label_truth, 
            explanation_source, 
            llm_guess,                 # New LLM guess
            is_correct,                # Evaluated correct boolean
            'localhost', 
            'API', 
            'OpenAI_Python_SDK', 
            '5.6', 
            'Python'
        ))
        
        dest_conn.commit()
        print(f"Processed text_id: {text_id} | Truth: {label_truth} | GPT-5.6 Guess: {llm_guess}")
        # break

    src_conn.close()
    dest_conn.close()
    print("Database evaluation complete.")

if __name__ == "__main__":
    main()