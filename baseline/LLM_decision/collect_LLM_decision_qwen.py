import argparse
import os
import re
import sqlite3
from typing import Dict, Iterable, List, Tuple
from vllm import LLM, SamplingParams

# ==========================================
# Configuration & Default Paths
# ==========================================
SOURCE_DB_PATH = "/projects/p32143/RL_human_decision/statistic/text_2026-09-02.db"
DEST_DB_PATH = "qwen36_27b_baseline.db"
MODEL_PATH = "/projects/p32143/cache/qwen36_27b"
MODEL_ID_NAME = "Qwen/Qwen3.6-27B"


def batched(items: List, batch_size: int) -> Iterable[List]:
    """Yield successive batch_size chunks from items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_with_chat_template(llm: LLM, prompt_text: str, enable_thinking: bool = False) -> str:
    """Formats the prompt using Qwen's chat template."""
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


def parse_prediction(raw_text: str) -> str:
    """Strips thinking tags if present and extracts a binary guess: 'AI' or 'Human'."""
    # Remove <think>...</think> block if thinking mode was active
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    
    # Match standalone 'AI' or 'Human' (case-insensitive)
    matches = re.findall(r"\b(AI|Human)\b", cleaned, flags=re.IGNORECASE)
    if matches:
        pred = matches[-1].strip().lower()
        return "AI" if pred == "ai" else "Human"
    
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen model baseline on text_guess database.")
    parser.add_argument("--source_db", type=str, default=SOURCE_DB_PATH)
    parser.add_argument("--dest_db", type=str, default=DEST_DB_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # Set to 1 GPU
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=1)            # Set to batch size 1
    parser.add_argument("--max_tokens", type=int, default=256) 
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen thinking tokens.")
    args = parser.parse_args()

    # 1. Connect to SQLite Databases
    src_conn = sqlite3.connect(args.source_db)
    src_cursor = src_conn.cursor()

    dest_conn = sqlite3.connect(args.dest_db)
    dest_cursor = dest_conn.cursor()

    # 2. Replicate Destination Table Schema
    dest_cursor.execute("""
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
    """)
    dest_conn.commit()

    # 3. Read Rows and Prepare Prompts
    src_cursor.execute("SELECT * FROM text_guess")
    rows = src_cursor.fetchall()
    columns = [desc[0] for desc in src_cursor.description]

    valid_samples: List[Tuple[Dict, str]] = []

    print(f"📂 Loading text and explanation files for {len(rows)} database entries...")
    for row in rows:
        row_dict = dict(zip(columns, row))
        text_id = row_dict["text_id"]
        label_truth = row_dict["label_truth"]
        explanation_source = row_dict["explanation_source"]

        # Resolve paths according to label_truth
        if label_truth == "AI":
            text_path = f"/projects/p32143/RL_human_decision/text_data/social/SocialMedia_rewrite/{text_id}.txt"
        elif label_truth == "Human":
            text_path = f"/projects/p32143/RL_human_decision/text_data/social/SocialMedia_Reddit/{text_id}.txt"
        else:
            continue

        expl_path = f"/projects/p32143/RL_human_decision/text_data/social/{explanation_source}/{text_id}.txt"

        try:
            with open(text_path, "r", encoding="utf-8") as tf:
                text_content = tf.read()
            with open(expl_path, "r", encoding="utf-8") as ef:
                explanation_content = ef.read()
        except FileNotFoundError as e:
            print(f"⚠️ File missing for text_id {text_id}: {e}")
            continue

        prompt = (
            "You are evaluating a text to determine if it was written by an AI or a Human.\n\n"
            f"Text:\n{text_content}\n\n"
            f"Explanation hint:\n{explanation_content}\n\n"
            "Based on the text and the explanation, make a decision. "
            'Respond with EXACTLY one word: either "AI" or "Human".'
        )
        valid_samples.append((row_dict, prompt))

    src_conn.close()
    print(f"✅ Successfully loaded {len(valid_samples)} valid samples.")

    # 4. Initialize vLLM Engine
    print("🚀 Initializing vLLM Engine on 1 GPU...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
        generation_config="vllm",
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 5. Run Batch Inference & Store Predictions
    print(f"🧠 Generating predictions (Batch size: {args.batch_size})...")
    for batch in batched(valid_samples, args.batch_size):
        row_dicts = [item[0] for item in batch]
        raw_prompts = [item[1] for item in batch]

        if args.use_chat_template:
            formatted_prompts = [
                format_with_chat_template(llm, p, args.enable_thinking) for p in raw_prompts
            ]
        else:
            formatted_prompts = raw_prompts

        # vLLM generation
        outputs = llm.generate(formatted_prompts, sampling)

        insert_records = []
        for row_dict, out in zip(row_dicts, outputs):
            generated_text = out.outputs[0].text if out.outputs else ""
            
            # guess = parse_prediction(generated_text)
            guess = generated_text
            is_correct = True if guess == row_dict["label_truth"] else False

            insert_records.append((
                row_dict["id"],
                MODEL_ID_NAME,
                row_dict["dataset"],
                row_dict["trial_num"],
                row_dict["stimulus_index"],
                row_dict["stimulus_id"],
                row_dict["text_id"],
                row_dict["label_truth"],
                row_dict["explanation_source"],
                guess,
                is_correct,
                "localhost",
                "Linux",
                "vLLM",
                "3.6",
                "Python",
            ))

        dest_cursor.executemany("""
            INSERT OR REPLACE INTO text_guess (
                id, user_id, dataset, trial_num, stimulus_index, stimulus_id, 
                text_id, label_truth, explanation_source, guess, correct, 
                ip, user_platform, user_browser, user_version, user_language, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, insert_records)
        dest_conn.commit()

        print(f"Processed item: text_id={row_dicts[0]['text_id']} | Truth={row_dicts[0]['label_truth']} | Guess={insert_records[0][9]}")

    dest_conn.close()
    print(f"\n🎉 Completed! Predictions saved to: {args.dest_db}")


if __name__ == "__main__":
    main()