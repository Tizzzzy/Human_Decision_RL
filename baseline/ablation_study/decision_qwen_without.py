import json
import torch
import csv
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc
from collections import defaultdict
import re

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen3.6-27B"
MODEL_PATH = "/projects/p32143/cache/qwen36_27b"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
CSV_FILE = "/projects/p32143/RL_human_decision/text_data/social/master_stimuli_database_socialmedia.csv"
OUTPUT_PREFERENCE = "decision_qwen_without.json"

# vLLM configuration
TENSOR_PARALLEL_SIZE = 2
DTYPE = "auto"
MAX_MODEL_LEN = 262144
GPU_MEMORY_UTILIZATION = 0.85
BATCH_SIZE = 1
MAX_TOKENS = 1024
TEMPERATURE = 0.5
TOP_P = 0.95

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of AI vs. Human text detection heuristics.

I will show you a social media text and you need to determine whether the text was written by a human or an AI.

Text:
{text}

First, provide a brief 1-2 sentence analytical justification noting specific linguistic markers (e.g., formatting, tone, phrasing). 
Then, on a new line, provide your final verdict wrapped in label tags.

Format your response EXACTLY like this:
[1-2 sentence justification]
<label>AI</label> OR <label>HUMAN</label>"""

# ==========================================
# Helper Functions
# ==========================================
def format_with_chat_template(llm: LLM, content: list, enable_thinking: bool = False) -> str:
    """Format prompt using the model's chat template."""
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Some tokenizer/vLLM versions pass Qwen chat-template kwargs this way.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )

def read_text_file(filepath):
    """Read text from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return None

def judge_text(llm: LLM, sampling_params: SamplingParams, text_id, text_content):

    # Format the judge prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        text=text_content
    )

    try:
        # Format prompt using chat template
        content = [{"type": "text", "text": prompt}]
        final_prompt_str = format_with_chat_template(llm, content, enable_thinking=False)

        # Generate judgment using vLLM
        outputs = llm.generate(final_prompt_str, sampling_params)
        response = outputs[0].outputs[0].text.strip()
        print(f"📝  {text_id}: Judge response: {response}")

        # Parse response to extract the predicted label using regex
        # This safely grabs whatever is inside the <label> tags
        match = re.search(r'<label>(.*?)</label>', response, re.IGNORECASE)
        prediction = match.group(1).strip().upper() if match else "UNKNOWN"

        return {
            'prediction': prediction,
            'judge_response': response
        }

    except Exception as e:
        print(f"⚠️  {text_id}: Error during judgment: {e}")
        import traceback
        traceback.print_exc()
        return None    

def load_text_from_csv(csv_file, text_data_dir):
    """
    Load explanations from CSV file and corresponding text files.
    """
    text_ids_to_texts = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_id = row['text_id']
                label = row['label']
                text_path = row['text_path']

                # Create composite key to avoid overwrites (text_id_human or text_id_ai)
                composite_key = f"{text_id}_{label.lower()}"

                # Load text content
                full_text_path = f"{text_data_dir}{text_path}"
                try:
                    with open(full_text_path, 'r', encoding='utf-8') as tf:
                        text_ids_to_texts[composite_key] = tf.read().strip()
                except Exception as e:
                    print(f"Error loading text {composite_key} from {full_text_path}: {e}")

    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return {}

    return text_ids_to_texts


def main():
    print("="*80)
    print("Ranking Explanations with Qwen/Qwen3.6-27B Judge (vLLM)")
    print("="*80)

    # ==========================================
    # 1. Load explanations from CSV
    # ==========================================
    print("\n[1/4] Loading explanations from CSV...")
    text_ids_to_texts = load_text_from_csv(CSV_FILE, TEXT_DATA_DIR)

    if not text_ids_to_texts:
        print("❌ No texts found. Exiting.")
        return

    print(f"✓ Loaded {len(text_ids_to_texts)} texts to evaluate.")

    # ==========================================
    # 2. Load Qwen/Qwen3.6-27B judge (vLLM)
    # ==========================================
    print("\n[2/4] Loading Qwen/Qwen3.6-27B judge with vLLM...")
    try:
        print("🚀 Initializing vLLM engine (this happens only once)...")
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype=DTYPE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
            generation_config="vllm",
        )

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )

        print("✓ Judge loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load judge: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 3. Judge explanations
    # ==========================================
    print("\n[3/4] Judging explanations...")
    judge_results = {}
    
    # Process items with a progress bar
    for composite_key, text_content in tqdm(text_ids_to_texts.items(), desc="Evaluating"):
        result = judge_text(llm, sampling_params, composite_key, text_content)

        if result:
            # Extract true label from the end of the composite key
            true_label = composite_key.split('_')[-1].upper()
            
            judge_results[composite_key] = {
                'true_label': true_label,
                'prediction': result['prediction'],
                'raw_response': result['judge_response'],
                'text_content': text_content
            }

    # ==========================================
    # 4. Save and Summarize Results
    # ==========================================
    print(f"\n[4/4] Saving results to {OUTPUT_PREFERENCE}...")
    try:
        with open(OUTPUT_PREFERENCE, 'w', encoding='utf-8') as f:
            json.dump(judge_results, f, indent=4)
        print("✓ Results saved successfully.")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        
    # Calculate and display basic accuracy
    correct = sum(1 for res in judge_results.values() if res['true_label'] == res['prediction'])
    total = len(judge_results)
    
    if total > 0:
        print("\n" + "="*40)
        print("🎯 FINAL RESULTS SUMMARY")
        print("="*40)
        print(f"Total Evaluated: {total}")
        print(f"Correct Guesses: {correct}")
        print(f"Overall Accuracy: {(correct / total) * 100:.2f}%")
        print("="*40)

if __name__ == "__main__":
    main()