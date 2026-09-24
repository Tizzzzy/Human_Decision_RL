"""
Use Qwen/Qwen3.5-9B as judge to rank explanations and create preference pairs.

For each text (AI and Human separately):
1. Load 6 explanations
2. Judge selects the best explanation using Qwen/Qwen3.5-9B
3. Create preference pair: (chosen, [rejected_1, rejected_2, ...])
4. Save to preference_pairs_ai.json and preference_pairs_human.json

Output format:
{
  "text_id": {
    "text": "the original text",
    "chosen_explanation": "the best explanation",
    "chosen_source": "source of best explanation",
    "rejected_explanations": [
      {"source": "...", "text": "..."},
      ...
    ]
  }
}
"""

import json
import torch
import csv
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc
from collections import defaultdict

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen3.6-27B"
MODEL_PATH = "/projects/p32143/cache/qwen36_27b"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
CSV_FILE = "/projects/p32143/RL_human_decision/text_data/social/master_stimuli_database_socialmedia.csv"
OUTPUT_PREFERENCE = "preference_pairs.json"

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

I will show you a text and 2 different explanations that describe linguistic markers for determining whether the text is AI-generated or human-written.

Your task: Select the SINGLE BEST explanation that most effectively captures the key linguistic markers that would help detect whether this text is AI or human.

Consider these criteria:
1. Clarity: Is the explanation clear and specific?
2. Relevance: Does it identify markers actually present in the text?
3. Usefulness: Would this explanation be most helpful for classification?
4. Distinctiveness: Does it capture markers unique to AI or human writing?

Text:
```
{text}
```

Explanations:
1. {exp_1}
2. {exp_2}

Respond with ONLY the number (1-2) of the best explanation, followed by a brief 1-2 sentence justification. Format:
[NUMBER]
[Justification]"""

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

def judge_explanations(llm: LLM, sampling_params: SamplingParams, text_id, text_content, explanations):
    """
    Use Qwen/Qwen3.6-27B judge to select the best explanation.

    Args:
        llm: vLLM LLM instance
        sampling_params: SamplingParams for generation
        text_id: identifier for the text
        text_content: the original text
        explanations: list of 2 explanation dicts with 'source' and 'text' keys

    Returns:
        dict with chosen explanation and index, or None on error
    """
    if len(explanations) < 2:
        print(f"⚠️  {text_id}: Only {len(explanations)} explanations found, skipping")
        return None

    # Format the judge prompt
    exp_texts = [exp['text'] for exp in explanations]
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        text=text_content,  # Limit text length to avoid token explosion
        exp_1=exp_texts[0],
        exp_2=exp_texts[1],
    )

    try:
        # Format prompt using chat template
        content = [{"type": "text", "text": prompt}]
        final_prompt_str = format_with_chat_template(llm, content, enable_thinking=False)

        # Generate judgment using vLLM
        outputs = llm.generate(final_prompt_str, sampling_params)
        response = outputs[0].outputs[0].text.strip()
        print(f"📝  {text_id}: Judge response: {response}")

        # Parse response to extract the chosen explanation index
        lines = response.split('\n')
        chosen_idx = None

        for line in lines:
            line = line.strip()
            if line.isdigit():
                idx = int(line)
                if 1 <= idx <= 2:
                    chosen_idx = idx - 1  # Convert to 0-indexed
                    break

        if chosen_idx is None:
            print(f"⚠️  {text_id}: Could not parse judge response: {response}")
            # Fallback: pick the first (default to existing explanations)
            chosen_idx = 0

        return {
            'chosen_idx': chosen_idx,
            'chosen_source': explanations[chosen_idx]['source'],
            'chosen_text': explanations[chosen_idx]['text'],
            'judge_response': response
        }

    except Exception as e:
        print(f"⚠️  {text_id}: Error during judgment: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_explanations_from_csv(csv_file, text_data_dir):
    """
    Load explanations from CSV file and corresponding explanation files.

    For each text_id, there are 2 versions (Human and AI), each with 2 explanations.
    Keys are formatted as text_id_human and text_id_ai to avoid overwrites.

    Args:
        csv_file: path to the CSV file
        text_data_dir: directory containing text and explanation files

    Returns:
        dict mapping (text_id_label) -> {label, text_content, explanations: [{source, text}, ...]}
    """
    explanations_data = defaultdict(lambda: {'explanations': []})
    text_ids_to_texts = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_id = row['text_id']
                label = row['label']
                explanation_path = row['explanation_path']
                text_path = row['text_path']

                # Create composite key to avoid overwrites (text_id_human or text_id_ai)
                composite_key = f"{text_id}_{label.lower()}"

                # Load explanation text
                full_explanation_path = f"{text_data_dir}{explanation_path}"
                try:
                    with open(full_explanation_path, 'r', encoding='utf-8') as ef:
                        explanation_text = ef.read().strip()

                    # Extract source from explanation_path (e.g., "SocialMedia_Reddit_explanation" or "SocialMedia_rewrite_explanation_gemini")
                    source = explanation_path.split('/')[0]

                    explanations_data[composite_key]['explanations'].append({
                        'source': source,
                        'text': explanation_text
                    })
                    explanations_data[composite_key]['label'] = label

                    # Load text content if not already loaded
                    if composite_key not in text_ids_to_texts:
                        full_text_path = f"{text_data_dir}{text_path}"
                        try:
                            with open(full_text_path, 'r', encoding='utf-8') as tf:
                                text_ids_to_texts[composite_key] = tf.read().strip()
                        except Exception as e:
                            print(f"Error loading text {composite_key} from {full_text_path}: {e}")

                except Exception as e:
                    print(f"Error loading explanation {full_explanation_path}: {e}")

    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return {}, {}

    return dict(explanations_data), text_ids_to_texts

def create_preference_pairs(judge_results, explanations_data, text_ids_to_texts):
    """
    Create preference pairs from judge results.

    Args:
        judge_results: dict of composite_key -> judgment result
        explanations_data: dict of composite_key -> {label, explanations: [...]}
        text_ids_to_texts: dict of composite_key -> text_content

    Returns:
        preference pairs dataset
    """
    preference_pairs = {}

    for composite_key, judgment in judge_results.items():
        if judgment is None:
            continue

        explanations = explanations_data.get(composite_key, {}).get('explanations', [])
        text_content = text_ids_to_texts.get(composite_key)
        label = explanations_data.get(composite_key, {}).get('label', '')

        if not text_content or len(explanations) < 2:
            continue

        chosen_idx = judgment['chosen_idx']
        chosen = explanations[chosen_idx]
        rejected = [exp for i, exp in enumerate(explanations) if i != chosen_idx]

        preference_pairs[composite_key] = {
            'text': text_content,
            'label': label,
            'chosen_explanation': chosen['text'],
            'chosen_source': chosen['source'],
            'judge_response': judgment['judge_response'],
            'rejected_explanations': rejected
        }

    return preference_pairs

def main():
    print("="*80)
    print("Ranking Explanations with Qwen/Qwen3.6-27B Judge (vLLM)")
    print("="*80)

    # ==========================================
    # 1. Load explanations from CSV
    # ==========================================
    print("\n[1/4] Loading explanations from CSV...")
    explanations_data, text_ids_to_texts = load_explanations_from_csv(CSV_FILE, TEXT_DATA_DIR)

    print(f"  Texts with explanations: {len(explanations_data)}")

    if not explanations_data:
        print("❌ No explanations found. Exiting.")
        return

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

    # Process all texts
    print("\n  Judging all texts...")
    for text_id, data in tqdm(explanations_data.items(), desc="Texts"):
        text_content = text_ids_to_texts.get(text_id)

        if text_content is None:
            continue

        explanations = data.get('explanations', [])
        judgment = judge_explanations(llm, sampling_params, text_id, text_content, explanations)
        judge_results[text_id] = judgment

        # Cleanup
        gc.collect()

    # ==========================================
    # 4. Create and save preference pairs
    # ==========================================
    print("\n[4/4] Creating preference pairs...")

    preference_pairs = create_preference_pairs(judge_results, explanations_data, text_ids_to_texts)

    # Save preferences
    with open(OUTPUT_PREFERENCE, 'w', encoding='utf-8') as f:
        json.dump(preference_pairs, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(preference_pairs)} preference pairs to {OUTPUT_PREFERENCE}")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total preference pairs: {len(preference_pairs)}")

    # Count by label
    human_count = sum(1 for item in preference_pairs.values() if item['label'] == 'Human')
    ai_count = sum(1 for item in preference_pairs.values() if item['label'] == 'AI')
    print(f"  Human texts: {human_count}")
    print(f"  AI texts: {ai_count}")

    # Count chosen explanation sources by label
    print("\nChosen explanation sources (Human texts):")
    human_sources = {}
    for item in preference_pairs.values():
        if item['label'] == 'Human':
            source = item['chosen_source']
            human_sources[source] = human_sources.get(source, 0) + 1
    for source in sorted(human_sources.keys()):
        print(f"  {source}: {human_sources[source]}")

    print("\nChosen explanation sources (AI texts):")
    ai_sources = {}
    for item in preference_pairs.values():
        if item['label'] == 'AI':
            source = item['chosen_source']
            ai_sources[source] = ai_sources.get(source, 0) + 1
    for source in sorted(ai_sources.keys()):
        print(f"  {source}: {ai_sources[source]}")

if __name__ == "__main__":
    main()
