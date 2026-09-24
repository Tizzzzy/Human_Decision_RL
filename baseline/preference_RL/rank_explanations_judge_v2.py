"""
Use Qwen/Qwen3.6-27B as judge to select the better explanation from explanation pairs.

For each text (AI and Human separately):
1. Load 2 explanations from explanations_pairs.json
2. Judge selects the better explanation using Qwen/Qwen3.6-27B
3. Create preference pair: (chosen, rejected)
4. Save to preference_pairs_ai.json and preference_pairs_human.json

Output format:
{
  "text_id": {
    "text": "the original text",
    "chosen_explanation": "the selected explanation",
    "chosen_source": "source of selected explanation",
    "rejected_explanations": [
      {"source": "...", "text": "..."}
    ]
  }
}
"""

import json
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc
import os

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen3.6-27B"
MODEL_PATH = "/projects/p32143/cache/qwen36_27b"
EXPLANATIONS_PAIRS_JSON = "/projects/p32143/Human_RL_baselines/data/explanations_pairs.json"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"

# Inference file paths for text labels
INFERENCE_HUMANRL = "/projects/p32143/Human_RL_baselines/inference_humanRL.json"

OUTPUT_PREFERENCE_AI = "preference_pairs_ai.json"
OUTPUT_PREFERENCE_HUMAN = "preference_pairs_human.json"

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

Your task: Select which explanation MORE effectively captures the key linguistic markers that would help detect whether this text is AI or human.

Consider these criteria:
1. Clarity: Is the explanation clear and specific?
2. Relevance: Does it identify markers actually present in the text?
3. Usefulness: Would this explanation be most helpful for classification?
4. Distinctiveness: Does it capture markers unique to AI or human writing?
5. Coverage: Does it capture more relevant linguistic features?

Text:
```
{text}
```

Explanation 1:
{exp_1}

Explanation 2:
{exp_2}

Respond with ONLY the number (1 or 2) of the better explanation, followed by a brief 1-2 sentence justification. Format:
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

def judge_explanation_pair(llm: LLM, sampling_params: SamplingParams, text_id, text_content,
                          exp_1, exp_1_source, exp_2, exp_2_source):
    """
    Use Qwen/Qwen3.6-27B judge to select the better explanation from 2 options.

    Args:
        llm: vLLM LLM instance
        sampling_params: SamplingParams for generation
        text_id: identifier for the text
        text_content: the original text
        exp_1: first explanation text
        exp_1_source: source of first explanation
        exp_2: second explanation text
        exp_2_source: source of second explanation

    Returns:
        dict with chosen explanation index and source, or None on error
    """
    # Format the judge prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        text=text_content,
        exp_1=exp_1,
        exp_2=exp_2,
    )

    try:
        # Format prompt using chat template
        content = [{"type": "text", "text": prompt}]
        final_prompt_str = format_with_chat_template(llm, content, enable_thinking=False)

        # Generate judgment using vLLM
        outputs = llm.generate(final_prompt_str, sampling_params)
        response = outputs[0].outputs[0].text.strip()
        print(f"📝  {text_id}: Judge response: {response[:100]}")

        # Parse response to extract the chosen explanation index
        lines = response.split('\n')
        chosen_idx = None

        for line in lines:
            line = line.strip()
            if line.isdigit():
                idx = int(line)
                if idx in [1, 2]:
                    chosen_idx = idx
                    break

        if chosen_idx is None:
            print(f"⚠️  {text_id}: Could not parse judge response, defaulting to explanation 1")
            chosen_idx = 1

        return {
            'chosen_idx': chosen_idx,
            'chosen_source': exp_1_source if chosen_idx == 1 else exp_2_source,
            'chosen_text': exp_1 if chosen_idx == 1 else exp_2,
            'rejected_source': exp_2_source if chosen_idx == 1 else exp_1_source,
            'rejected_text': exp_2 if chosen_idx == 1 else exp_1,
            'judge_response': response
        }

    except Exception as e:
        print(f"⚠️  {text_id}: Error during judgment: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to explanation 1
        return {
            'chosen_idx': 1,
            'chosen_source': exp_1_source,
            'chosen_text': exp_1,
            'rejected_source': exp_2_source,
            'rejected_text': exp_2,
            'judge_response': f'Error: {str(e)}'
        }

def load_explanation_pairs(json_file):
    """Load explanation pairs from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {}

def load_inference_data(json_file):
    """Load inference data to get text labels (AI/Human)."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
            # Build text_id -> label mapping
            text_labels = {}
            for record in records:
                text_id = record['text_id']
                label = record['label']
                if text_id not in text_labels:
                    text_labels[text_id] = label
            return text_labels
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {}

def create_preference_pairs(judge_results, text_ids_to_texts):
    """
    Create preference pairs from judge results.

    Args:
        judge_results: dict of text_id -> judgment result
        text_ids_to_texts: dict of text_id -> text_content

    Returns:
        preference pairs dataset
    """
    preference_pairs = {}

    for text_id, judgment in judge_results.items():
        if judgment is None:
            continue

        text_content = text_ids_to_texts.get(text_id)

        if not text_content:
            continue

        preference_pairs[text_id] = {
            'text': text_content,
            'chosen_explanation': judgment['chosen_text'],
            'chosen_source': judgment['chosen_source'],
            'rejected_explanations': [
                {
                    'source': judgment['rejected_source'],
                    'text': judgment['rejected_text']
                }
            ]
        }

    return preference_pairs

def main():
    print("="*80)
    print("Ranking Explanation Pairs with Qwen/Qwen3.6-27B Judge (vLLM)")
    print("="*80)

    # ==========================================
    # 1. Load explanation pairs and inference data
    # ==========================================
    print("\n[1/4] Loading explanation pairs and inference data...")

    explanation_pairs = load_explanation_pairs(EXPLANATIONS_PAIRS_JSON)
    text_labels = load_inference_data(INFERENCE_HUMANRL)

    print(f"  Total explanation pairs: {len(explanation_pairs)}")
    print(f"  Text labels loaded: {len(text_labels)}")

    if not explanation_pairs or not text_labels:
        print("❌ No data found. Exiting.")
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
    # 3. Judge explanation pairs
    # ==========================================
    print("\n[3/4] Judging explanation pairs...")

    # Map text_ids to actual text content
    text_ids_to_texts = {}

    judge_results_ai = {}
    judge_results_human = {}

    # Separate pairs by label
    ai_pairs = {}
    human_pairs = {}

    for key, pair in explanation_pairs.items():
        # Extract text_id from key (format: "redis_XXXX_AI.txt" or "redis_XXXX_Human.txt")
        text_id = key.rsplit('_', 1)[0]  # Remove the suffix

        if text_id not in text_labels:
            print(f"⚠️  {text_id}: Not found in text labels, skipping")
            continue

        label = text_labels[text_id]

        if label == "AI":
            ai_pairs[text_id] = pair
        else:
            human_pairs[text_id] = pair

    print(f"  AI pairs to judge: {len(ai_pairs)}")
    print(f"  Human pairs to judge: {len(human_pairs)}")

    # Process AI texts
    print("\n  Processing AI texts...")
    for text_id, pair in tqdm(ai_pairs.items(), desc="AI texts"):
        # Load actual text content
        text_content = read_text_file(
            f"{TEXT_DATA_DIR}SocialMedia_rewrite/{text_id}.txt"
        )

        if text_content is None:
            print(f"⚠️  {text_id}: Could not read text file")
            continue

        text_ids_to_texts[text_id] = text_content

        exp_1 = pair['explanation_1']['content']
        exp_1_source = pair['explanation_1']['source_path']
        exp_2 = pair['explanation_2']['content']
        exp_2_source = pair['explanation_2']['source_path']

        judgment = judge_explanation_pair(
            llm, sampling_params, text_id, text_content,
            exp_1, exp_1_source, exp_2, exp_2_source
        )
        judge_results_ai[text_id] = judgment

        # Cleanup
        gc.collect()

    # Process Human texts
    print("\n  Processing Human texts...")
    for text_id, pair in tqdm(human_pairs.items(), desc="Human texts"):
        # Load actual text content
        text_content = read_text_file(
            f"{TEXT_DATA_DIR}SocialMedia_Reddit/{text_id}.txt"
        )

        if text_content is None:
            print(f"⚠️  {text_id}: Could not read text file")
            continue

        text_ids_to_texts[text_id] = text_content

        exp_1 = pair['explanation_1']['content']
        exp_1_source = pair['explanation_1']['source_path']
        exp_2 = pair['explanation_2']['content']
        exp_2_source = pair['explanation_2']['source_path']

        judgment = judge_explanation_pair(
            llm, sampling_params, text_id, text_content,
            exp_1, exp_1_source, exp_2, exp_2_source
        )
        judge_results_human[text_id] = judgment

        # Cleanup
        gc.collect()

    # ==========================================
    # 4. Create and save preference pairs
    # ==========================================
    print("\n[4/4] Creating preference pairs...")

    preference_ai = create_preference_pairs(judge_results_ai, text_ids_to_texts)
    preference_human = create_preference_pairs(judge_results_human, text_ids_to_texts)

    # Save AI preferences
    with open(OUTPUT_PREFERENCE_AI, 'w', encoding='utf-8') as f:
        json.dump(preference_ai, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(preference_ai)} AI preference pairs to {OUTPUT_PREFERENCE_AI}")

    # Save Human preferences
    with open(OUTPUT_PREFERENCE_HUMAN, 'w', encoding='utf-8') as f:
        json.dump(preference_human, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(preference_human)} Human preference pairs to {OUTPUT_PREFERENCE_HUMAN}")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"AI preference pairs: {len(preference_ai)}")
    print(f"Human preference pairs: {len(preference_human)}")
    print(f"Total preference pairs: {len(preference_ai) + len(preference_human)}")

    # Count chosen explanation sources
    ai_sources = {}
    for item in preference_ai.values():
        source = item['chosen_source']
        ai_sources[source] = ai_sources.get(source, 0) + 1

    human_sources = {}
    for item in preference_human.values():
        source = item['chosen_source']
        human_sources[source] = human_sources.get(source, 0) + 1

    print("\nChosen explanation sources (AI texts):")
    for source in sorted(ai_sources.keys()):
        print(f"  {source}: {ai_sources[source]}")

    print("\nChosen explanation sources (Human texts):")
    for source in sorted(human_sources.keys()):
        print(f"  {source}: {human_sources[source]}")

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review the preference pairs in the JSON files")
    print("  2. Run train_rl_dpo.py to train model using these preferences")
    print("="*80)

if __name__ == "__main__":
    main()
