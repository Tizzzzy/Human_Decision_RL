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
from vllm import LLM, SamplingParams
from tqdm import tqdm
import gc

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "Qwen/Qwen3.6-27B"
MODEL_PATH = "/projects/p32143/cache/qwen36_27b"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
EXPLANATIONS_AI_JSON = "explanations_gemma4_ai.json"
EXPLANATIONS_HUMAN_JSON = "explanations_gemma4_human.json"
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

I will show you a text and 6 different explanations that describe linguistic markers for determining whether the text is AI-generated or human-written.

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
3. {exp_3}
4. {exp_4}
5. {exp_5}
6. {exp_6}

Respond with ONLY the number (1-6) of the best explanation, followed by a brief 1-2 sentence justification. Format:
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
        explanations: list of 6 explanation dicts with 'source' and 'text' keys

    Returns:
        dict with chosen explanation and index, or None on error
    """
    if len(explanations) < 6:
        print(f"⚠️  {text_id}: Only {len(explanations)} explanations found, skipping")
        return None

    # Format the judge prompt
    exp_texts = [exp['text'] for exp in explanations]
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        text=text_content,  # Limit text length to avoid token explosion
        exp_1=exp_texts[0],
        exp_2=exp_texts[1],
        exp_3=exp_texts[2],
        exp_4=exp_texts[3],
        exp_5=exp_texts[4],
        exp_6=exp_texts[5],
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
                if 1 <= idx <= 6:
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

def load_explanations(json_file):
    """Load explanations from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {}

def create_preference_pairs(judge_results, explanations_data, text_ids_to_texts):
    """
    Create preference pairs from judge results.

    Args:
        judge_results: dict of text_id -> judgment result
        explanations_data: dict of text_id -> explanations
        text_ids_to_texts: dict of text_id -> text_content

    Returns:
        preference pairs dataset
    """
    preference_pairs = {}

    for text_id, judgment in judge_results.items():
        if judgment is None:
            continue

        explanations = explanations_data.get(text_id, {}).get('explanations', [])
        text_content = text_ids_to_texts.get(text_id)

        if not text_content or len(explanations) < 6:
            continue

        chosen_idx = judgment['chosen_idx']
        chosen = explanations[chosen_idx]
        rejected = [exp for i, exp in enumerate(explanations) if i != chosen_idx]

        preference_pairs[text_id] = {
            'text': text_content,
            'chosen_explanation': chosen['text'],
            'chosen_source': chosen['source'],
            'rejected_explanations': rejected
        }

    return preference_pairs

def main():
    print("="*80)
    print("Ranking Explanations with Qwen/Qwen3.6-27B Judge (vLLM)")
    print("="*80)

    # ==========================================
    # 1. Load explanations
    # ==========================================
    print("\n[1/4] Loading explanations...")
    explanations_ai = load_explanations(EXPLANATIONS_AI_JSON)
    explanations_human = load_explanations(EXPLANATIONS_HUMAN_JSON)

    print(f"  AI texts: {len(explanations_ai)}")
    print(f"  Human texts: {len(explanations_human)}")

    if not explanations_ai and not explanations_human:
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

    # Map text_ids to actual text content for preference pairs
    text_ids_to_texts = {}

    judge_results_ai = {}
    judge_results_human = {}

    # Process AI texts
    print("\n  Processing AI texts...")
    for text_id, data in tqdm(explanations_ai.items(), desc="AI texts"):
        text_content = read_text_file(
            f"{TEXT_DATA_DIR}SocialMedia_rewrite/{text_id}.txt"
        )

        if text_content is None:
            continue

        text_ids_to_texts[text_id] = text_content
        explanations = data.get('explanations', [])
        judgment = judge_explanations(llm, sampling_params, text_id, text_content, explanations)
        judge_results_ai[text_id] = judgment

        # Cleanup
        gc.collect()

    # Process Human texts
    print("\n  Processing Human texts...")
    for text_id, data in tqdm(explanations_human.items(), desc="Human texts"):
        text_content = read_text_file(
            f"{TEXT_DATA_DIR}SocialMedia_Reddit/{text_id}.txt"
        )

        if text_content is None:
            continue

        text_ids_to_texts[text_id] = text_content
        explanations = data.get('explanations', [])
        judgment = judge_explanations(llm, sampling_params, text_id, text_content, explanations)
        judge_results_human[text_id] = judgment

        # Cleanup
        gc.collect()

    # ==========================================
    # 4. Create and save preference pairs
    # ==========================================
    print("\n[4/4] Creating preference pairs...")

    preference_ai = create_preference_pairs(judge_results_ai, explanations_ai, text_ids_to_texts)
    preference_human = create_preference_pairs(judge_results_human, explanations_human, text_ids_to_texts)

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

if __name__ == "__main__":
    main()
