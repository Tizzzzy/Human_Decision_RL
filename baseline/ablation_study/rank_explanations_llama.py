"""
Use Llama-3.1-8B-Instruct as judge to rank explanations and create preference pairs.

For each text:
1. Load 2 explanations from CSV
2. Judge selects the best explanation using Llama-3.1-8B-Instruct
3. Create preference pair: (chosen, [rejected])
4. Save to preference_pairs_llama.json

Output format:
{
  "text_id_human": {
    "text": "the original text",
    "label": "Human",
    "chosen_explanation": "the best explanation",
    "chosen_source": "source of best explanation",
    "judge_response": "model's ranking message",
    "rejected_explanations": [
      {"source": "...", "text": "..."},
      ...
    ]
  },
  "text_id_ai": {
    "text": "the original text",
    "label": "AI",
    "chosen_explanation": "the best explanation",
    "chosen_source": "source of best explanation",
    "judge_response": "model's ranking message",
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
from collections import defaultdict

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
CSV_FILE = "/projects/p32143/RL_human_decision/text_data/social/master_stimuli_database_socialmedia.csv"
OUTPUT_PREFERENCE = "preference_pairs_llama.json"

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.5,
    "top_p": 0.95,
    "do_sample": True,
}

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


def judge_explanations(tokenizer, model, text_id, text_content, explanations):
    """
    Use Llama-3.1-8B-Instruct judge to select the best explanation.

    Args:
        tokenizer: AutoTokenizer instance
        model: AutoModelForCausalLM instance
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
        text=text_content,
        exp_1=exp_texts[0],
        exp_2=exp_texts[1],
    )

    try:
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Apply chat template and get inputs ready for the model
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG
            )

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
            # Fallback: pick the first
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
    print("Ranking Explanations with Llama-3.1-8B-Instruct Judge")
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
    # 2. Load Llama-3.1-8B-Instruct judge
    # ==========================================
    print("\n[2/4] Loading Llama-3.1-8B-Instruct judge...")
    try:
        print("🚀 Initializing Llama model (this may take a moment)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
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
    for composite_key, data in tqdm(explanations_data.items(), desc="Texts"):
        text_content = text_ids_to_texts.get(composite_key)

        if text_content is None:
            continue

        explanations = data.get('explanations', [])
        judgment = judge_explanations(tokenizer, model, composite_key, text_content, explanations)
        judge_results[composite_key] = judgment

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

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
