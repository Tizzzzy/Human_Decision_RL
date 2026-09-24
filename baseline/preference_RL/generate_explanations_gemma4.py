"""
Generate explanations for social media texts using Gemma-4-E2B-it.

For each text in SocialMedia_Reddit and SocialMedia_rewrite directories:
1. Load existing 3 explanations (if available)
2. Generate 3 new explanations using Gemma-4-E2B-it
3. Store all 6 explanations in a JSON database

Output: explanations_gemma4.json with structure:
{
  "text_id": {
    "text_source": "reddit" or "rewrite",
    "explanations": [
      {"source": "reddit_explanation", "text": "..."},
      {"source": "reddit_explanation_claude", "text": "..."},
      {"source": "reddit_explanation_gemini", "text": "..."},
      {"source": "gemma4_1", "text": "..."},
      {"source": "gemma4_2", "text": "..."},
      {"source": "gemma4_3", "text": "..."}
    ]
  }
}
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForMultimodalLM
import gc
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
TEXT_DATA_DIR = "/projects/p32143/RL_human_decision/text_data/social/"
CACHE_DIR = "/projects/p32143/cache/huggingface"
MODEL_NAME = "google/gemma-4-E2B-it"
OUTPUT_JSON = "explanations_gemma4_human.json"

# Explanation source directories for each text source
EXPLANATION_SOURCES = {
    "reddit": [
        "SocialMedia_Reddit_explanation",
        "SocialMedia_Reddit_explanation_claude",
        "SocialMedia_Reddit_explanation_gemini"
    ],
    "rewrite": [
        "SocialMedia_rewrite_explanation",
        "SocialMedia_rewrite_explanation_claude",
        "SocialMedia_rewrite_explanation_gemini"
    ]
}

# ==========================================
# Prompting & Generation Config
# ==========================================
EXPLANATION_PROMPT_TEMPLATE = """Task: Analyze the provided social media post for linguistic markers of AI or human authorship.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list.
2. Do NOT provide an introduction, a final verdict, or a summary.
3. Do NOT group the bullet points into sub-headings (keep it a flat list).
4. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category or type of phrasing]."
    - "Look for [linguistic category or type of phrasing] like '[short example]'."
5. When providing examples, you may list multiple words separated by commas (e.g., like "word 1", "word 2").
6. Do NOT use bolding, italics, or any special Markdown formatting.
7. Limit the output to exactly 2 to 6 bullet points.

Post:
```
{text}
```

Explanation:"""

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
}

# ==========================================
# Helper Functions
# ==========================================
def read_text_file(filepath):
    """Read text from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def find_all_texts(base_dir):
    """Find all text files in Reddit and rewrite directories."""
    texts = {}

    for source in ["reddit"]:
        dir_name = "SocialMedia_Reddit" if source == "reddit" else "SocialMedia_rewrite"
        text_dir = os.path.join(base_dir, dir_name)

        if not os.path.exists(text_dir):
            print(f"⚠️  Directory not found: {text_dir}")
            continue

        text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
        print(f"Found {len(text_files)} files in {dir_name}")

        for filename in text_files:
            text_id = filename.replace('.txt', '')
            filepath = os.path.join(text_dir, filename)
            text_content = read_text_file(filepath)

            if text_content:
                texts[text_id] = {
                    'source': source,
                    'text': text_content
                }

    print(f"Total texts loaded: {len(texts)}")
    return texts

def load_existing_explanations(base_dir, text_id, text_source):
    """Load existing 3 explanations for a text."""
    explanations = []
    sources = EXPLANATION_SOURCES[text_source]

    for source_dir in sources:
        exp_file = os.path.join(base_dir, source_dir, f"{text_id}.txt")
        exp_text = read_text_file(exp_file)

        if exp_text:
            explanations.append({
                'source': source_dir,
                'text': exp_text
            })
        else:
            # Missing explanation file is OK, but log it
            pass

    return explanations

def generate_explanations_batch(processor, model, texts_batch, num_explanations=3):
    """
    Generate explanations for a batch of texts using Gemma-4-E2B-it.

    Args:
        processor: AutoProcessor for the model
        model: AutoModelForMultimodalLM
        texts_batch: list of (text_id, text_content) tuples
        num_explanations: how many explanations to generate per text

    Returns:
        dict mapping text_id -> list of generated explanations
    """
    results = {}

    for text_id, text_content in tqdm(texts_batch, desc="Generating explanations"):
        system_prompt = "You are a helpful assistant that analyzes text for linguistic markers."
        user_prompt = EXPLANATION_PROMPT_TEMPLATE.format(text=text_content)

        try:
            # Generate multiple explanations (with different random seeds)
            generated_explanations = []
            for i in range(num_explanations):
                # Format messages for chat template
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Process input using chat template
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                # Generate output
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **GENERATION_CONFIG
                    )

                # Decode response
                response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
                explanation = response.strip()

                generated_explanations.append({
                    'source': f'gemma4_{i+1}',
                    'text': explanation
                })

            results[text_id] = generated_explanations

        except Exception as e:
            print(f"⚠️  Error generating for {text_id}: {e}")
            results[text_id] = []

    return results

def main():
    print("="*80)
    print("Generating Explanations with Gemma-4-E2B-it")
    print("="*80)

    # ==========================================
    # 1. Load all texts
    # ==========================================
    print("\n[1/4] Loading all texts...")
    texts = find_all_texts(TEXT_DATA_DIR)

    if not texts:
        print("❌ No texts found. Exiting.")
        return

    # ==========================================
    # 2. Initialize Gemma model & processor
    # ==========================================
    print("\n[2/4] Loading Gemma-4-E2B-it model...")
    try:
        print(f"Loading processor for {MODEL_NAME}...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

        print(f"Loading model for {MODEL_NAME}...")
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_NAME,
            dtype="auto",
            device_map="auto",
            cache_dir=CACHE_DIR
        )
        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 3. Load existing explanations & generate new ones
    # ==========================================
    print("\n[3/4] Loading existing explanations and generating new ones...")

    all_data = {}
    batch_size = 8  # Process texts in batches

    text_items = list(texts.items())
    num_processed = 0

    for i in range(0, len(text_items), batch_size):
        batch_items = text_items[i:i+batch_size]
        text_ids_batch = [text_id for text_id, _ in batch_items]
        texts_batch = [(text_id, texts[text_id]['text']) for text_id, _ in batch_items]

        print(f"\n  Processing batch {i//batch_size + 1}/{(len(text_items)-1)//batch_size + 1}...")

        # Generate new explanations for this batch
        generated = generate_explanations_batch(processor, model, texts_batch, num_explanations=3)

        # Combine with existing explanations
        for text_id in text_ids_batch:
            text_source = texts[text_id]['source']

            # Load existing explanations
            existing = load_existing_explanations(TEXT_DATA_DIR, text_id, text_source)

            # Combine with generated
            all_explanations = existing + generated.get(text_id, [])

            all_data[text_id] = {
                'text_source': text_source,
                'explanations': all_explanations
            }

            num_processed += 1

        # Cleanup to save memory
        del generated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================
    # 4. Save to JSON
    # ==========================================
    print(f"\n[4/4] Saving results...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {num_processed} texts with explanations to {OUTPUT_JSON}")

    # Summary
    total_explanations = sum(len(item['explanations']) for item in all_data.values())
    avg_per_text = total_explanations / num_processed if num_processed else 0

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total texts processed: {num_processed}")
    print(f"Total explanations: {total_explanations}")
    print(f"Average explanations per text: {avg_per_text:.1f}")

    # Distribution by source
    source_counts = {}
    for item in all_data.values():
        for exp in item['explanations']:
            source = exp['source']
            source_counts[source] = source_counts.get(source, 0) + 1

    print("\nExplanations by source:")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        print(f"  {source}: {count}")

if __name__ == "__main__":
    main()
