"""
Smoke tests for verifiable-reward RL pipeline.

Run these tests BEFORE submitting a full training job to catch errors early.

Tests:
1. Dataset loading and prompt building
2. Judgment parsing and reward computation
3. Small dataset sanity check
4. Model loading (requires GPU)

Usage:
  conda activate ppo_2
  python smoke_test.py
  # Or on GPU node:
  salloc --partition=gengpu --gres=gpu:a100:1 --time=01:00:00 --mem=100G
  python smoke_test.py
"""

import sys
import torch

try:
    from . import config, dataset, prompt_templates, reward
except ImportError:
    import config
    import dataset
    import prompt_templates
    import reward


def test_prompt_parsing():
    """Test judgment parsing from model outputs."""
    print("\n" + "="*70)
    print("TEST 1: Prompt Parsing and Judgment Extraction")
    print("="*70)

    # Test case 1: Valid AI judgment
    text1 = """Look for formal vocabulary and professional tone.
Look for lack of contractions like "don't" or "can't".
Look for repetitive phrasing patterns.
Look for consistent sentence structure like "The text shows" or "This indicates".

JUDGMENT: AI"""

    judgment1, parsed1 = prompt_templates.parse_generation(text1)
    print(f"Test 1 - AI judgment: {parsed1} (expected: AI) - {'✓' if parsed1 == 'AI' else '✗'}")

    # Test case 2: Valid Human judgment
    text2 = """Look for casual language and contractions like "don't", "can't", "it's".
Look for personal pronouns like "I", "we", "you".
Look for emotional expressions.

JUDGMENT: Human"""

    judgment2, parsed2 = prompt_templates.parse_generation(text2)
    print(f"Test 2 - Human judgment: {parsed2} (expected: Human) - {'✓' if parsed2 == 'Human' else '✗'}")

    # Test case 3: Invalid/missing judgment
    text3 = """Look for something.
Look for something else."""

    judgment3, parsed3 = prompt_templates.parse_generation(text3)
    print(f"Test 3 - Missing judgment: {parsed3} (expected: None) - {'✓' if parsed3 is None else '✗'}")

    print("✓ Judgment parsing tests completed")


def test_reward_computation():
    """Test reward function with hand-picked examples."""
    print("\n" + "="*70)
    print("TEST 2: Reward Computation")
    print("="*70)

    # Test correct predictions
    reward1 = reward.compute_reward("AI", "AI")
    print(f"Correct (AI==AI): {reward1} (expected: {config.REWARD_CORRECT}) - {'✓' if reward1 == config.REWARD_CORRECT else '✗'}")

    reward2 = reward.compute_reward("Human", "Human")
    print(f"Correct (Human==Human): {reward2} (expected: {config.REWARD_CORRECT}) - {'✓' if reward2 == config.REWARD_CORRECT else '✗'}")

    # Test incorrect predictions
    reward3 = reward.compute_reward("AI", "Human")
    print(f"Incorrect (AI!=Human): {reward3} (expected: {config.REWARD_INCORRECT}) - {'✓' if reward3 == config.REWARD_INCORRECT else '✗'}")

    reward4 = reward.compute_reward("Human", "AI")
    print(f"Incorrect (Human!=AI): {reward4} (expected: {config.REWARD_INCORRECT}) - {'✓' if reward4 == config.REWARD_INCORRECT else '✗'}")

    # Test None judgment (parsing failed)
    reward5 = reward.compute_reward(None, "AI")
    print(f"Parse failed (None): {reward5} (expected: {config.REWARD_INCORRECT}) - {'✓' if reward5 == config.REWARD_INCORRECT else '✗'}")

    print("✓ Reward computation tests completed")


def test_dataset_loading():
    """Test dataset loading from DB."""
    print("\n" + "="*70)
    print("TEST 3: Dataset Loading")
    print("="*70)

    try:
        records = dataset.load_distinct_texts_from_db()
        print(f"Loaded {len(records)} records from DB")

        if len(records) > 0:
            # Show sample
            sample = records[0]
            print(f"\nSample record:")
            print(f"  text_id: {sample['text_id']}")
            print(f"  label_truth: {sample['label_truth']}")
            print(f"  label_numeric: {sample['label_truth_numeric']}")
            print(f"  text length: {len(sample['text_content'])} chars")

            # Check label distribution
            ai_count = sum(1 for r in records if r['label_truth'] == 'AI')
            human_count = sum(1 for r in records if r['label_truth'] == 'Human')
            print(f"\nLabel distribution:")
            print(f"  AI: {ai_count}")
            print(f"  Human: {human_count}")

            print("✓ Dataset loading tests completed")
        else:
            print("✗ No records loaded from DB!")
            return False

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_prompt_building():
    """Test prompt building with actual tokenizer."""
    print("\n" + "="*70)
    print("TEST 4: Prompt Building with Tokenizer")
    print("="*70)

    try:
        from transformers import AutoTokenizer

        print("[Loading tokenizer...]")
        tokenizer = AutoTokenizer.from_pretrained(
            config.POLICY_MODEL_NAME,
            cache_dir=config.POLICY_CACHE_DIR,
            trust_remote_code=True,
        )

        # Test prompt building
        test_text = "This is a test text. I really enjoyed the movie!"
        prompt = prompt_templates.build_policy_prompt(test_text, tokenizer)

        print(f"Built prompt length: {len(prompt)} chars")
        print(f"First 200 chars:\n{prompt[:200]}...")
        print(f"\n✓ Prompt building tests completed")
        return True

    except Exception as e:
        print(f"✗ Error building prompt: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading (requires GPU)."""
    print("\n" + "="*70)
    print("TEST 5: Model Loading")
    print("="*70)

    try:
        from transformers import AutoModelForCausalLM
        import torch

        # Check GPU availability
        if not torch.cuda.is_available():
            print("⚠ No GPU available - skipping model loading test")
            print("  Run this test on a GPU node: salloc --partition=gengpu --gres=gpu:a100:1")
            return True

        print(f"GPU available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")

        print("\n[Loading policy model...]")
        model = AutoModelForCausalLM.from_pretrained(
            config.POLICY_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=config.POLICY_CACHE_DIR,
            trust_remote_code=True,
        )

        print(f"Model loaded: {model.num_parameters():,} parameters")

        # Test generation
        print("\n[Testing generation...]")
        tokenizer = AutoTokenizer.from_pretrained(
            config.POLICY_MODEL_NAME,
            cache_dir=config.POLICY_CACHE_DIR,
            trust_remote_code=True,
        )

        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Generated text: {generated}")
        print(f"✓ Model loading tests completed")
        return True

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("     SMOKE TESTS: Verifiable-Reward RL Pipeline")
    print("="*70)

    results = {
        "Prompt Parsing": True,
        "Reward Computation": True,
        "Dataset Loading": True,
        "Prompt Building": True,
        "Model Loading": True,
    }

    # Run tests
    try:
        test_prompt_parsing()
    except Exception as e:
        print(f"✗ Prompt parsing test failed: {e}")
        results["Prompt Parsing"] = False

    try:
        test_reward_computation()
    except Exception as e:
        print(f"✗ Reward computation test failed: {e}")
        results["Reward Computation"] = False

    try:
        if not test_dataset_loading():
            results["Dataset Loading"] = False
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        results["Dataset Loading"] = False

    try:
        if not test_prompt_building():
            results["Prompt Building"] = False
    except Exception as e:
        print(f"✗ Prompt building test failed: {e}")
        results["Prompt Building"] = False

    try:
        if not test_model_loading():
            results["Model Loading"] = False
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        results["Model Loading"] = False

    # Summary
    print("\n" + "="*70)
    print("SMOKE TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All smoke tests passed! Ready for training.")
    else:
        print("\n✗ Some tests failed. Fix these issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
