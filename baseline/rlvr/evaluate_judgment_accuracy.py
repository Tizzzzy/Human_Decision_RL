"""
Evaluate judgment accuracy of the trained model on the test set.

After GRPO training completes, use this script to measure:
- Accuracy: % of judgments that match ground truth
- Precision/Recall by class
- Sample outputs for inspection

Usage:
  conda activate ppo_2
  python evaluate_judgment_accuracy.py --checkpoint-path checkpoints/final_adapter
"""

import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    from . import config, dataset, prompt_templates, reward
except ImportError:
    import config
    import dataset
    import prompt_templates
    import reward


def load_trained_model(checkpoint_path, tokenizer):
    """
    Load the trained policy model with LoRA adapter.

    Args:
        checkpoint_path (str): Path to final_adapter directory
        tokenizer: HF tokenizer

    Returns:
        model: Model with LoRA adapter loaded
    """
    print(f"[Loading base model from {config.POLICY_MODEL_NAME}]")
    model = AutoModelForCausalLM.from_pretrained(
        config.POLICY_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=config.POLICY_CACHE_DIR,
        trust_remote_code=True,
    )

    print(f"[Loading LoRA adapter from {checkpoint_path}]")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    return model


def generate_judgment(model, tokenizer, prompt, max_new_tokens=256):
    """
    Generate explanation + judgment for a given prompt.

    Args:
        model: Model with LoRA adapter
        tokenizer: HF tokenizer
        prompt (str): Formatted prompt
        max_new_tokens (int): Max generation length

    Returns:
        str: Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=config.TEMPERATURE,
            top_p=1.0,
            do_sample=True,
            repetition_penalty=1.1,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (remove prompt)
    prompt_length = len(tokenizer.encode(prompt))
    generated_only = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    return generated_only.strip()


def evaluate_dataset(model, tokenizer, test_dataset, max_samples=None):
    """
    Evaluate model judgment accuracy on test dataset.

    Args:
        model: Model with LoRA adapter
        tokenizer: HF tokenizer
        test_dataset: HF Dataset with test samples
        max_samples (int): Limit evaluation to N samples (None = all)

    Returns:
        dict: Evaluation results
    """
    predictions = []
    ground_truths = []
    accuracies_by_class = {"AI": [], "Human": []}
    sample_outputs = []

    # Limit dataset if requested
    if max_samples is not None:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

    print(f"\nEvaluating on {len(test_dataset)} test samples...")

    for idx, sample in enumerate(tqdm(test_dataset)):
        prompt = sample["prompt"]
        true_label = sample["label_truth"]

        try:
            # Generate judgment
            generated = generate_judgment(model, tokenizer, prompt)

            # Extract judgment
            judgment = prompt_templates.parse_generation(generated)[1]

            if judgment is None:
                judgment = "PARSE_ERROR"

            predictions.append(judgment)
            ground_truths.append(true_label)

            # Store sample outputs for inspection
            if idx < 5:  # First 5 samples
                sample_outputs.append({
                    "text_id": sample["text_id"],
                    "true_label": true_label,
                    "predicted_judgment": judgment,
                    "generation": generated[:200] + ("..." if len(generated) > 200 else ""),
                })

        except Exception as e:
            print(f"Error at sample {idx}: {e}")
            predictions.append("ERROR")
            ground_truths.append(true_label)

    # Compute metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    # Overall accuracy
    valid_predictions = [p for p in predictions if p not in ["PARSE_ERROR", "ERROR"]]
    valid_ground_truths = [g for p, g in zip(predictions, ground_truths) if p not in ["PARSE_ERROR", "ERROR"]]

    if valid_predictions:
        overall_accuracy = accuracy_score(valid_ground_truths, valid_predictions)
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({len(valid_predictions)}/{len(predictions)} valid)")

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_ground_truths, valid_predictions, labels=["AI", "Human"], zero_division=0
        )

        print(f"\nPer-Class Metrics:")
        print(f"  AI:    Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f} (n={support[0]})")
        print(f"  Human: Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f} (n={support[1]})")

        # Confusion matrix
        cm = confusion_matrix(valid_ground_truths, valid_predictions, labels=["AI", "Human"])
        print(f"\nConfusion Matrix:")
        print(f"           Pred_AI  Pred_Human")
        print(f"True_AI    {cm[0, 0]:5d}    {cm[0, 1]:5d}")
        print(f"True_Human {cm[1, 0]:5d}    {cm[1, 1]:5d}")

        # Errors
        parse_errors = sum(1 for p in predictions if p == "PARSE_ERROR")
        runtime_errors = sum(1 for p in predictions if p == "ERROR")
        if parse_errors > 0:
            print(f"\nParsing Errors: {parse_errors} ({100*parse_errors/len(predictions):.1f}%)")
        if runtime_errors > 0:
            print(f"Runtime Errors: {runtime_errors} ({100*runtime_errors/len(predictions):.1f}%)")

    else:
        print("No valid predictions!")

    # Sample outputs
    print(f"\n" + "="*70)
    print("SAMPLE OUTPUTS (first 5)")
    print("="*70)
    for i, sample in enumerate(sample_outputs):
        print(f"\nSample {i+1}:")
        print(f"  Text ID: {sample['text_id']}")
        print(f"  True Label: {sample['true_label']}")
        print(f"  Predicted: {sample['predicted_judgment']}")
        print(f"  Generation: {sample['generation']}")

    return {
        "overall_accuracy": overall_accuracy if valid_predictions else 0,
        "num_valid_predictions": len(valid_predictions),
        "num_parse_errors": parse_errors if valid_predictions else 0,
        "num_runtime_errors": runtime_errors if valid_predictions else 0,
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model judgment accuracy")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/final_adapter",
        help="Path to trained LoRA adapter"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (default: all)"
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("EVALUATE JUDGMENT ACCURACY")
    print("="*70 + "\n")

    # Load tokenizer
    print("[Loading tokenizer...]")
    tokenizer = AutoTokenizer.from_pretrained(
        config.POLICY_MODEL_NAME,
        cache_dir=config.POLICY_CACHE_DIR,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load trained model
    print("[Loading trained model...]")
    model = load_trained_model(args.checkpoint_path, tokenizer)

    # Load test dataset
    print("[Loading test dataset...]")
    _, test_dataset = dataset.get_train_test_datasets(tokenizer)

    # Evaluate
    results = evaluate_dataset(model, tokenizer, test_dataset, max_samples=args.max_samples)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
