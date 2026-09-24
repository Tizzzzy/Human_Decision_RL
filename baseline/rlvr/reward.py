"""
Verifiable reward function for GRPO training.

Rewards are based on whether the model's judgment (AI/Human) matches ground truth.
- Correct judgment: +1.0
- Incorrect judgment: -5.0
- Parsing error (can't extract judgment): -5.0

This is a "verifiable" reward because correctness is binary and checkable
against ground truth labels from the database.
"""

try:
    from . import config, prompt_templates
except ImportError:
    import config
    import prompt_templates


def extract_judgment(completion_text: str) -> str:
    """
    Extract the judgment from model completion text.

    Args:
        completion_text (str): Raw model output

    Returns:
        str: "AI" or "Human" if successfully parsed, None otherwise
    """
    _, judgment_class = prompt_templates.parse_generation(completion_text)
    return judgment_class


def compute_reward(predicted_judgment: str, true_label: str) -> float:
    """
    Compute reward based on judgment correctness.

    Args:
        predicted_judgment (str): Model's predicted judgment ("AI", "Human", or None)
        true_label (str): Ground truth label ("AI" or "Human")

    Returns:
        float: Reward value (1.0 for correct, -5.0 for incorrect)
    """
    if predicted_judgment is None:
        # Failed to parse judgment
        return config.REWARD_INCORRECT

    # Normalize for comparison
    predicted = predicted_judgment.upper()
    true = true_label.upper()

    if predicted == true:
        return config.REWARD_CORRECT
    else:
        return config.REWARD_INCORRECT


def reward_func(
    prompts,
    completions,
    text_content=None,
    label_truth=None,
    label_truth_numeric=None,
    text_id=None,
    **kwargs
) -> list:
    """
    Reward function for GRPO training.

    Compatible with GRPOTrainer's reward_funcs interface.
    Processes each completion and returns a list of reward scores.

    Args:
        prompts: List of prompt strings (unused, for signature compatibility)
        completions: List of model completions (generated text)
        text_content: List of original text contents (unused)
        label_truth: List of ground truth labels ("AI" or "Human")
        label_truth_numeric: List of numeric labels (1 for AI, 0 for Human)
        text_id: List of text IDs (unused)
        **kwargs: Other fields from dataset (ignored)

    Returns:
        list[float]: Reward for each completion
    """
    rewards = []

    for i, completion in enumerate(completions):
        # Extract judgment from completion
        judgment = extract_judgment(completion)

        # Get ground truth
        true_label = label_truth[i] if label_truth is not None else None

        # Compute reward
        if true_label is None:
            # No ground truth available
            reward = config.REWARD_INCORRECT
        else:
            reward = compute_reward(judgment, true_label)

        rewards.append(reward)

    return rewards
