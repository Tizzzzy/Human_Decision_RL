"""
Prompt templates for verifiable-reward RL.

The model must generate:
1. Explanations (2-6 bullet points of linguistic markers)
2. A clear judgment: "JUDGMENT: AI" or "JUDGMENT: Human"
"""

# ========================================================================
# Policy Instruction
# ========================================================================
POLICY_INSTRUCTION = """Task: Analyze the provided social media post for linguistic markers of AI or human authorship.

Constraints:
1. Start the response IMMEDIATELY with the bulleted list of linguistic markers.
2. Do NOT provide an introduction, a final verdict summary, or other preamble.
3. Do NOT group the bullet points into sub-headings (keep it a flat list).
4. Every bullet point MUST follow one of these two exact templates:
    - "Look for [linguistic category or type of phrasing]."
    - "Look for [linguistic category or type of phrasing] like '[short example]'."
5. When providing examples, you may list multiple words/phrases separated by commas.
6. Do NOT use bolding, italics, or any special Markdown formatting.
7. Limit the explanations to exactly 2 to 6 bullet points.
8. After the bullet points, end with exactly this line: "JUDGMENT: AI" or "JUDGMENT: Human" (no extra text after this).

Post:
```
{text}
```

Explanation:
"""


def build_policy_prompt(text, tokenizer):
    """
    Build a policy prompt for generating explanation + judgment.

    Args:
        text (str): The input text to analyze
        tokenizer: HF tokenizer with apply_chat_template

    Returns:
        str: Formatted prompt ready for generation
    """
    instruction = POLICY_INSTRUCTION.format(text=text)

    messages = [
        {"role": "user", "content": instruction}
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        # Fallback for some versions of the transformers library
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False}
        )

    return prompt


def parse_generation(generated_text):
    """
    Parse the LLM's generation to extract judgment.

    Expected format:
    - Bullet points (ignored)
    - Last line: "JUDGMENT: AI" or "JUDGMENT: Human"

    Args:
        generated_text (str): Raw model output

    Returns:
        tuple: (judgment_text, judgment_class)
            - judgment_text: str, the full judgment line
            - judgment_class: "AI" or "Human" or None if parsing failed
    """
    lines = generated_text.strip().split('\n')

    # Find the JUDGMENT line (should be last)
    judgment_line = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("JUDGMENT:"):
            judgment_line = line
            break

    if judgment_line is None:
        return None, None

    # Parse judgment: "JUDGMENT: AI" or "JUDGMENT: Human"
    parts = judgment_line.split(":")
    if len(parts) < 2:
        return judgment_line, None

    judgment_class = parts[1].strip().upper()

    # Normalize
    if judgment_class in ["AI"]:
        return judgment_line, "AI"
    elif judgment_class in ["HUMAN"]:
        return judgment_line, "Human"
    else:
        # Try to match partial strings
        if "AI" in judgment_class:
            return judgment_line, "AI"
        elif "HUMAN" in judgment_class:
            return judgment_line, "Human"

    return judgment_line, None
