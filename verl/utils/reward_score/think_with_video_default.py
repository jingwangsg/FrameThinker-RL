import re


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
):
    """
    Simplified reward function for default template (no tool calls).
    Only checks:
    1. Format: <think>...</think><answer>...</answer>
    2. Correctness: final answer matches ground truth

    Returns:
        total_score: acc_score (0.0 or 1.0)
        acc_score: 1.0 if answer is correct, 0.0 otherwise
        format_score: 1.0 if format is valid, 0.0 otherwise
        other_score: always 0.0 (reserved for future use)
    """
    format_score = 0.0
    acc_score = 0.0
    other_score = 0.0
    total_score = 0.0

    try:
        # Extract <think> and <answer> blocks
        think_contents = re.findall(r"<think>(.*?)</think>", predict_str, re.DOTALL)
        answer_contents = re.findall(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    except Exception as e:
        return total_score, acc_score, format_score, other_score

    # Check format: must have at least one <think> block and exactly one <answer> block
    if not think_contents or len(answer_contents) != 1:
        return total_score, acc_score, format_score, other_score

    # Check that <think> content is not empty
    if not think_contents[0].strip():
        return total_score, acc_score, format_score, other_score

    # Format is valid
    format_score = 1.0

    # Extract the final answer
    model_answer = answer_contents[0].strip()

    # Check if answer matches ground truth
    if model_answer == ground_truth:
        acc_score = 1.0

    total_score = acc_score

    return total_score, acc_score, format_score, other_score
