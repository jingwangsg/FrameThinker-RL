def get_system_prompt():
    SYSTEM_PROMPT = """Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."""
    return SYSTEM_PROMPT


def get_image_placeholders(num_frames: int) -> str:
    return f"<image>" * num_frames


def apply_message_template(messages, **kwargs):
    assert (
        messages[0]["role"] != "system"
    ), "System message should not be applied to the message template"

    assert len(messages) == 1, "Only one message is allowed"

    # total_frames = kwargs["extra_info"]["total_frames"]
    num_frames = len(kwargs["multi_modal_data"]["image"])
    image_placeholders = get_image_placeholders(num_frames=num_frames)

    question = messages[0]["content"]
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": question + "\n" + image_placeholders},
    ]
    return messages
