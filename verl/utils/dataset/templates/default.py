import numpy as np


def get_system_prompt() -> str:
    SYSTEM_PROMPT = """You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by the final answer (OPTION only) within an <action> </action> tag."""
    return SYSTEM_PROMPT


def get_image_placeholders(num_frames: int, total_frames: int) -> str:
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    return "\n".join([f"frame {idx}:<image>" for idx in frame_indices])


def apply_message_template(messages, **kwargs):
    assert (
        messages[0]["role"] != "system"
    ), "System message should not be applied to the message template"

    assert len(messages) == 1, "Only one message is allowed"

    total_frames = kwargs["extra_info"]["total_frames"]
    images = kwargs["images"]

    image_placeholders = get_image_placeholders(num_frames=len(images), total_frames=total_frames)

    question = messages[0]["content"]
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": question + "\n" + image_placeholders},
    ]

    return messages
