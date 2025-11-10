import numpy as np

def get_system_prompt(num_frames: int) -> str:
    SYSTEM_PROMPT = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. You MUST choose frames from 0 to {num_frames - 1}.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time. Convert hours to minutes if needed (e.g., for 1 hour, 2 minutes, and 30 seconds, use 62:30).
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image. The frame index must be an integer that appears in previous conversations. You MUST choose frames from 0 to {num_frames - 1}.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident."""

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
        {"role": "system", "content": get_system_prompt(num_frames=total_frames)},
        {"role": "user", "content": question + "\n" + image_placeholders},
    ]

    return messages

