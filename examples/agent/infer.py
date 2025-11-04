import os
import cv2
import json
import re
import numpy as np
import shutil
import torch
import sglang as sgl
import argparse
from collections import defaultdict
from itertools import combinations
import decord

CONFIG = {
    # "MODEL_PATH": "/mnt/aws-lfs-01/shared/checkpoints/jingwang/video_reason/sft/qwen2_5vl_7b_sft_full_framethinker_base/",
    "MODEL_PATH": "Qwen/Qwen2.5-VL-7B-Instruct",
    "BASE_VIDEO_DIR": "/mnt/aws-lfs-01/shared/datasets/s3:/video_reason/",
    "BASE_FRAME_DIR_ROOT": "/tmp/video_frames",
    "TARGET_JSON_PATH": "/mnt/aws-lfs-01/shared/datasets/s3:/video_reason/Video-Holmes/test.json",
    "MAX_ITERATIONS": 5,
    "MAX_RETRIES": 3,
    "NUM_FRAMES_TO_SAMPLE": 8,
    "NUM_FRAMES_TO_SAMPLE_LONG": 12,
    "DEFAULT_GPUS": 8,
    "MAX_FRAME_WIDTH": 640,
    "MAX_FRAME_HEIGHT": 360,
    "MAX_FRAME_WIDTH_LONG": 448,
    "MAX_FRAME_HEIGHT_LONG": 252
}


def load_sglang_engine(model_path, tp_size=1):
    """Load SGLang engine for vision-language model inference"""
    # Import sglang here to ensure CUDA is initialized first

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        trust_remote_code=True,
        chat_template="qwen2-vl",
        mm_attention_backend="sdpa",
    )
    return engine


def convert_messages_to_sglang_format(messages):
    """
    Convert transformers-style messages to SGLang format.

    Returns:
        prompt: Formatted prompt string with <image> tokens
        image_paths: List of image file paths
    """
    from sglang.srt.parser.conversation import chat_templates

    chat_template = chat_templates["qwen2-vl"].copy()
    image_token = chat_template.image_token
    image_paths = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            chat_template.set_system_message(content)
        elif role == "user":
            if isinstance(content, str):
                # Simple text content
                chat_template.append_message(chat_template.roles[0], content)
            elif isinstance(content, list):
                # Mixed content with text and images
                user_text_parts = []
                for item in content:
                    if item["type"] == "text":
                        user_text_parts.append(item["text"])
                    elif item["type"] == "image":
                        user_text_parts.append(image_token)
                        image_paths.append(item["image"])

                user_message = "\n".join(user_text_parts)
                chat_template.append_message(chat_template.roles[0], user_message)
        elif role == "assistant":
            chat_template.append_message(chat_template.roles[1], content)

    # Add empty assistant message for generation
    chat_template.append_message(chat_template.roles[1], None)

    prompt = chat_template.get_prompt()
    return prompt, image_paths


def run_inference(engine, messages):
    """Run inference using SGLang engine"""
    prompt, image_paths = convert_messages_to_sglang_format(messages)

    # Generate response using SGLang
    outputs = engine.generate(
        prompt=prompt,
        image_data=image_paths if image_paths else None,
        sampling_params={"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 2048}
    )

    # Extract text from outputs
    if isinstance(outputs, list):
        output_text = outputs[0]["text"] if outputs else ""
    else:
        output_text = outputs.get("text", str(outputs))

    return output_text


def parse_model_response(response):
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    return (think_match.group(1).strip(), action_match.group(1).strip()) if think_match and action_match else (
    None, None)


def get_video_metadata(video_path):
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        frame_count = len(vr)
        del vr
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps, frame_count
    except Exception:
        return 0, 0


def scale_down_preserving_aspect_ratio(image, max_width=640, max_height=360):
    h, w = image.shape[:2]

    if w <= max_width and h <= max_height:
        return image

    ratio_w = max_width / w
    ratio_h = max_height / h
    scale_ratio = min(ratio_w, ratio_h)

    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized_image

def extract_frames(video_path, frame_indices, output_dir, max_width, max_height):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        frames_array = vr.get_batch(frame_indices).asnumpy()
        for i, frame_idx in enumerate(frame_indices):
            frame_img_bgr = cv2.cvtColor(frames_array[i], cv2.COLOR_RGB2BGR)
            scaled_frame = scale_down_preserving_aspect_ratio(
                frame_img_bgr,
                max_width=max_width,
                max_height=max_height
            )
            output_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(output_path, scaled_frame)
            saved_paths.append(output_path)
        del vr
    except Exception as e:
        print(f"!!!!!! AN EXCEPTION OCCURRED in extract_frames !!!!!!")
        print(f"Video Path: {video_path}")
        print(f"Frame Indices: {frame_indices}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
    return saved_paths


def handle_get_frame_number(action_content, fps):
    time_match = re.match(r'get frame number at time\s+(\S+)', action_content.strip())
    if not time_match: return None
    try:
        minutes, seconds = map(int, time_match.group(1).split(':'))
        return f"Frame number at time {time_match.group(1)} is: {int((minutes * 60 + seconds) * fps)}."
    except ValueError:
        return None

def handle_choose_frames(action_content, video_path, video_id, iteration, total_frames, base_frame_dir,
                         num_frames_to_sample, max_width, max_height):
    match = re.search(r"choose frames (.*)\s+(\d+)\s+and\s+(\d+)", action_content)
    if not match: return None
    try:
        start_frame, end_frame = map(int, match.groups())
    except ValueError:
        return None
    if start_frame >= end_frame or end_frame >= total_frames or (end_frame - start_frame) <= num_frames_to_sample:
        return None
    frame_indices = np.linspace(start_frame + 1, end_frame - 1, num_frames_to_sample, dtype=int).tolist()
    output_dir = os.path.join(base_frame_dir, f"{video_id}_iter_{iteration}")
    image_paths = extract_frames(video_path, frame_indices, output_dir, max_width, max_height)
    user_content = []
    for path, index in zip(image_paths, frame_indices):
        user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])
    return user_content

def parse_frame_number(filepath: str) -> int:
    match = re.search(r'_(\d+)\.jpg$', filepath)
    if match:
        return int(match.group(1))
    return -1

def count_choose_frames_actions(conversation_history: list) -> int:
    """Count how many 'choose frames between' actions appear in assistant messages."""
    count = 0
    for turn in conversation_history:
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            if isinstance(content, str):
                # Find all <action> tags
                action_matches = re.findall(r'<action>(.*?)</action>', content, re.DOTALL)
                for action in action_matches:
                    if action.strip().startswith("choose frames between"):
                        count += 1
    return count

def stringify_conversation(conversation_history: dict):
    print(conversation_history)
    image_paths = conversation_history[-1].get("images", [])
    image_groups = [image_paths[i:i + CONFIG["NUM_FRAMES_TO_SAMPLE"]] for i in
                    range(0, len(image_paths), CONFIG["NUM_FRAMES_TO_SAMPLE"])]
    frame_intervals = []
    for group in image_groups:
        if not group:
            continue
        start_frame_num = parse_frame_number(group[0])
        end_frame_num = parse_frame_number(group[-1])
        if start_frame_num != -1 and end_frame_num != -1:
            frame_intervals.append((min(start_frame_num, end_frame_num), max(start_frame_num, end_frame_num)))

    if len(frame_intervals) > 1:
        for interval1, interval2 in combinations(frame_intervals, 2):
            s1, e1 = interval1
            s2, e2 = interval2
            if s1 == s2 and e1 == e2:
                return "error"
            if abs(s1 - s2) <= 1 and abs(e1 - e2) <= 1:
                return "error"
    stringified_content = "".join([
        turn['content'] if isinstance(turn.get('content'), str) else "\n[New frames provided.]\n"
        for turn in conversation_history[2:]
    ])

    return stringified_content

def validate_reasoning_process(predict_str: str, num_frames_to_sample: int):
    try:
        think_contents = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        action_contents = re.findall(r'<action>(.*?)</action>', predict_str, re.DOTALL)
        system_responses = re.findall(r'</action>(.*?)(?:<think>|$)', predict_str, re.DOTALL)
    except Exception:
        return False, 0, 0

    if not think_contents or not action_contents or len(think_contents) != len(action_contents):
        return False, 0, 0
    if not action_contents[-1].strip().startswith('output answer:'):
        return False, 0, 0

    tool_call_count = len(action_contents) - 1
    image_add_count = 0

    action_frame_pairs, requested_times = [], []
    expected_frame_in_next_action = None

    for i, action in enumerate(action_contents[:-1]):
        action = action.strip()

        if expected_frame_in_next_action is not None:
            frame_match_check = re.match(r'choose frames between (\d+) and (\d+)', action)
            if not frame_match_check: return False, 0, 0
            start_f, end_f = map(int, frame_match_check.groups())
            if not (start_f <= expected_frame_in_next_action <= end_f): return False, 0, 0
            expected_frame_in_next_action = None

        frame_match = re.match(r'choose frames between (\d+) and (\d+)', action)
        if frame_match:
            image_add_count += 1
            num1, num2 = map(int, frame_match.groups())
            current_pair = (num1, num2)
            if current_pair in action_frame_pairs: return False, 0, 0
            action_frame_pairs.append(current_pair)
            if num1 >= num2 - num_frames_to_sample: return False, 0, 0
            continue

        time_match = re.match(r'get frame number at time\s+(\S+)', action)
        if time_match:
            time_str_action = time_match.group(1)
            if time_str_action in requested_times: return False, 0, 0
            requested_times.append(time_str_action)
            if i >= len(system_responses): return False, 0, 0
            response_match = re.search(r'is:\s*(\d+)', system_responses[i].strip())
            if response_match:
                expected_frame_in_next_action = int(response_match.group(1))
                continue
            return False, 0, 0

        return False, 0, 0

    if expected_frame_in_next_action is not None: return False, 0, 0

    return True, tool_call_count, image_add_count


def process_single_problem(qa_item, engine, base_frame_dir_root, rank=0):
    # Extract Video-Holmes format data
    video_rel_path = qa_item['video_path']  # e.g., "Video-Holmes/videos/xxx.mp4"
    video_path = os.path.join(CONFIG["BASE_VIDEO_DIR"], video_rel_path)
    video_id = qa_item['metadata']['video_id']
    correct_answer = qa_item['metadata']['answer_key']  # Option letter (A, B, C, etc.)

    # Format question with options
    question = qa_item['question']
    options = qa_item['metadata']['options']
    options_text = '\n'.join([f"{key}. {value}" for key, value in sorted(options.items())])
    question = f"{question}\n{options_text}"

    fps, frame_count = get_video_metadata(video_path)
    if frame_count == 0 or fps == 0: return {"status": "format_error"}

    duration_in_seconds = frame_count / fps
    if duration_in_seconds > 300:
        num_frames_to_sample = CONFIG["NUM_FRAMES_TO_SAMPLE_LONG"]
        max_frame_width = CONFIG["MAX_FRAME_WIDTH_LONG"]
        max_frame_height = CONFIG["MAX_FRAME_HEIGHT_LONG"]
    else:
        num_frames_to_sample = CONFIG["NUM_FRAMES_TO_SAMPLE"]
        max_frame_width = CONFIG["MAX_FRAME_WIDTH"]
        max_frame_height = CONFIG["MAX_FRAME_HEIGHT"]

    system_prompt = f"You are an expert AI assistant that answers questions about a video by iteratively analyzing it.\nYour task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.\nPossible actions are:\n1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. The number of frames is fixed, currently {num_frames_to_sample}.\n2. `get frame number at time MM:SS`: Get the exact frame number for a specific time. Convert hours to minutes if needed (e.g., for 1 hour, 2 minutes, and 30 seconds, use 62:30).\n3. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident."

    initial_prompt_text = f"{question}\n"
    base_frame_dir = os.path.join(base_frame_dir_root, f"gpu_{rank}")

    for times in range(CONFIG["MAX_RETRIES"]):
        if times > 1:
            system_prompt = f"You are an expert AI assistant that answers questions about a video.\nYour task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag. Your only action is: `output answer: OPTION`: Provide the final answer (e.g., A, B, C...)."

        if os.path.exists(base_frame_dir): shutil.rmtree(base_frame_dir)
        os.makedirs(base_frame_dir)

        initial_indices = np.linspace(0, frame_count - 1, num_frames_to_sample, dtype=int).tolist()
        initial_frames_dir = os.path.join(base_frame_dir, f"{video_id}")
        initial_image_paths = extract_frames(video_path, initial_indices, initial_frames_dir, max_frame_width,
                                             max_frame_height)
        if not initial_image_paths:
            print(f"Warning: Initial frame extraction failed for {video_id}. Retrying...")
            continue
        initial_user_content = [{"type": "text", "text": initial_prompt_text}]
        for path, index in zip(initial_image_paths, initial_indices):
            initial_user_content.extend([{"type": "text", "text": f"frame {index}:"}, {"type": "image", "image": path}])

        conversation_history = [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": initial_user_content}]

        final_answer = None
        for i in range(CONFIG["MAX_ITERATIONS"]):
            model_response_str = run_inference(engine, conversation_history)
            conversation_history.append({"role": "assistant", "content": model_response_str})

            _, action = parse_model_response(model_response_str)
            if action is None: break

            if action.startswith("output answer:"):
                final_answer = action.replace("output answer:", "").strip()
                break

            feedback = handle_get_frame_number(action, fps) if "get frame" in action else \
                handle_choose_frames(action, video_path, video_id, i + 1, frame_count, base_frame_dir,
                                     num_frames_to_sample, max_frame_width, max_frame_height)

            if feedback is None: break
            conversation_history.append({"role": "user", "content": feedback})

        if final_answer is not None:
            full_str = stringify_conversation(conversation_history)
            is_valid, tool_calls, image_adds = validate_reasoning_process(full_str, num_frames_to_sample)
            if is_valid:
                images_used = num_frames_to_sample * (1 + image_adds)
                status = "correct" if final_answer == correct_answer else "wrong_answer"

                # Count choose frames actions and print results
                choose_frames_count = count_choose_frames_actions(conversation_history)
                answer_match = "correct" if final_answer == correct_answer else "wrong"
                print(f"[GPU {rank}] video_id: {video_id} | Answer: {answer_match} | Choose frames count: {choose_frames_count}")

                return {"status": status, "tool_calls": tool_calls, "images_used": images_used, "choose_frames_count": choose_frames_count}

    # Print format error case
    choose_frames_count = count_choose_frames_actions(conversation_history)
    print(f"[GPU {rank}] video_id: {video_id} | Answer: format_error | Choose frames count: {choose_frames_count}")
    return {"status": "format_error", "choose_frames_count": choose_frames_count}


def aggregate_and_print_results(all_results):
    total_problems = len(all_results)
    if total_problems == 0:
        print("No results to aggregate.")
        return

    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_wrong_answer = 0
    total_format_error = 0
    total_images_used = 0
    total_choose_frames_count = 0

    for res in all_results:
        status = res.get("status")
        if status == "correct":
            total_correct += 1
            tool_calls = res.get("tool_calls", 0)
            stats[tool_calls]['correct'] += 1
            stats[tool_calls]['total'] += 1
            total_images_used += res.get("images_used", 0)
            total_choose_frames_count += res.get("choose_frames_count", 0)
        elif status == "wrong_answer":
            total_wrong_answer += 1
            tool_calls = res.get("tool_calls", 0)
            stats[tool_calls]['total'] += 1
            total_images_used += res.get("images_used", 0)
            total_choose_frames_count += res.get("choose_frames_count", 0)
        else:
            total_format_error += 1
            total_choose_frames_count += res.get("choose_frames_count", 0)

    print("\n" + "=" * 50)
    print(" " * 15 + "AGGREGATED RESULTS")
    print("=" * 50)
    print(f"Total Problems Processed: {total_problems}\n")

    print("--- Overall Performance ---")
    accuracy = total_correct / total_problems if total_problems > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({total_correct}/{total_problems})")
    print(f"Correct Answers: {total_correct} ({total_correct / total_problems:.2%})")
    print(f"Wrong Answers (Valid Format): {total_wrong_answer} ({total_wrong_answer / total_problems:.2%})")
    print(f"Format Errors (After Retries): {total_format_error} ({total_format_error / total_problems:.2%})")
    print("-" * 25)

    valid_format_problems = total_correct + total_wrong_answer
    if valid_format_problems > 0:
        avg_images = total_images_used / valid_format_problems
        print(f"Average Images Used (per valid problem): {avg_images:.2f}")

        total_tool_calls_made = 0
        for i in stats.keys():
            total_tool_calls_made += i * stats[i]['total']

        avg_tool_calls = total_tool_calls_made / valid_format_problems
        print(f"Average Tool Calls Used (per valid problem): {avg_tool_calls:.2f}")

    # Print average choose frames count for all problems
    if total_problems > 0:
        avg_choose_frames = total_choose_frames_count / total_problems
        print(f"Average Choose Frames Count (per problem): {avg_choose_frames:.2f}")

    print("\n--- Tool Call Analysis (for valid format attempts) ---")
    sorted_tool_calls = sorted(stats.keys())
    for i in sorted_tool_calls:
        correct = stats[i]['correct']
        total = stats[i]['total']
        accuracy = correct / total if total > 0 else 0
        print(f"Problems with {i} tool calls: {total} | Accuracy: {accuracy:.2%}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Run single-GPU video QA evaluation with SGLang.")
    parser.add_argument("-m", "--model", type=str, default=CONFIG["MODEL_PATH"],
                        help=f"Model path. Default: {CONFIG['MODEL_PATH']}")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="Tensor parallelism size (default: 1)")
    args = parser.parse_args()
    model_path = args.model

    # Create unique temporary directory for this run (avoid conflicts with multiple concurrent runs)
    pid = os.getpid()
    base_frame_dir_root = f"/tmp/video_frames_pid_{pid}"
    print(f"Using temporary directory: {base_frame_dir_root}")

    # Set GPU device
    print(f"Using TP size: {args.tp_size}")
    print(f"Model path: {model_path}")

    # Load dataset
    try:
        with open(CONFIG["TARGET_JSON_PATH"], 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} problems from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize SGLang engine
    print("Initializing SGLang engine...")
    engine = load_sglang_engine(model_path, tp_size=args.tp_size)
    print("✓ SGLang engine loaded successfully")

    # Process each problem sequentially
    all_results = []
    for i, qa_item in enumerate(dataset):
        print(f"\nProcessing item {i + 1}/{len(dataset)}...")
        result = process_single_problem(qa_item, engine, base_frame_dir_root, rank=0)
        all_results.append(result)

    # Cleanup
    del engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Print aggregated results
    aggregate_and_print_results(all_results)

    # Clean up temporary directory
    if os.path.exists(base_frame_dir_root):
        print(f"\nCleaning up temporary directory: {base_frame_dir_root}")
        shutil.rmtree(base_frame_dir_root)
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()