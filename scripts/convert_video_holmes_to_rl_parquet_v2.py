#!/usr/bin/env python3
"""
Convert Video-Holmes JSON files to RL training parquet format.

This script creates complete RL-format parquet files from Video-Holmes JSON,
matching the official format from GitHub issue #4.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import io
import os

from torchcodec.decoders import VideoDecoder
import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
from einops import rearrange


def get_system_prompt(num_frames: int) -> str:
    SYSTEM_PROMPT = f"""You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
    Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
    Possible actions are:
    1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. You MUST choose frames from 0 to {num_frames - 1}.
    2. `get frame number at time MM:SS`: Get the exact frame number for a specific time. Convert hours to minutes if needed (e.g., for 1 hour, 2 minutes, and 30 seconds, use 62:30).
    3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image. The frame index must be an integer that appears in previous conversations. You MUST choose frames from 0 to {num_frames - 1}.
    4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident."""

    return SYSTEM_PROMPT


def get_video_metadata(video_path: str) -> Dict[str, any]:
    """
    Extract video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing fps, total_frames, width, height
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_frames,width,height",
        "-of",
        "json",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]

        # Parse frame rate (e.g., "24000/1001" -> 23.976)
        r_frame_rate = stream["r_frame_rate"]
        num, den = map(int, r_frame_rate.split("/"))
        fps = num / den

        return {
            "fps": float(fps),
            "total_frames": int(stream["nb_frames"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
        }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        raise


def extract_frames(
    video_path: str, num_frames: int = 8
) -> tuple[List[Dict], List[int]]:
    """
    Extract evenly-spaced frames from video using torchcodec.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 8)

    Returns:
        Tuple of (frames, frame_indices)
        - frames: List of dicts with 'bytes' and 'path' keys
        - frame_indices: List of actual frame indices extracted
    """
    # Open video with torchcodec
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)
    total_frames = len(decoder)

    # Calculate frame indices (evenly spaced, excluding last frame)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    # Get all frames at once
    frames_tensor = decoder.get_frames_at(frame_indices).data
    frames_tensor = rearrange(frames_tensor, 't c h w -> t h w c')
    frames_array = frames_tensor.cpu().numpy()

    frames = []
    for i, idx in enumerate(frame_indices):
        # Convert to PIL Image (already in RGB format)
        pil_image = Image.fromarray(frames_array[i])

        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        frames.append({"bytes": image_bytes, "path": None})

    return frames, frame_indices


def build_prompt(question: str, frame_indices: List[int], num_frames: int) -> List[Dict]:
    """
    Build prompt with system message and user message with frame placeholders.

    Args:
        question: Question text
        frame_indices: List of actual frame indices to display

    Returns:
        List of message dicts
    """
    # Build user content with actual frame indices
    frame_placeholders = "\n".join([f"frame {idx}:<image>" for idx in frame_indices])
    user_content = f"{frame_placeholders}\n{question}"
    system_content = get_system_prompt(num_frames=num_frames)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def process_single_sample(
    example: Dict[str, Any], idx: int, video_dir: str, num_frames: int = 8
) -> Dict[str, Any]:
    """
    Process a single sample for datasets.map().

    Args:
        example: Single sample from dataset
        idx: Sample index
        video_dir: Directory containing video files
        num_frames: Number of initial frames to extract

    Returns:
        Processed sample in RL format, or None if processing failed
    """
    try:
        # Get video path (could be relative or absolute)
        video_rel_path = example["video_path"]
        if video_rel_path.startswith("Video-Holmes/videos/"):
            # Extract just the filename
            video_filename = video_rel_path.split("/")[-1]
            video_path = str(Path(video_dir) / video_filename)
        elif video_rel_path.startswith("Video-Holmes/"):
            # Remove Video-Holmes/ prefix
            remaining_path = video_rel_path.replace("Video-Holmes/", "")
            video_path = str(Path(video_dir).parent / remaining_path)
        else:
            video_path = str(Path(video_dir) / video_rel_path)

        # Check if video exists
        if not Path(video_path).exists():
            print(f"\nWarning: Video not found: {video_path}")
            return None

        # Get video metadata
        video_meta = get_video_metadata(video_path)

        # Extract frames
        frames, frame_indices = extract_frames(video_path, num_frames=num_frames)
        assert (
            len(frames) == num_frames
        ), f"Expected {num_frames} frames, got {len(frames)}"
        assert (
            len(frame_indices) == num_frames
        ), f"Expected {num_frames} frame indices, got {len(frame_indices)}"

        # Build prompt with actual frame indices
        question = example["question"]
        prompt = build_prompt(question, frame_indices=frame_indices, num_frames=video_meta["total_frames"])

        # Get ground truth
        ground_truth = example["answer"]

        # Build extra_info
        extra_info = {
            "fps": video_meta["fps"],
            "height": video_meta["height"],
            "width": video_meta["width"],
            "total_frames": video_meta["total_frames"],
            "video_path": video_path,
            "answer": ground_truth,
            "question": question,
            "split": example.get("metadata", {}).get("split", "train"),
            "index": idx,
        }

        # Add optional fields if present
        if "thinking" in example:
            extra_info["thinking"] = example["thinking"]
        if "explanation" in example.get("metadata", {}):
            extra_info["explanation"] = example["metadata"]["explanation"]

        # Build RL format sample (matching official format)
        return {
            "video_path": video_path,
            "metadata": example.get("metadata", {}),
            "question_type": example.get("question_type", "mcq"),
            "agent_name": "action_agent",
            "data_source": "TencentARC/Video-Holmes",
            "prompt": prompt,
            "images": frames,  # datasets will handle numpy array
            "ability": "vl_video_reasoning",
            "env_name": "think_with_video",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "ground_truth": ground_truth,
            "extra_info": extra_info,
        }

    except Exception as e:
        print(f"\nError processing sample {idx}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert Video-Holmes JSON to RL parquet format (using datasets with multiprocessing)"
    )
    parser.add_argument(
        "json_file", type=str, help="Path to input JSON file (train.json or test.json)"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output parquet file (default: same name as JSON with .parquet extension)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of initial frames to extract (default: 8)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel processing (default: number of CPU cores)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = Path(args.json_file).with_suffix(".parquet")
    else:
        output_path = Path(args.output)

    # Determine number of processes
    num_proc = args.num_proc if args.num_proc is not None else os.cpu_count()
    print(f"Using {num_proc} processes for parallel processing")

    # Load dataset using datasets library
    print(f"Loading JSON from: {args.json_file}")
    dataset = load_dataset("json", data_files=args.json_file, split="train")
    print(f"Found {len(dataset)} samples")

    # Process dataset using map with multiprocessing
    print(f"Processing samples with {num_proc} processes...")

    def process_wrapper(example, idx):
        """Wrapper function for datasets.map()"""
        result = process_single_sample(
            example=example,
            idx=idx,
            video_dir=args.video_dir,
            num_frames=args.num_frames,
        )
        # Return empty dict if processing failed (will be filtered out)
        if result is None:
            return {"_skip": True}
        return result

    processed_dataset = dataset.map(
        process_wrapper, with_indices=True, num_proc=num_proc, desc="Processing samples"
    )

    # Filter out failed samples
    if "_skip" in processed_dataset.column_names:
        original_len = len(processed_dataset)
        processed_dataset = processed_dataset.filter(
            lambda x: "_skip" not in x or not x["_skip"]
        )
        processed_dataset = processed_dataset.remove_columns(["_skip"])
        print(f"Filtered out {original_len - len(processed_dataset)} failed samples")

    print(f"\nSuccessfully processed {len(processed_dataset)} samples")

    # Save to parquet
    print(f"Saving to: {output_path}")
    processed_dataset.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
