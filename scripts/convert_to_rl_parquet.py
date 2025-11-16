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

import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
from torchcodec.decoders import VideoDecoder
from einops import rearrange


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

def compute_target_size(width: int, height: int, size: int) -> tuple[int, int]:
    """
    Compute target size for video frame resizing.
    """
    if width > height:
        return size, int(size * height / width)
    else:
        return int(size * width / height), size

def extract_frames(
    video_path: str, num_frames: int = 8, size=360,
) -> tuple[List[Dict], List[int]]:
    """
    Extract evenly-spaced frames from video using torchcodec VideoDecoder.

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

    frames = decoder.get_frames_at(frame_indices).data
    frames = rearrange(frames, 't c h w -> t h w c')
    frames = frames.cpu().numpy()

    outputs = []

    for frame in frames:
        # Convert to PIL Image (ensure uint8)
        pil_image = Image.fromarray(frame)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        outputs.append({"bytes": image_bytes, "path": None})

    return outputs, frame_indices


def build_prompt(question: str) -> List[Dict]:
    """
    Build prompt with system message and user message with frame placeholders.

    Args:
        question: Question text
        frame_indices: List of actual frame indices to display

    Returns:
        List of message dicts
    """
    return [{"role": "user", "content": question}]


def process_single_sample(
    example: Dict[str, Any],
    idx: int,
    media_dir: str,
    num_frames: int = 8,
    data_source: str = "TencentARC/Video-Holmes",
) -> Dict[str, Any]:
    """
    Process a single sample for datasets.map().

    Args:
        example: Single sample from dataset
        idx: Sample index
        media_dir: Directory containing media files
        num_frames: Number of initial frames to extract

    Returns:
        Processed sample in RL format, or None if processing failed
    """
    try:
        # Get video path (could be relative or absolute)
        video_path = example["video_path"]
        abs_video_path = os.path.join(media_dir, video_path)

        # Check if video exists
        if not Path(abs_video_path).exists():
            print(f"\nWarning: Video not found: {abs_video_path}")
            return None

        # Get video metadata
        video_meta = get_video_metadata(abs_video_path)

        # Extract frames
        # frames, frame_indices = extract_frames(abs_video_path, num_frames=num_frames)

        # Build prompt with actual frame indices
        question = example["question"]
        prompt = build_prompt(question)

        # Get ground truth
        ground_truth = example["answer"]

        # Build extra_info
        extra_info = {
            "fps": video_meta["fps"],
            "height": video_meta["height"],
            "width": video_meta["width"],
            "total_frames": video_meta["total_frames"],
            "video_path": video_path,  # should be relative path
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

        # Build RL format sample
        return {
            "data_source": data_source,
            "prompt": prompt,
            # "images": frames,
            "ability": "vl_video_reasoning",
            "env_name": "think_with_video",
            "video_path": video_path,
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "ground_truth": ground_truth,
            "question_type": example.get("question_type", "mcq"),
            "metadata": example.get("metadata", {}),
            "extra_info": extra_info,
        }

    except Exception as e:
        print(f"\nError processing sample {idx}: {e}")
        return None


def get_empty_sample_schema():
    """
    Return empty schema matching successful samples for failed processing.

    This ensures schema consistency across all samples during multiprocessing,
    preventing KeyError when datasets.map() tries to merge results.
    """
    return {
        "_skip": True,
        "data_source": "",
        "prompt": [],
        "ability": "",
        "env_name": "",
        "reward_model": {},
        "ground_truth": "",
        "question_type": "",
        "metadata": {},
        "extra_info": {},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Video-Holmes JSON to RL parquet format (using datasets with multiprocessing)"
    )
    parser.add_argument(
        "json_file", type=str, help="Path to input JSON file (train.json or test.json)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output parquet file (default: same name as JSON with .parquet extension)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="TencentARC/Video-Holmes",
        help="Data source name (default: TencentARC/Video-Holmes)",
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
    parser.add_argument(
        "--media-dir",
        type=str,
        default=None,
        help="Directory containing media files (default: None)",
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
            media_dir=args.media_dir,
            num_frames=args.num_frames,
            data_source=args.data_source,
        )
        # Return consistent schema if processing failed (will be filtered out)
        if result is None:
            return get_empty_sample_schema()
        # Add skip flag to successful results for consistency
        result["_skip"] = False
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
