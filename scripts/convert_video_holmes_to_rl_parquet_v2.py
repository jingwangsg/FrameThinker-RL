#!/usr/bin/env python3
"""
Convert Video-Holmes JSON files to RL training parquet format.

This script creates complete RL-format parquet files from Video-Holmes JSON,
matching the official format from GitHub issue #4.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import io

import decord
import numpy as np
from datasets import load_dataset
from PIL import Image

# Set decord to use native bridge
decord.bridge.set_bridge('native')


SYSTEM_PROMPT = """You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
For each turn, you MUST do the following
1. Start by discussing the observed evidence and how it relates to the question.
2. MUST provide the rationale to take the next action within the <thinking> </thinking> tag
3. MUST provide the action to take within an <action> </action> tag.

Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. The number of frames is fixed, currently 8. You must choose START_FRAME and END_FRAME from frame indices in the previous turn.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time. Convert hours to minutes if needed (e.g., for 1 hour, 2 minutes, and 30 seconds, use 62:30). You must reference MM:SS from the previous turn or given question.
3. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident."""


def get_video_metadata(video_path: str) -> Dict[str, any]:
    """
    Extract video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing fps, total_frames, width, height
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,width,height',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data['streams'][0]

        # Parse frame rate (e.g., "24000/1001" -> 23.976)
        r_frame_rate = stream['r_frame_rate']
        num, den = map(int, r_frame_rate.split('/'))
        fps = num / den

        return {
            'fps': float(fps),
            'total_frames': int(stream['nb_frames']),
            'width': int(stream['width']),
            'height': int(stream['height'])
        }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        raise


def extract_frames(video_path: str, num_frames: int = 8) -> tuple[List[Dict], List[int]]:
    """
    Extract evenly-spaced frames from video using decord.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 8)

    Returns:
        Tuple of (frames, frame_indices)
        - frames: List of dicts with 'bytes' and 'path' keys
        - frame_indices: List of actual frame indices extracted
    """
    # Open video with decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)

    # Calculate frame indices (evenly spaced, excluding last frame)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        # Get frame (already in RGB format)
        frame = vr[idx].asnumpy()

        # Convert to PIL Image
        pil_image = Image.fromarray(frame)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        frames.append({
            'bytes': image_bytes,
            'path': None
        })

    return frames, frame_indices


def build_prompt(question: str, frame_indices: List[int]) -> List[Dict]:
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

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def process_sample(example: Dict, idx: int, video_dir: str, num_frames: int = 8) -> Dict:
    """
    Process a single sample from Video-Holmes JSON to RL format.

    This function is designed to work with datasets.map() for parallel processing.

    Args:
        example: Single sample dict from JSON
        idx: Sample index
        video_dir: Directory containing video files
        num_frames: Number of initial frames to extract

    Returns:
        Dict with RL format and '_success' flag indicating processing status
    """
    try:
        # Get video path (could be relative or absolute)
        video_rel_path = example['video_path']
        if video_rel_path.startswith('Video-Holmes/videos/'):
            # Extract just the filename
            video_filename = video_rel_path.split('/')[-1]
            video_path = str(Path(video_dir) / video_filename)
        elif video_rel_path.startswith('Video-Holmes/'):
            # Remove Video-Holmes/ prefix
            remaining_path = video_rel_path.replace('Video-Holmes/', '')
            video_path = str(Path(video_dir).parent / remaining_path)
        else:
            video_path = str(Path(video_dir) / video_rel_path)

        # Check if video exists
        if not Path(video_path).exists():
            print(f"Warning: Video not found: {video_path}")
            return {'_success': False}

        # Get video metadata
        video_meta = get_video_metadata(video_path)

        # Extract frames
        frames, frame_indices = extract_frames(video_path, num_frames=num_frames)

        # Build prompt with actual frame indices
        question = example['question']
        prompt = build_prompt(question, frame_indices=frame_indices)

        # Get ground truth
        ground_truth = example['answer']

        # Build RL format sample
        rl_sample = {
            "agent_name": "action_agent",
            'data_source': 'TencentARC/Video-Holmes',
            'prompt': prompt,
            'images': frames,  # datasets will handle the list of dicts
            'ability': 'vl_video_reasoning',
            'env_name': 'think_with_video',
            'reward_model': {
                'ground_truth': ground_truth,
                'style': 'rule'
            },
            'ground_truth': ground_truth,
            'question_type': example.get('question_type', 'mcq'),
            'video_path': video_path,
            'metadata': example.get('metadata', {}),
            'extra_info': {
                'fps': video_meta['fps'],
                'height': video_meta['height'],
                'width': video_meta['width'],
                'total_frames': video_meta['total_frames'],
                'video_path': video_path,
                'answer': ground_truth,
                'question': question,
                'split': example.get('metadata', {}).get('split', 'train'),
                'index': idx
            },
            '_success': True  # Mark as successfully processed
        }

        # Add optional fields if present
        if 'thinking' in example:
            rl_sample['extra_info']['thinking'] = example['thinking']
        if 'explanation' in example.get('metadata', {}):
            rl_sample['extra_info']['explanation'] = example['metadata']['explanation']

        return rl_sample

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return {'_success': False}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Video-Holmes JSON to RL parquet format using datasets library with multiprocessing"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to input JSON file (train.json or test.json)"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output parquet file (default: same name as JSON with .parquet extension)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of initial frames to extract (default: 8)"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for parallel processing (default: 1)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = str(Path(args.json_file).with_suffix('.parquet'))
    else:
        output_path = str(args.output)

    # Load JSON dataset
    print(f"Loading JSON from: {args.json_file}")
    dataset = load_dataset("json", data_files=args.json_file, split="train")
    print(f"Loaded {len(dataset)} samples")

    # Process samples with multiprocessing
    print(f"Processing samples with {args.num_proc} processes...")

    def map_fn(example, idx):
        """Wrapper function for dataset.map()"""
        return process_sample(
            example=example,
            idx=idx,
            video_dir=args.video_dir,
            num_frames=args.num_frames
        )

    # Apply processing with multiprocessing
    processed_dataset = dataset.map(
        map_fn,
        with_indices=True,
        num_proc=args.num_proc,
        desc="Processing samples",
        remove_columns=dataset.column_names  # Remove original columns
    )

    # Filter out failed samples
    processed_dataset = processed_dataset.filter(
        lambda x: x.get('_success', False),
        desc="Filtering failed samples"
    )

    print(f"Successfully processed {len(processed_dataset)} samples")

    # Remove the _success flag before saving
    processed_dataset = processed_dataset.remove_columns(['_success'])

    # Save to parquet
    print(f"Saving to: {output_path}")
    processed_dataset.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
