#!/usr/bin/env python3
"""
Add video metadata to Video-Holmes parquet files for RL training.

This script extracts video metadata (fps, total_frames, width, height) using ffprobe
and adds it to the extra_info field of each row in the parquet file.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


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


def process_parquet(input_path: str, output_path: str = None, backup: bool = True):
    """
    Add video metadata to parquet file.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (if None, overwrites input)
        backup: Whether to create backup of original file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Create backup if requested
    if backup and output_path == input_path:
        backup_path = input_path.with_suffix('.parquet.backup')
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)

    print(f"Reading parquet file: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"Processing {len(df)} rows...")

    # Track statistics
    success_count = 0
    error_count = 0
    error_videos = []

    # Update each row
    updated_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        row_dict = row.to_dict()
        video_path = row_dict['video_path']

        try:
            # Get video metadata
            video_meta = get_video_metadata(video_path)

            # Update extra_info
            extra_info = row_dict['extra_info']
            extra_info['fps'] = video_meta['fps']
            extra_info['video_path'] = video_path  # Redundant but required by RL dataset loader
            extra_info['total_frames'] = video_meta['total_frames']
            extra_info['height'] = video_meta['height']
            extra_info['width'] = video_meta['width']

            row_dict['extra_info'] = extra_info

            # Add required fields for RL training (based on official issue #4 example)
            if 'env_name' not in row_dict:
                row_dict['env_name'] = 'think_with_video'
            if 'ability' not in row_dict:
                row_dict['ability'] = 'vl_video_reasoning'

            updated_rows.append(row_dict)
            success_count += 1

        except Exception as e:
            print(f"\nError processing row {idx} (video: {video_path}): {e}")
            error_count += 1
            error_videos.append((idx, video_path, str(e)))
            # Keep original row if metadata extraction fails
            updated_rows.append(row_dict)

    # Create new dataframe and save
    print(f"\nSaving updated parquet to: {output_path}")
    updated_df = pd.DataFrame(updated_rows)
    updated_df.to_parquet(output_path, index=False)

    # Print summary
    print("\n" + "="*60)
    print("Processing Summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors: {error_count}")

    if error_videos:
        print(f"\nFailed videos:")
        for idx, video_path, error in error_videos[:10]:  # Show first 10 errors
            print(f"  Row {idx}: {video_path}")
            print(f"    Error: {error}")
        if len(error_videos) > 10:
            print(f"  ... and {len(error_videos) - 10} more errors")

    print("="*60)

    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Add video metadata to Video-Holmes parquet files"
    )
    parser.add_argument(
        "input_parquet",
        type=str,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output parquet file (default: overwrite input)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup of original file"
    )

    args = parser.parse_args()

    process_parquet(
        input_path=args.input_parquet,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
