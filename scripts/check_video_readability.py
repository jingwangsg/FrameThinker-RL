#!/usr/bin/env python3
"""
Check video readability using torchcodec.

This script scans video directories and checks if each video can be decoded
using torchcodec.decoders.VideoDecoder. It uses multiprocessing to parallelize
the checking process and generates a detailed JSON report.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


def get_video_files(
    directories: List[str], extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
) -> List[Tuple[str, str]]:
    """
    Scan directories for video files.

    Args:
        directories: List of directory paths to scan
        extensions: Tuple of video file extensions to look for

    Returns:
        List of tuples (video_path, parent_directory)
    """
    video_files = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {directory}")
            continue

        print(f"Scanning directory: {directory}")
        for video_file in dir_path.rglob("*"):
            if video_file.is_file() and video_file.suffix.lower() in extensions:
                video_files.append((str(video_file), directory))

    return video_files


def check_single_video(video_info: Tuple[str, str]) -> Dict:
    """
    Check if a single video can be read with torchcodec.

    Args:
        video_info: Tuple of (video_path, parent_directory)

    Returns:
        Dictionary with check results
    """
    video_path, parent_dir = video_info

    result = {
        "path": video_path,
        "parent_directory": parent_dir,
        "readable": False,
        "error_type": None,
        "error_message": None,
        "file_size_mb": None,
    }

    try:
        # Get file size
        file_size = os.path.getsize(video_path)
        result["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        # Try to create decoder and read first frame
        # Import here to avoid issues with multiprocessing
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)

        # Try to get metadata
        metadata = decoder.metadata
        num_frames = metadata.num_frames

        # Try to decode the first frame
        if num_frames > 0:
            _ = decoder[0]

        result["readable"] = True

    except FileNotFoundError as e:
        result["error_type"] = "FileNotFoundError"
        result["error_message"] = str(e)
    except PermissionError as e:
        result["error_type"] = "PermissionError"
        result["error_message"] = str(e)
    except Exception as e:
        # Catch all other exceptions (decoder errors, format errors, etc.)
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)

    return result


def generate_report(
    results: List[Dict], total_time: float, output_path: Optional[str] = None
) -> Dict:
    """
    Generate summary report from check results.

    Args:
        results: List of check result dictionaries
        total_time: Total time taken for checking
        output_path: Optional path to save JSON report

    Returns:
        Report dictionary
    """
    # Calculate summary statistics
    total_videos = len(results)
    readable = sum(1 for r in results if r["readable"])
    unreadable = total_videos - readable

    # Group by directory
    by_directory = defaultdict(lambda: {"total": 0, "readable": 0, "unreadable": 0})
    for result in results:
        parent_dir = result["parent_directory"]
        by_directory[parent_dir]["total"] += 1
        if result["readable"]:
            by_directory[parent_dir]["readable"] += 1
        else:
            by_directory[parent_dir]["unreadable"] += 1

    # Get unreadable videos
    unreadable_videos = [r for r in results if not r["readable"]]

    # Build report
    report = {
        "summary": {
            "total_videos": total_videos,
            "readable": readable,
            "unreadable": unreadable,
            "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(total_time, 2),
        },
        "by_directory": dict(by_directory),
        "unreadable_videos": unreadable_videos,
    }

    # Save to file if requested
    if output_path:
        output_file = Path(output_path)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check video readability using torchcodec with multiprocessing"
    )
    parser.add_argument(
        "--video-dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of video directories to check",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=144,
        help="Number of parallel workers (default: 144)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="video_check_report.json",
        help="Output JSON report file path (default: video_check_report.json)",
    )
    parser.add_argument(
        "--video-extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        help="Video file extensions to check (default: .mp4 .avi .mov .mkv .webm)",
    )

    args = parser.parse_args()

    # Convert extensions to tuple (with lowercase and leading dot)
    extensions = tuple(
        ext if ext.startswith(".") else f".{ext}" for ext in args.video_extensions
    )
    extensions = tuple(ext.lower() for ext in extensions)

    print("=" * 80)
    print("Video Readability Check using TorchCodec")
    print("=" * 80)
    print(f"Directories to check: {args.video_dirs}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Video extensions: {extensions}")
    print(f"Output report: {args.output}")
    print("=" * 80)

    # Scan for video files
    print("\nScanning for video files...")
    video_files = get_video_files(args.video_dirs, extensions)
    print(f"Found {len(video_files)} video files")

    if len(video_files) == 0:
        print("No video files found. Exiting.")
        return

    # Process videos with multiprocessing
    print(f"\nChecking videos with {args.num_workers} workers...")
    start_time = time.time()

    with mp.Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(check_single_video, video_files),
                total=len(video_files),
                desc="Checking videos",
                unit="video",
            )
        )

    end_time = time.time()
    total_time = end_time - start_time

    # Generate and save report
    print("\nGenerating report...")
    report = generate_report(results, total_time, args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total videos checked: {report['summary']['total_videos']}")
    print(f"Readable: {report['summary']['readable']}")
    print(f"Unreadable: {report['summary']['unreadable']}")
    print(f"Duration: {report['summary']['duration_seconds']:.2f} seconds")
    print()

    # Print per-directory summary
    print("Per-directory breakdown:")
    for directory, stats in report["by_directory"].items():
        print(f"  {directory}:")
        print(f"    Total: {stats['total']}")
        print(f"    Readable: {stats['readable']}")
        print(f"    Unreadable: {stats['unreadable']}")

    # Print some unreadable examples if any
    if report["summary"]["unreadable"] > 0:
        print(f"\nFirst 10 unreadable videos:")
        for video in report["unreadable_videos"][:10]:
            print(f"  {video['path']}")
            print(f"    Error: {video['error_type']}: {video['error_message']}")

    print("\n" + "=" * 80)
    print("Check complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
