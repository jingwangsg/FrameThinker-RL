#!/usr/bin/env python3
"""
Filter train.json to remove samples with non-existent video paths,
then resample train_10k.json and train_3k.json from the filtered data.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def check_video_exists(sample_info: tuple) -> Dict:
    """
    Check if video file exists for a single sample.

    Args:
        sample_info: Tuple of (sample, base_dir)

    Returns:
        Dictionary with check results
    """
    sample, base_dir = sample_info
    video_path = sample.get("video_path", "")

    if not video_path:
        return {"sample": sample, "exists": False, "reason": "no_video_path"}

    # Construct full path
    full_path = os.path.join(base_dir, video_path)

    exists = os.path.exists(full_path)

    return {
        "sample": sample,
        "exists": exists,
        "video_path": video_path,
        "full_path": full_path,
        "reason": "exists" if exists else "file_not_found"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter train.json and resample train_10k and train_3k"
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to train.json file",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/amlfs-02/shared/datasets/s3/video_reason",
        help="Base directory for video paths (default: /mnt/amlfs-02/shared/datasets/s3/video_reason)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input directory)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=144,
        help="Number of parallel workers (default: 144)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.train_json).parent)

    print("=" * 80)
    print("Filter and Resample Train Data")
    print("=" * 80)
    print(f"Input train.json: {args.train_json}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Load train.json
    print("\nLoading train.json...")
    with open(args.train_json, 'r') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} samples")

    # Check video existence with multiprocessing
    print(f"\nChecking video existence with {args.num_workers} workers...")
    sample_infos = [(sample, args.base_dir) for sample in train_data]

    with mp.Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(check_video_exists, sample_infos),
                total=len(sample_infos),
                desc="Checking videos",
                unit="sample",
            )
        )

    # Filter valid samples
    print("\nFiltering samples...")
    valid_samples = [r["sample"] for r in results if r["exists"]]
    invalid_samples = [r for r in results if not r["exists"]]

    print(f"\nResults:")
    print(f"  Total samples: {len(train_data)}")
    print(f"  Valid samples: {len(valid_samples)}")
    print(f"  Invalid samples: {len(invalid_samples)}")

    # Show some invalid samples
    if invalid_samples:
        print(f"\nFirst 10 invalid samples:")
        for result in invalid_samples[:10]:
            print(f"  {result['video_path']} - {result['reason']}")

    # Save filtered train.json
    output_train = os.path.join(args.output_dir, "train_filtered.json")
    print(f"\nSaving filtered train.json to: {output_train}")
    with open(output_train, 'w') as f:
        json.dump(valid_samples, f, indent=2)

    # Sample 10k
    if len(valid_samples) >= 10000:
        train_10k = random.sample(valid_samples, 10000)
        output_10k = os.path.join(args.output_dir, "train_10k.json")
        print(f"Saving train_10k.json to: {output_10k}")
        with open(output_10k, 'w') as f:
            json.dump(train_10k, f, indent=2)
    else:
        print(f"\nWarning: Not enough samples for 10k (have {len(valid_samples)})")

    # Sample 3k
    if len(valid_samples) >= 3000:
        train_3k = random.sample(valid_samples, 3000)
        output_3k = os.path.join(args.output_dir, "train_3k.json")
        print(f"Saving train_3k.json to: {output_3k}")
        with open(output_3k, 'w') as f:
            json.dump(train_3k, f, indent=2)
    else:
        print(f"\nWarning: Not enough samples for 3k (have {len(valid_samples)})")

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
