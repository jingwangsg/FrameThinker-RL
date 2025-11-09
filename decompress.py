#!/usr/bin/env python3

import glob
import tarfile
import os
from multiprocessing import Pool


def extract_tar(tar_file):
    try:
        with tarfile.open(tar_file, 'r') as tar:
            for member in tar:
                try:
                    tar.extract(member, path='.')
                except FileExistsError:
                    continue
                except Exception as e:
                    if "File exists" in str(e) or "already exists" in str(e):
                        continue
                    raise

        print(f"Extracted: {tar_file}")
        return tar_file
    except Exception as e:
        print(f"Error extracting {tar_file}: {e}")
        return None


def main():
    tar_files = glob.glob("*.tar")

    if not tar_files:
        print("No tar files found in current directory")
        return

    print(f"Found {len(tar_files)} tar files")

    with Pool() as pool:
        pool.map(extract_tar, tar_files)

    print("All extractions completed")


if __name__ == "__main__":
    main()