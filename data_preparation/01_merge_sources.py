"""Step 1: merge every source dataset into a single images/ + labels/ folder.

Walks ROOT, collects every file found in any folder named ``images`` (jpg/jpeg/png)
or ``labels`` (json), and copies them into ROOT/images and ROOT/labels.
A file whose name already exists in the output is skipped, which removes
duplicate images that appear in several source datasets.

Usage:
    python data_preparation/01_merge_sources.py path/to/raw_sources
"""
import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description="Merge source datasets into ROOT/images and ROOT/labels.")
    parser.add_argument("root", type=Path, help="folder containing the downloaded source datasets")
    args = parser.parse_args()

    out_images, out_labels = args.root / "images", args.root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    to_copy = []
    for dirpath, dirnames, files in os.walk(args.root):
        dirnames.sort()          # visit sources in a fixed order: the first copy of a duplicate wins
        files.sort()
        folder = Path(dirpath)
        if folder in (out_images, out_labels):
            continue
        if folder.name == "images":
            to_copy += [(folder / f, out_images) for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        elif folder.name == "labels":
            to_copy += [(folder / f, out_labels) for f in files if f.endswith(".json")]

    copied = skipped = 0
    for src, dst_dir in tqdm(to_copy, desc="Merging", unit="file"):
        dst = dst_dir / src.name
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"Done: {copied} files copied, {skipped} duplicates skipped.")


if __name__ == "__main__":
    main()
