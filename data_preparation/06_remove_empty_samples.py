"""Step 6: delete samples that no longer contain any sign.

After steps 2-5 some label files have an empty ``objects`` list. This script
deletes those label files and their images.

Usage:
    python data_preparation/06_remove_empty_samples.py path/to/dataset
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser(description="Delete label files with no objects, and their images.")
    parser.add_argument("dataset", type=Path, help="dataset folder that contains images/ and labels/")
    args = parser.parse_args()

    images_dir, labels_dir = args.dataset / "images", args.dataset / "labels"
    deleted = 0
    for path in tqdm(sorted(labels_dir.glob("*.json")), desc="Removing empty samples", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "objects" in data and not data["objects"]:
            path.unlink()
            for ext in IMAGE_EXTENSIONS:
                image = images_dir / (path.stem + ext)
                if image.exists():
                    image.unlink()
                    break
            deleted += 1

    print(f"Done: {deleted} empty samples deleted.")


if __name__ == "__main__":
    main()
