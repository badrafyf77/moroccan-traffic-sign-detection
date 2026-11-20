"""Count boxes per class and write the result to a text file (most frequent first).

Usage:
    python data_preparation/count_labels.py path/to/dataset [--output label_counts.txt]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Count boxes per class.")
    parser.add_argument("dataset", type=Path, help="dataset folder that contains labels/")
    parser.add_argument("--output", type=Path, default=Path("label_counts.txt"), help="output text file")
    args = parser.parse_args()

    counts = Counter()
    for path in tqdm(sorted((args.dataset / "labels").glob("*.json")), desc="Counting", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        counts.update(obj["label"] for obj in data.get("objects", []) if obj.get("label"))

    args.output.write_text("".join(f"{label}: {n}\n" for label, n in counts.most_common()), encoding="utf-8")
    print(f"{len(counts)} classes, {sum(counts.values())} boxes. Counts saved to {args.output}")


if __name__ == "__main__":
    main()
