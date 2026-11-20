"""Step 5: remove the catch-all ``other-sign`` class.

``other-sign`` groups unrelated signs together, which confuses the detector,
so those boxes are dropped.

Usage:
    python data_preparation/05_remove_other_sign.py path/to/dataset
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Remove boxes labelled 'other-sign'.")
    parser.add_argument("dataset", type=Path, help="dataset folder that contains labels/")
    args = parser.parse_args()

    removed = 0
    for path in tqdm(sorted((args.dataset / "labels").rglob("*.json")), desc="Removing other-sign", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        objects = data.get("objects", [])
        kept = [obj for obj in objects if obj.get("label") != "other-sign"]
        removed += len(objects) - len(kept)
        data["objects"] = kept
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Done: {removed} 'other-sign' boxes removed.")


if __name__ == "__main__":
    main()
