"""Step 2: normalize class names and drop boxes that are too small.

* Removes the sign-type prefix from Mapillary class names
  (``regulatory--stop--g1`` -> ``stop--g1``).
* Removes boxes smaller than MIN_SIZE pixels in width or height: they are too
  small to be readable and mostly add noise.

Usage:
    python data_preparation/02_normalize_labels.py path/to/dataset [--min-size 50]
"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

TYPE_PREFIX = re.compile(r"^.*?--")


def main():
    parser = argparse.ArgumentParser(description="Strip class-name prefixes and drop tiny boxes.")
    parser.add_argument("dataset", type=Path, help="dataset folder that contains labels/")
    parser.add_argument("--min-size", type=int, default=50, help="minimum box width and height in pixels")
    args = parser.parse_args()

    removed = 0
    for path in tqdm(sorted((args.dataset / "labels").glob("*.json")), desc="Normalizing", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        kept = []
        for obj in data.get("objects", []):
            box = obj.get("bbox", {})
            width = box.get("xmax", 0) - box.get("xmin", 0)
            height = box.get("ymax", 0) - box.get("ymin", 0)
            if width >= args.min_size and height >= args.min_size:
                obj["label"] = TYPE_PREFIX.sub("", obj.get("label", ""))
                kept.append(obj)
            else:
                removed += 1
        data["objects"] = kept
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done: {removed} boxes smaller than {args.min_size}px removed.")


if __name__ == "__main__":
    main()
