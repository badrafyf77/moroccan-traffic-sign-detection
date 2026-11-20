"""Step 4: remove the Mapillary version suffix from class names.

``maximum-speed-limit-40--g1`` -> ``maximum-speed-limit-40``, so that design
variants of the same sign (``--g1``, ``--g15``, ...) share one class.

Usage:
    python data_preparation/04_strip_version_suffix.py path/to/dataset
"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

VERSION_SUFFIX = re.compile(r"--g\d+$")


def main():
    parser = argparse.ArgumentParser(description="Strip --gN version suffixes from class names.")
    parser.add_argument("dataset", type=Path, help="dataset folder that contains labels/")
    args = parser.parse_args()

    for path in tqdm(sorted((args.dataset / "labels").glob("*.json")), desc="Stripping suffixes", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for obj in data.get("objects", []):
            obj["label"] = VERSION_SUFFIX.sub("", obj.get("label", ""))
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done: version suffixes removed.")


if __name__ == "__main__":
    main()
