"""Step 3: keep only the sign classes used on Moroccan roads.

Every box whose class is not in TARGET_CLASSES is removed. Class names still
carry their Mapillary version suffix (``--g1``) at this point; step 4 strips it.

Usage:
    python data_preparation/03_keep_target_classes.py path/to/dataset
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

TARGET_CLASSES = {
    "airport--g1", "bicycles-only--g1", "buses-only--g1", "children--g1", "crossroads--g1",
    "curve-left--g1", "curve-right--g1", "distance--g1", "domestic-animals--g1", "double-curve-first-left--g1",
    "double-curve-first-right--g1", "end-of-bicycles-only--g1", "end-of-buses-only--g1", "end-of-motorway--g1",
    "end-of-priority-road--g1", "end-of-prohibition--g1", "end-of-speed-limit-zone--g1",
    "falling-rocks-or-debris-right--g1", "gas-station--g1", "give-way-to-oncoming-traffic--g1", "go-straight--g1",
    "go-straight-or-turn-left--g1", "go-straight-or-turn-right--g1", "height-limit--g1", "hospital--g1",
    "keep-left--g1", "keep-right--g1", "maximum-speed-limit-10--g1", "maximum-speed-limit-100--g1",
    "maximum-speed-limit-120--g1", "maximum-speed-limit-20--g1", "maximum-speed-limit-30--g1",
    "maximum-speed-limit-40--g1", "maximum-speed-limit-60--g1", "maximum-speed-limit-80--g1",
    "minimum-safe-distance--g1", "motorway--g1", "no-bicycles--g1", "no-entry--g1", "no-heavy-goods-vehicles--g1",
    "no-left-turn--g1", "no-motor-vehicles--g1", "no-motor-vehicles-except-motorcycles--g1", "no-motorcycles--g1",
    "no-overtaking--g1", "no-overtaking-by-heavy-goods-vehicles--g1", "no-parking--g1", "no-pedestrians--g1",
    "no-right-turn--g1", "no-stopping--g1", "no-stopping--g15", "no-u-turn--g1",
    "no-vehicles-carrying-dangerous-goods--g1", "one-way-left--g1", "one-way-right--g1", "other-danger--g1",
    "parking--g1", "pedestrians-crossing--g1", "pedestrians-only--g1", "priority-route-at-intersection--g1",
    "road-bump--g1", "road-narrows--g1", "road-narrows-left--g1", "road-narrows-right--g1", "roadworks--g1",
    "roundabout--g1", "slippery-road-surface--g1", "traffic-signals--g1", "two-way-traffic--g1", "yield--g1",
    # classes coming from the non-Mapillary sources
    "Guide_Sign", "stop", "stop--g1", "other-sign",
}


def main():
    parser = argparse.ArgumentParser(description="Remove boxes whose class is not a target class.")
    parser.add_argument("dataset", type=Path, help="dataset folder that contains labels/")
    args = parser.parse_args()

    removed = 0
    for path in tqdm(sorted((args.dataset / "labels").glob("*.json")), desc="Filtering classes", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        objects = data.get("objects", [])
        kept = [obj for obj in objects if obj.get("label", "") in TARGET_CLASSES]
        removed += len(objects) - len(kept)
        data["objects"] = kept
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Done: {removed} boxes of non-target classes removed.")


if __name__ == "__main__":
    main()
