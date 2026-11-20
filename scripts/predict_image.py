"""Detect traffic signs on one image or a folder of images and save the annotated results.

Usage:
    python scripts/predict_image.py --source path/to/image.jpg
    python scripts/predict_image.py --source path/to/folder --conf 0.4 --show
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "weights" / "yolo11n_traffic_signs.pt"


def main():
    parser = argparse.ArgumentParser(description="Traffic sign detection on images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to the YOLO weights (.pt)")
    parser.add_argument("--source", required=True, help="image file or folder of images")
    parser.add_argument("--output", type=Path, default=Path("predictions"), help="folder for annotated images")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--show", action="store_true", help="also open each result in a window")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    for result in model.predict(args.source, conf=args.conf, stream=True, verbose=False):
        out_path = args.output / f"{Path(result.path).stem}_pred.jpg"
        result.save(filename=str(out_path))
        found = [f"{model.names[int(c)]} ({s:.2f})" for c, s in zip(result.boxes.cls, result.boxes.conf)]
        print(f"{Path(result.path).name}: {', '.join(found) or 'no signs'} -> {out_path}")
        if args.show:
            result.show()


if __name__ == "__main__":
    main()
