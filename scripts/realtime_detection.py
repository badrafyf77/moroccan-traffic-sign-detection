"""Real-time traffic sign detection with distance estimation (video file, webcam, image or folder).

Shows each detection with its class, confidence and estimated distance, plus a sidebar
that lists the signs in the current frame and the processing speed (FPS).

Usage:
    python scripts/realtime_detection.py --source video.mp4
    python scripts/realtime_detection.py --source 0                 # webcam
    python scripts/realtime_detection.py --source video.mp4 --record --use_gpu

Keys: Q / Esc quit · S pause (any key resumes) · P save the current frame to capture.png
"""
import os
import sys
import argparse
import glob
import time
import threading
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "weights" / "yolo11n_traffic_signs.pt"

# --- Distance estimation (pinhole camera model) ---
# distance = real_sign_height * focal_length / box_height_in_pixels - offset
# The defaults are approximate values tuned on the test videos, not a camera calibration:
# measure your camera's focal length (in pixels) for accurate distances.
DEFAULT_SIGN_HEIGHT_M = 0.7    # typical height of a road sign panel, in meters
DEFAULT_FOCAL_LENGTH_PX = 800  # camera focal length, in pixels
DEFAULT_DISTANCE_OFFSET_M = 3.5  # empirical correction measured on the test videos

SIDEBAR_WIDTH = 240


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time traffic sign detection with distance estimation.")
    parser.add_argument('--model', default=str(DEFAULT_MODEL), help='Path to YOLO model file (.pt)')
    parser.add_argument('--source', required=True,
                        help='Image/video file, folder, or camera index (integer). '
                             'For video, provide the file (e.g., "traffic.mp4"). For webcam, use 0.')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--resolution', default=None, help='Window resolution WxH, e.g., 640x480. If omitted, video resolution is used.')
    parser.add_argument('--record', action='store_true', help='Record the annotated output to --record_path')
    parser.add_argument('--record_path', default='detections.avi', help='Output file used with --record')
    parser.add_argument('--keep_aspect', action='store_true',
                        help='Kept for compatibility: frames are always resized keeping their aspect ratio.')
    parser.add_argument('--skip_frames', type=int, default=2, help='Skip every n frames for video input')
    parser.add_argument('--use_gpu', action='store_true', help='Use CUDA GPU if available')
    parser.add_argument('--max_display_width', type=int, default=1280, help='Maximum total width (video+sidebar) for display')
    parser.add_argument('--max_display_height', type=int, default=720, help='Maximum total height for display')
    parser.add_argument('--sign_height', type=float, default=DEFAULT_SIGN_HEIGHT_M, help='Real sign height in meters')
    parser.add_argument('--focal_length', type=float, default=DEFAULT_FOCAL_LENGTH_PX, help='Camera focal length in pixels')
    parser.add_argument('--distance_offset', type=float, default=DEFAULT_DISTANCE_OFFSET_M,
                        help='Correction subtracted from the raw distance, in meters')
    return parser.parse_args()


def get_source_type(src):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    vid_ext = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
    if os.path.isdir(src):
        return 'folder'
    elif os.path.isfile(src):
        _, ext = os.path.splitext(src)
        ext = ext.lower()
        if ext in img_ext:
            return 'image'
        elif ext in vid_ext:
            return 'video'
        else:
            raise ValueError(f'File extension {ext} is unsupported.')
    else:
        try:
            cam_idx = int(src)
            return 'webcam', cam_idx
        except ValueError:
            raise ValueError('Source is invalid. Use file path or camera index (e.g., 0).')


def draw_sidebar(frame, detected_signs, resW, resH):
    sidebar_w = SIDEBAR_WIDTH
    sidebar = np.full((resH, sidebar_w, 3), 40, dtype=np.uint8)
    cv2.rectangle(sidebar, (0, 0), (sidebar_w - 1, resH - 1), (70, 70, 70), thickness=2)
    cv2.putText(sidebar, "Traffic Signs", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.line(sidebar, (10, 40), (sidebar_w - 10, 40), (80, 80, 80), 2)
    cv2.putText(sidebar, "Detections", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    y = 95
    if detected_signs:
        for sign, info in detected_signs.items():
            count, confs, dists = info
            msg = f"{sign}: {count}x"
            cv2.putText(sidebar, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            y += 28
            for conf, dist in zip(confs, dists):
                conf_msg = f"  {conf * 100:.1f}% {dist:.2f}m"
                cv2.putText(sidebar, conf_msg, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
                y += 22
            y += 6
    else:
        cv2.putText(sidebar, "No signs", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 120), 2)
    cv2.line(sidebar, (10, resH - 70), (sidebar_w - 10, resH - 70), (80, 80, 80), 2)
    cv2.putText(sidebar, "Controls:", (10, resH - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(sidebar, "Q: Quit | S: Pause", (10, resH - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(sidebar, "P: Save frame", (10, resH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return np.hstack([frame, sidebar])


def estimate_distance(box_height, sign_height=DEFAULT_SIGN_HEIGHT_M,
                      focal_length=DEFAULT_FOCAL_LENGTH_PX, offset=DEFAULT_DISTANCE_OFFSET_M):
    """Distance to the sign in meters, from the height of its bounding box in pixels."""
    if box_height <= 0:
        return 0.0
    return max(0.0, (sign_height * focal_length) / box_height - offset)


def process_and_show(frame, results, labels, bbox_colors, args, resW, resH, avg_frame_rate):
    detected_signs = {}
    for i in range(len(results[0].boxes)):
        xyxy = results[0].boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(results[0].boxes[i].cls.item())
        classname = labels[classidx]
        conf = results[0].boxes[i].conf.item()
        if conf > args.thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            box_height = ymax - ymin
            distance_m = estimate_distance(box_height, args.sign_height, args.focal_length, args.distance_offset)
            label = f'{classname} {distance_m:.2f}m'
            label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(
                frame,
                (xmin, label_ymin - label_size[1] - 10),
                (xmin + label_size[0], label_ymin + baseLine - 10),
                color, cv2.FILLED
            )
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            if classname not in detected_signs:
                detected_signs[classname] = [0, [], []]
            detected_signs[classname][0] += 1
            detected_signs[classname][1].append(conf)
            detected_signs[classname][2].append(distance_m)
    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    total_objects = sum([c[0] for c in detected_signs.values()])
    cv2.putText(frame, f'Objects: {total_objects}', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    frame_with_sidebar = draw_sidebar(frame, detected_signs, resW, resH)
    return frame_with_sidebar


def fit_with_sidebar_limit(frame, sidebar_w, max_total_w, max_total_h):
    """Resize frame so that frame+sidebar <= max_total_w (width), preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale_w = (max_total_w - sidebar_w) / w if w + sidebar_w > max_total_w else 1.0
    scale_h = max_total_h / h if h > max_total_h else 1.0
    scale = min(scale_w, scale_h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame


def video_reader_worker(cap, queue, skip_frames):
    """Read frames in a background thread so decoding doesn't slow down inference."""
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            queue.put(None)
            break
        frame_id += 1
        if skip_frames > 1 and (frame_id % skip_frames != 0):
            continue
        queue.put(frame)


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print('ERROR: Model path is invalid.')
        sys.exit(1)

    model = YOLO(args.model)
    if args.use_gpu:
        model.to('cuda')
    labels = model.names

    try:
        src_type = get_source_type(args.source)
    except Exception as e:
        print(e)
        sys.exit(1)

    resW = resH = None
    origW = origH = None
    sidebar_w = SIDEBAR_WIDTH

    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]
    frame_rate_buffer = []
    fps_avg_len = 200
    avg_frame_rate = 0

    video_fps = None
    recorder = None

    frame_queue = Queue(maxsize=5)
    video_thread = None
    skip_frames = args.skip_frames

    max_display_width = args.max_display_width
    max_display_height = args.max_display_height

    if src_type == 'folder':
        imgs_list = sorted([f for f in glob.glob(os.path.join(args.source, '*')) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        img_count = 0

        def next_frame():
            nonlocal img_count
            if img_count >= len(imgs_list):
                return None
            img = cv2.imread(imgs_list[img_count])
            img_count += 1
            return img
        if not args.resolution:
            img = cv2.imread(imgs_list[0])
            origH, origW = img.shape[:2]
    elif src_type == 'image':
        imgs_list = [args.source]
        img_count = 0

        def next_frame():
            nonlocal img_count
            if img_count >= len(imgs_list):
                return None
            img = cv2.imread(imgs_list[img_count])
            img_count += 1
            return img
        if not args.resolution:
            img = cv2.imread(imgs_list[0])
            origH, origW = img.shape[:2]
    elif isinstance(src_type, tuple) and src_type[0] == 'webcam':
        cam_idx = src_type[1]
        cap = cv2.VideoCapture(cam_idx)
        ret, temp = cap.read()
        if not ret:
            print("Error: Could not open webcam.")
            sys.exit(1)
        origH, origW = temp.shape[:2]

        def next_frame():
            ret, frame = cap.read()
            if not ret:
                return None
            return frame
    elif src_type == 'video':
        cap = cv2.VideoCapture(args.source)
        ret, temp = cap.read()
        if not ret:
            print("Error: Could not read video file.")
            sys.exit(1)
        origH, origW = temp.shape[:2]
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_thread = threading.Thread(target=video_reader_worker, args=(cap, frame_queue, skip_frames), daemon=True)
        video_thread.start()

        def next_frame():
            return frame_queue.get()
    else:
        print('Invalid source type.')
        sys.exit(1)

    if args.resolution:
        resW, resH = map(int, args.resolution.split('x'))
    else:
        resW, resH = origW, origH

    while True:
        t_start = time.perf_counter()
        frame = next_frame()
        if frame is None:
            print('Stream ended. Exiting.')
            break

        # Scale frame if needed to keep sidebar visible
        frame = fit_with_sidebar_limit(frame, sidebar_w, max_display_width, max_display_height)

        resH, resW = frame.shape[:2]
        results = model(frame, verbose=False)
        frame_with_sidebar = process_and_show(
            frame, results, labels, bbox_colors, args, resW, resH, avg_frame_rate
        )

        t_stop = time.perf_counter()
        frame_time = t_stop - t_start
        frame_rate_calc = float(1 / (frame_time + 1e-8))
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

        cv2.imshow('Traffic Sign Detection', frame_with_sidebar)
        if args.record:
            if recorder is None:
                # Created on the first frame so its size matches the displayed (resized) frames
                out_h, out_w = frame_with_sidebar.shape[:2]
                recorder = cv2.VideoWriter(args.record_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (out_w, out_h))
            recorder.write(frame_with_sidebar)

        if video_fps:
            delay = max(1, int(1000 / video_fps - (frame_time * 1000)))
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            cv2.waitKey()
        elif key == ord('p'):
            cv2.imwrite('capture.png', frame_with_sidebar)

    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if 'cap' in locals():
        cap.release()
    if recorder:
        recorder.release()
        print(f'Recording saved to {args.record_path}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
