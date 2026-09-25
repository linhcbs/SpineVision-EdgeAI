"""
YOLOv8n-pose - Multi-person pose estimation
Using ultralytics library (v8.4.92)
Model: yolov8n-pose.pt (auto-download if not present)
17 keypoints (COCO format), multi-person support

source: https://docs.ultralytics.com/tasks/pose
"""

import os
import sys
import time
import cv2
import numpy as np
import psutil
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    from drawing_utils import resolve_weight_path
except ImportError:
    from benchmarks.models.drawing_utils import resolve_weight_path

# ============ CONFIG ============
MODEL_PATH = resolve_weight_path("yolov8n-pose.pt")
CAM_INDEX  = 1
CONF_THRESH = 0.5

# COCO skeleton connections
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Color per body part group (BGR)
PART_COLORS = {
    "head":  (0, 255, 255),
    "upper": (0, 200, 255),
    "torso": (0, 255, 100),
    "legs":  (100, 0, 255),
}
BONE_COLORS = [
    PART_COLORS["head"],  PART_COLORS["head"],  PART_COLORS["head"], PART_COLORS["head"],
    PART_COLORS["upper"],
    PART_COLORS["upper"], PART_COLORS["upper"],
    PART_COLORS["upper"], PART_COLORS["upper"],
    PART_COLORS["torso"], PART_COLORS["torso"],
    PART_COLORS["torso"],
    PART_COLORS["legs"],  PART_COLORS["legs"],
    PART_COLORS["legs"],  PART_COLORS["legs"],
]

COLOR_FPS   = (0, 255, 255)
COLOR_LAT   = (0, 255, 128)
COLOR_RAM   = (255, 200, 0)
COLOR_MODEL = (200, 200, 255)
COLOR_JOINT_B = (255, 255, 255)


def open_camera(preferred_index=1):
    for idx in [preferred_index] + [i for i in range(6) if i != preferred_index]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                if idx != preferred_index:
                    print(f"-> Auto-switched to camera {idx}")
                return cap, idx
            cap.release()
    return None, -1


def draw_skeleton(frame, keypoints_xy, keypoints_conf, vis_thresh=0.3):
    pts = []
    for i in range(len(keypoints_xy)):
        x, y = keypoints_xy[i]
        conf = float(keypoints_conf[i]) if keypoints_conf is not None else 1.0
        pts.append((int(x), int(y), conf))

    for idx, (s, e) in enumerate(CONNECTIONS):
        if s >= len(pts) or e >= len(pts):
            continue
        x1, y1, v1 = pts[s]
        x2, y2, v2 = pts[e]
        if v1 < vis_thresh or v2 < vis_thresh:
            continue
        color = BONE_COLORS[idx] if idx < len(BONE_COLORS) else (0, 200, 0)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    for i, (x, y, v) in enumerate(pts):
        if v < vis_thresh:
            continue
        # Color by body part
        if i < 5:
            jc = PART_COLORS["head"]
        elif i < 11:
            jc = PART_COLORS["upper"]
        elif i < 13:
            jc = PART_COLORS["torso"]
        else:
            jc = PART_COLORS["legs"]
        cv2.circle(frame, (x, y), 5, jc, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, COLOR_JOINT_B, 1, cv2.LINE_AA)


def draw_hud(frame, fps, latency_ms, ram_mb, n_persons):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (320, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "YOLOv8n-pose (Multi-person)", (14, 30),  font, 0.55, COLOR_MODEL, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS     : {fps:5.1f}",        (14, 58),  font, 0.6,  COLOR_FPS,   2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency : {latency_ms:5.1f} ms",(14, 84),font, 0.6,  COLOR_LAT,   2, cv2.LINE_AA)
    cv2.putText(frame, f"RAM     : {ram_mb:5.0f} MB",  (14, 110), font, 0.6,  COLOR_RAM,   2, cv2.LINE_AA)
    cv2.putText(frame, f"Persons : {n_persons}",       (14, 136), font, 0.6,  (180,180,180), 2, cv2.LINE_AA)


def main():
    print("[YOLOv8n-pose] Loading model...")
    model = YOLO(MODEL_PATH)   # Auto-downloads if not present
    proc  = psutil.Process()

    cap, _ = open_camera(CAM_INDEX)
    if cap is None:
        raise RuntimeError("No webcam found.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_ema = 0.0
    prev_t  = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            results = model(frame, verbose=False, conf=CONF_THRESH)
            lat_ms  = (time.perf_counter() - t0) * 1000

            n_persons = 0
            for r in results:
                if r.keypoints is None:
                    continue
                kps_xy   = r.keypoints.xy.cpu().numpy()    # (N, 17, 2)
                kps_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                n_persons = len(kps_xy)
                for i in range(n_persons):
                    conf_arr = kps_conf[i] if kps_conf is not None else None
                    draw_skeleton(frame, kps_xy[i], conf_arr, vis_thresh=CONF_THRESH)

            now    = time.time()
            inst   = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = 0.9 * fps_ema + 0.1 * inst
            ram_mb  = proc.memory_info().rss / 1024**2

            draw_hud(frame, fps_ema, lat_ms, ram_mb, n_persons)
            cv2.imshow("YOLOv8n-pose | press q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
