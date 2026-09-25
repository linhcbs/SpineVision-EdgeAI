"""
YOLO26m-pose (Medium) - Multi-person pose estimation
Architecture: YOLO-Pose Medium / Ultralytics Pose
Weights: yolo11m-pose.pt / yolo26m-pose.pt
17 keypoints (COCO format), multi-person support, high accuracy balanced for real-time applications.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import psutil
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    from drawing_utils import (
        load_config, resolve_weight_path, open_camera, draw_coco_landmarks, draw_hud,
        COLOR_FPS, COLOR_LAT, COLOR_RAM, COLOR_MODEL, COCO_CONNECTIONS
    )
except ImportError:
    from benchmarks.models.drawing_utils import (
        load_config, resolve_weight_path, open_camera, draw_coco_landmarks, draw_hud,
        COLOR_FPS, COLOR_LAT, COLOR_RAM, COLOR_MODEL, COCO_CONNECTIONS
    )

# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION CONSTANTS (ALL CAPS)
# ==============================================================================
CONFIG_FILE_PATH              = os.path.join(os.path.dirname(SCRIPT_DIR), "configs", "pose_models_config.json")
MODEL_KEY                     = "yolo26m_pose"
_CFG, _GLOBAL                 = load_config(MODEL_KEY, CONFIG_FILE_PATH)

MODEL_NAME                    = _CFG.get("model_name", "YOLO26m-pose (Medium)")
_WEIGHTS_FILE                 = _CFG.get("weights_file", "yolo11m-pose.pt")
WEIGHTS_PATH                  = resolve_weight_path(_WEIGHTS_FILE)

INPUT_SIZE                    = _CFG.get("input_size", 640)
CONF_THRESHOLD                = _CFG.get("conf_threshold", 0.5)
IOU_THRESHOLD                 = _CFG.get("iou_threshold", 0.45)
DEVICE                        = _CFG.get("device", "cpu")
HALF_PRECISION                = _CFG.get("half_precision", False)
HAS_3D_DEPTH                  = _CFG.get("has_3d_depth", False)

POINT_RADIUS_MIN              = _CFG.get("point_radius_min", 3)
POINT_RADIUS_MAX              = _CFG.get("point_radius_max", 8)

DEFAULT_CAMERA_INDEX          = _GLOBAL.get("camera", {}).get("camera_index", 1)
CAMERA_FRAME_WIDTH            = _GLOBAL.get("camera", {}).get("width", 1280)
CAMERA_FRAME_HEIGHT           = _GLOBAL.get("camera", {}).get("height", 720)
FALLBACK_CAMERA_INDICES       = _GLOBAL.get("camera", {}).get("fallback_indices", [0, 1, 2, 3, 4, 5])

# Training defaults from config
_TRAIN_CFG                    = _CFG.get("train_config", {})
TRAIN_DATASET_YAML            = _TRAIN_CFG.get("dataset_yaml", "coco8-pose.yaml")
TRAIN_EPOCHS                  = _TRAIN_CFG.get("epochs", 50)
TRAIN_BATCH_SIZE              = _TRAIN_CFG.get("batch_size", 8)
TRAIN_LEARNING_RATE           = _TRAIN_CFG.get("lr0", 0.01)
TRAIN_SAVE_DIR                = _TRAIN_CFG.get("save_dir", "runs/pose/yolo26m_pose")


# ==============================================================================
# DETECTOR CLASS (EASY INFERENCE, EXTENSION & TRAINING)
# ==============================================================================
class YOLOPoseDetector:
    """Wrapper class for YOLO-Pose (Medium) with inference & train methods."""

    def __init__(self,
                 weights_path=WEIGHTS_PATH,
                 conf_thresh=CONF_THRESHOLD,
                 iou_thresh=IOU_THRESHOLD,
                 device=DEVICE,
                 half=HALF_PRECISION,
                 imgsz=INPUT_SIZE):
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.half = half
        self.imgsz = imgsz
        print(f"[{MODEL_NAME}] Loading YOLO model from {self.weights_path}...")
        self.model = YOLO(self.weights_path)

    def infer(self, frame_bgr):
        """
        Run YOLO pose inference on a BGR image frame.
        Returns: (results, list_of_person_keypoints, latency_ms)
        """
        t0 = time.perf_counter()
        kwargs = {
            "verbose": False,
            "conf": self.conf_thresh,
            "iou": self.iou_thresh,
            "device": self.device,
            "imgsz": self.imgsz,
        }
        if self.half and str(self.device).lower() not in ["cpu", ""]:
            kwargs["half"] = True

        results = self.model(frame_bgr, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        all_persons_kps = []
        for r in results:
            if r.keypoints is None:
                continue
            kps_xy = r.keypoints.xy.cpu().numpy()     # (N, 17, 2)
            kps_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            for i in range(len(kps_xy)):
                pts_with_conf = []
                for j in range(len(kps_xy[i])):
                    x, y = kps_xy[i][j]
                    c = float(kps_conf[i][j]) if kps_conf is not None else 1.0
                    pts_with_conf.append((x, y, c))
                all_persons_kps.append(pts_with_conf)

        return results, all_persons_kps, latency_ms

    def draw(self, frame_bgr, all_persons_kps, conf_thresh=CONF_THRESHOLD,
             radius_min=POINT_RADIUS_MIN, radius_max=POINT_RADIUS_MAX):
        """Draw multi-person COCO skeleton on the frame."""
        n_persons = len(all_persons_kps)
        for person_kps in all_persons_kps:
            draw_coco_landmarks(
                frame_bgr,
                person_kps,
                connections=COCO_CONNECTIONS,
                conf_thresh=conf_thresh,
                radius_min=radius_min,
                radius_max=radius_max
            )
        return n_persons

    def train(self,
              data=TRAIN_DATASET_YAML,
              epochs=TRAIN_EPOCHS,
              batch=TRAIN_BATCH_SIZE,
              imgsz=INPUT_SIZE,
              lr0=TRAIN_LEARNING_RATE,
              project=TRAIN_SAVE_DIR):
        """Fine-tune / train the YOLO-Pose model."""
        print(f"[{MODEL_NAME}] Starting training on {data} for {epochs} epochs...")
        results = self.model.train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            project=project,
            device=self.device
        )
        return results

    def close(self):
        pass


# ==============================================================================
# MAIN WEBCAM INFERENCE & TRAINING RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} - Inference & Training CLI")
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IOU threshold")
    parser.add_argument("--imgsz", type=int, default=INPUT_SIZE, help="Inference image size")
    parser.add_argument("--device", type=str, default=DEVICE, help="Inference device (cpu / cuda / 0)")
    parser.add_argument("--train", action="store_true", help="Run training instead of webcam inference")
    parser.add_argument("--dataset", type=str, default=TRAIN_DATASET_YAML, help="Dataset yaml for training")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE, help="Batch size for training")
    args = parser.parse_args()

    detector = YOLOPoseDetector(
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        imgsz=args.imgsz
    )

    if args.train:
        detector.train(data=args.dataset, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
        return

    proc = psutil.Process()
    cap, actual_cam = open_camera(
        preferred_index=args.cam,
        fallback_indices=FALLBACK_CAMERA_INDICES,
        width=CAMERA_FRAME_WIDTH,
        height=CAMERA_FRAME_HEIGHT
    )
    if cap is None:
        raise RuntimeError(f"No webcam could be opened (tested index {args.cam} and fallbacks).")

    print(f"[{MODEL_NAME}] Running on Camera {actual_cam}. Press 'q' or 'ESC' to quit.")
    fps_ema = 0.0
    prev_t  = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            _, all_persons_kps, lat_ms = detector.infer(frame)
            n_persons = detector.draw(frame, all_persons_kps, conf_thresh=args.conf)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            ram_mb = proc.memory_info().rss / 1024**2

            draw_hud(frame, MODEL_NAME, fps_ema, lat_ms, ram_mb, n_persons, extra_text=f"{args.device}")
            cv2.imshow(f"{MODEL_NAME} | Press Q to Exit", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"[{MODEL_NAME}] Cleanup complete.")


if __name__ == "__main__":
    main()
