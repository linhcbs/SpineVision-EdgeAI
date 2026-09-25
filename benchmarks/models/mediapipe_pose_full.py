"""
MediaPipe Pose Landmarker - FULL (complexity = 1)
Upgraded MediaPipe Tasks Vision API (mediapipe >= 0.10)
Model: pose_landmarker_full.task (float16, ~9.4 MB, 33 3D Keypoints)
Depth-Aware Visualization: Z-coordinate modulates landmark radius and color intensity.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import psutil
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    from drawing_utils import (
        load_config, resolve_weight_path, open_camera, draw_mediapipe_landmarks_3d, draw_hud,
        COLOR_FPS, COLOR_LAT, COLOR_RAM, COLOR_MODEL
    )
except ImportError:
    from benchmarks.models.drawing_utils import (
        load_config, resolve_weight_path, open_camera, draw_mediapipe_landmarks_3d, draw_hud,
        COLOR_FPS, COLOR_LAT, COLOR_RAM, COLOR_MODEL
    )

# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION CONSTANTS (ALL CAPS)
# ==============================================================================
CONFIG_FILE_PATH              = os.path.join(os.path.dirname(SCRIPT_DIR), "configs", "pose_models_config.json")
MODEL_KEY                     = "mediapipe_pose_full"
_CFG, _GLOBAL                 = load_config(MODEL_KEY, CONFIG_FILE_PATH)

MODEL_NAME                    = _CFG.get("model_name", "MediaPipe Pose FULL (complexity=1)")
MODEL_PATH                    = resolve_weight_path(_CFG.get("task_file", "pose_landmarker_full.task"))
NUM_POSES                     = _CFG.get("num_poses", 4)
MIN_POSE_DETECTION_CONFIDENCE = _CFG.get("min_pose_detection_confidence", 0.5)
MIN_POSE_PRESENCE_CONFIDENCE  = _CFG.get("min_pose_presence_confidence", 0.5)
MIN_TRACKING_CONFIDENCE       = _CFG.get("min_tracking_confidence", 0.5)
VISIBILITY_DRAW_THRESHOLD     = _CFG.get("visibility_draw_threshold", 0.2)
HAS_3D_DEPTH                  = _CFG.get("has_3d_depth", True)

POINT_RADIUS_MIN              = _CFG.get("point_radius_min", 3)
POINT_RADIUS_MAX              = _CFG.get("point_radius_max", 10)

DEFAULT_CAMERA_INDEX          = _GLOBAL.get("camera", {}).get("camera_index", 1)
CAMERA_FRAME_WIDTH            = _GLOBAL.get("camera", {}).get("width", 1280)
CAMERA_FRAME_HEIGHT           = _GLOBAL.get("camera", {}).get("height", 720)
FALLBACK_CAMERA_INDICES       = _GLOBAL.get("camera", {}).get("fallback_indices", [0, 1, 2, 3, 4, 5])

CONNECTIONS                   = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]


# ==============================================================================
# DETECTOR CLASS (EASY INFERENCE & EXTENSION)
# ==============================================================================
class MediaPipePoseDetector:
    """Wrapper class for MediaPipe Pose Tasks API with 3D Depth Landmark estimation."""

    def __init__(self,
                 model_path=MODEL_PATH,
                 num_poses=NUM_POSES,
                 min_detection_conf=MIN_POSE_DETECTION_CONFIDENCE,
                 min_presence_conf=MIN_POSE_PRESENCE_CONFIDENCE,
                 min_tracking_conf=MIN_TRACKING_CONFIDENCE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_conf,
            min_pose_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def infer(self, frame_bgr, timestamp_ms=None):
        """Run pose detection on a BGR image frame."""
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        result = self.landmarker.detect_for_video(mp_img, timestamp_ms)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return result, latency_ms

    def draw(self, frame_bgr, result, vis_thresh=VISIBILITY_DRAW_THRESHOLD,
             radius_min=POINT_RADIUS_MIN, radius_max=POINT_RADIUS_MAX):
        """Draw 3D depth-modulated landmarks and skeleton connections on the frame."""
        n_persons = 0
        if result and result.pose_landmarks:
            n_persons = len(result.pose_landmarks)
            for lms in result.pose_landmarks:
                draw_mediapipe_landmarks_3d(
                    frame_bgr, lms, CONNECTIONS,
                    vis_thresh=vis_thresh,
                    radius_min=radius_min,
                    radius_max=radius_max
                )
        return n_persons

    def close(self):
        """Release underlying landmarker resources."""
        self.landmarker.close()


# ==============================================================================
# MAIN WEBCAM INFERENCE RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose Full Inference & Webcam Demo")
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index")
    parser.add_argument("--num-poses", type=int, default=NUM_POSES, help="Maximum poses to detect")
    parser.add_argument("--det-conf", type=float, default=MIN_POSE_DETECTION_CONFIDENCE, help="Min detection confidence")
    parser.add_argument("--vis-thresh", type=float, default=VISIBILITY_DRAW_THRESHOLD, help="Draw threshold")
    args = parser.parse_args()

    print(f"[{MODEL_NAME}] Initializing detector...")
    detector = MediaPipePoseDetector(
        num_poses=args.num_poses,
        min_detection_conf=args.det_conf
    )
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

            res, lat_ms = detector.infer(frame)
            n_persons = detector.draw(frame, res, vis_thresh=args.vis_thresh)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            ram_mb = proc.memory_info().rss / 1024**2

            draw_hud(frame, MODEL_NAME, fps_ema, lat_ms, ram_mb, n_persons, extra_text="3D Depth Enabled")
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
