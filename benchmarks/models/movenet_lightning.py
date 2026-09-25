"""
MoveNet Lightning - Single-person pose estimation
Backend: ai-edge-litert 2.2 / tflite-runtime / tensorflow
Model: movenet_lightning.tflite (uint8 quantized, ~4.8 MB)
Confidence-Aware Visualization: Keypoint confidence modulates landmark radius and color intensity.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import psutil

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

# ---- Backend Priority: ai-edge-litert > tflite-runtime > tensorflow ----
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
    def make_interpreter(path): return _Interpreter(model_path=path)
    DEFAULT_BACKEND = "ai-edge-litert"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite_mod
        def make_interpreter(path): return tflite_mod.Interpreter(model_path=path)
        DEFAULT_BACKEND = "tflite-runtime"
    except ImportError:
        import tensorflow as tf
        def make_interpreter(path): return tf.lite.Interpreter(model_path=path)
        DEFAULT_BACKEND = "tensorflow"

# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION CONSTANTS (ALL CAPS)
# ==============================================================================
CONFIG_FILE_PATH              = os.path.join(os.path.dirname(SCRIPT_DIR), "configs", "pose_models_config.json")
MODEL_KEY                     = "movenet_lightning"
_CFG, _GLOBAL                 = load_config(MODEL_KEY, CONFIG_FILE_PATH)

MODEL_NAME                    = _CFG.get("model_name", "MoveNet Lightning")
MODEL_PATH                    = resolve_weight_path(_CFG.get("model_file", "movenet_lightning.tflite"))
INPUT_SIZE                    = _CFG.get("input_size", 192)
SCORE_THRESHOLD               = _CFG.get("score_threshold", 0.3)
HAS_3D_DEPTH                  = _CFG.get("has_3d_depth", False)

POINT_RADIUS_MIN              = _CFG.get("point_radius_min", 3)
POINT_RADIUS_MAX              = _CFG.get("point_radius_max", 8)

DEFAULT_CAMERA_INDEX          = _GLOBAL.get("camera", {}).get("camera_index", 1)
CAMERA_FRAME_WIDTH            = _GLOBAL.get("camera", {}).get("width", 1280)
CAMERA_FRAME_HEIGHT           = _GLOBAL.get("camera", {}).get("height", 720)
FALLBACK_CAMERA_INDICES       = _GLOBAL.get("camera", {}).get("fallback_indices", [0, 1, 2, 3, 4, 5])


# ==============================================================================
# DETECTOR CLASS (EASY INFERENCE & EXTENSION)
# ==============================================================================
class MoveNetPoseDetector:
    """Wrapper class for MoveNet Single-Person Pose Estimation."""

    def __init__(self, model_path=MODEL_PATH, input_size=INPUT_SIZE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.input_size = input_size
        self.backend = DEFAULT_BACKEND
        self.interp = make_interpreter(model_path)
        self.interp.allocate_tensors()
        self.in_details = self.interp.get_input_details()
        self.out_details = self.interp.get_output_details()
        self.in_idx = self.in_details[0]['index']
        self.out_idx = self.out_details[0]['index']
        self.in_dtype = self.in_details[0]['dtype']

    def preprocess(self, frame_bgr):
        """Resize and cast input frame to match model input requirements."""
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img, axis=0)
        if self.in_dtype == np.float32:
            inp = inp.astype(np.float32) / 255.0
        elif self.in_dtype == np.int32:
            inp = inp.astype(np.int32)
        else:
            inp = inp.astype(np.uint8)
        return inp

    def infer(self, frame_bgr):
        """Run MoveNet inference and return 17 keypoints with confidence scores."""
        inp = self.preprocess(frame_bgr)
        t0 = time.perf_counter()
        self.interp.set_tensor(self.in_idx, inp)
        self.interp.invoke()
        output = self.interp.get_tensor(self.out_idx)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Output shape is [1, 1, 17, 3] with [y, x, score]
        keypoints = output[0][0]
        return keypoints, latency_ms

    def draw(self, frame_bgr, keypoints, score_thresh=SCORE_THRESHOLD,
             radius_min=POINT_RADIUS_MIN, radius_max=POINT_RADIUS_MAX):
        """Draw COCO skeleton with confidence-modulated circle sizes."""
        if keypoints is not None and len(keypoints) > 0:
            draw_coco_landmarks(
                frame_bgr, keypoints,
                connections=COCO_CONNECTIONS,
                conf_thresh=score_thresh,
                radius_min=radius_min,
                radius_max=radius_max
            )
            # Count person detected if any keypoints exceed the threshold
            valid_kps = sum(1 for kp in keypoints if float(kp[2]) >= score_thresh)
            return 1 if valid_kps >= 2 else 0
        return 0

    def close(self):
        pass


# ==============================================================================
# MAIN WEBCAM INFERENCE RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MoveNet Lightning Inference & Webcam Demo")
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index")
    parser.add_argument("--thresh", type=float, default=SCORE_THRESHOLD, help="Keypoint score threshold")
    args = parser.parse_args()

    print(f"[{MODEL_NAME}] Initializing MoveNet detector ({DEFAULT_BACKEND})...")
    detector = MoveNetPoseDetector()
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

            kps, lat_ms = detector.infer(frame)
            n_persons = detector.draw(frame, kps, score_thresh=args.thresh)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            ram_mb = proc.memory_info().rss / 1024**2

            draw_hud(frame, MODEL_NAME, fps_ema, lat_ms, ram_mb, n_persons, extra_text=f"{DEFAULT_BACKEND}")
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
