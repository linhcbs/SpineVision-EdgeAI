"""
YOLO11n (Nano) - Real-Time Object & Screen Detection Pipeline
Architecture: Ultralytics YOLO11 Nano (v11)
Weights: yolo11n.pt (~5.6 MB, COCO 80 classes)

Features:
- Real-time webcam object detection & bounding box visualization
- Comprehensive RAM consumption measurement & profiling (RSS, VMS, Peak RAM, Delta MB)
- HUD Overlay (FPS, Latency, RAM MB, Detection Count, Device)
- Clean CLI for inference, threshold tuning, and benchmark profiling
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
        load_config, resolve_weight_path, open_camera, draw_hud,
        COLOR_FPS, COLOR_LAT, COLOR_RAM, COLOR_MODEL
    )
except ImportError:
    from benchmarks.models.drawing_utils import (
        load_config, resolve_weight_path, open_camera, draw_hud,
        COLOR_FPS, COLOR_LAT, COLOR_RAM, COLOR_MODEL
    )

# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION CONSTANTS
# ==============================================================================
CONFIG_FILE_PATH              = os.path.join(os.path.dirname(SCRIPT_DIR), "configs", "pose_models_config.json")
MODEL_KEY                     = "yolo11n"
_CFG, _GLOBAL                 = load_config(MODEL_KEY, CONFIG_FILE_PATH)

MODEL_NAME                    = _CFG.get("model_name", "YOLO11n (Nano Object Detector)")
_WEIGHTS_FILE                 = _CFG.get("weights_file", "yolo11n.pt")
WEIGHTS_PATH                  = resolve_weight_path(_WEIGHTS_FILE)

INPUT_SIZE                    = _CFG.get("input_size", 640)
CONF_THRESHOLD                = _CFG.get("conf_threshold", 0.35)
IOU_THRESHOLD                 = _CFG.get("iou_threshold", 0.45)
DEVICE                        = _CFG.get("device", "cpu")
HALF_PRECISION                = _CFG.get("half_precision", False)

DEFAULT_CAMERA_INDEX          = _GLOBAL.get("camera", {}).get("camera_index", 1)
CAMERA_FRAME_WIDTH            = _GLOBAL.get("camera", {}).get("width", 1280)
CAMERA_FRAME_HEIGHT           = _GLOBAL.get("camera", {}).get("height", 720)
FALLBACK_CAMERA_INDICES       = _GLOBAL.get("camera", {}).get("fallback_indices", [0, 1, 2, 3, 4, 5])


# ==============================================================================
# RAM PROFILER & MEMORY UTILITIES
# ==============================================================================
class RAMProfiler:
    """Utility to measure process RSS, VMS, and system-wide memory footprint."""

    def __init__(self):
        self.proc = psutil.Process()
        self.peak_rss_mb = 0.0
        self.initial_rss_mb = self.get_rss_mb()
        self.initial_vms_mb = self.get_vms_mb()

    def get_rss_mb(self) -> float:
        """Returns Resident Set Size (RSS) in MB."""
        rss = self.proc.memory_info().rss / (1024 ** 2)
        if rss > self.peak_rss_mb:
            self.peak_rss_mb = rss
        return rss

    def get_vms_mb(self) -> float:
        """Returns Virtual Memory Size (VMS) in MB."""
        return self.proc.memory_info().vms / (1024 ** 2)

    def get_system_ram_gb(self) -> dict:
        """Returns total, available, and used system RAM in GB."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_gb": round((mem.total - mem.available) / (1024 ** 3), 2),
            "percent_used": mem.percent
        }


# ==============================================================================
# DETECTOR CLASS WITH INFERENCE & RAM MEASUREMENT
# ==============================================================================
class YOLO11Detector:
    """Wrapper class for Ultralytics YOLO11 object detection and RAM profiling."""

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
        
        self.profiler = RAMProfiler()
        self.ram_before_load_mb = self.profiler.get_rss_mb()

        print(f"[{MODEL_NAME}] Baseline Process RAM: {self.ram_before_load_mb:.2f} MB")
        print(f"[{MODEL_NAME}] Loading YOLO11 model weights from {self.weights_path}...")
        
        t0 = time.perf_counter()
        self.model = YOLO(self.weights_path)
        load_time_ms = (time.perf_counter() - t0) * 1000.0

        self.ram_after_load_mb = self.profiler.get_rss_mb()
        self.model_ram_delta_mb = self.ram_after_load_mb - self.ram_before_load_mb

        print(f"[{MODEL_NAME}] Model loaded in {load_time_ms:.1f} ms.")
        print(f"[{MODEL_NAME}] Post-Load Process RAM: {self.ram_after_load_mb:.2f} MB "
              f"(Model RAM footprint: +{self.model_ram_delta_mb:.2f} MB)")

    def infer(self, frame_bgr, classes=None):
        """
        Run YOLO11 object detection on a BGR frame.
        Returns: (results, list_of_detections, latency_ms)
        """
        t0 = time.perf_counter()
        kwargs = {
            "verbose": False,
            "conf": self.conf_thresh,
            "iou": self.iou_thresh,
            "device": self.device,
            "imgsz": self.imgsz,
        }
        if classes is not None:
            kwargs["classes"] = classes
        if self.half and str(self.device).lower() not in ["cpu", ""]:
            kwargs["half"] = True

        results = self.model(frame_bgr, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                label = names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                })

        return results, detections, latency_ms

    def draw(self, frame_bgr, detections):
        """Draw bounding boxes, confidence labels, and class tags."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            cls_id = det["class_id"]

            # Deterministic color scheme per class
            np.random.seed(cls_id * 31 + 7)
            color = tuple(map(int, np.random.randint(60, 255, size=3)))

            # Semi-transparent box overlay
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame_bgr, 0.85, 0, frame_bgr)

            # Solid bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Text label badge
            caption = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ly = max(y1 - 6, th + 4)
            cv2.rectangle(frame_bgr, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), (20, 20, 20), -1)
            cv2.putText(frame_bgr, caption, (x1 + 3, ly - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        return len(detections)

    def close(self):
        """Teardown detector."""
        pass


# ==============================================================================
# MAIN PIPELINE & WEBCAM RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} - Inference & RAM Profiling Pipeline")
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IOU threshold")
    parser.add_argument("--imgsz", type=int, default=INPUT_SIZE, help="Inference image size")
    parser.add_argument("--device", type=str, default=DEVICE, help="Inference device (cpu / cuda / 0)")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Specific COCO class IDs to detect")
    parser.add_argument("--benchmark", type=int, default=0, help="Run automatic benchmark mode for N frames")
    args = parser.parse_args()

    detector = YOLO11Detector(
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        imgsz=args.imgsz
    )

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
    latencies = []
    ram_samples = []

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            _, detections, lat_ms = detector.infer(frame, classes=args.classes)
            n_dets = detector.draw(frame, detections)
            latencies.append(lat_ms)

            current_ram_mb = detector.profiler.get_rss_mb()
            ram_samples.append(current_ram_mb)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps if fps_ema > 0 else inst_fps

            draw_hud(frame, MODEL_NAME, fps_ema, lat_ms, current_ram_mb, n_dets, extra_text=f"RAM+:{detector.model_ram_delta_mb:.1f}MB")
            cv2.imshow(f"{MODEL_NAME} | Press Q to Exit", frame)

            frame_count += 1
            if args.benchmark > 0 and frame_count >= args.benchmark:
                print(f"[{MODEL_NAME}] Benchmark completed {frame_count} frames.")
                break

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        # Detailed RAM and Performance Summary Report
        avg_lat = np.mean(latencies) if latencies else 0.0
        avg_ram = np.mean(ram_samples) if ram_samples else detector.ram_after_load_mb
        peak_ram = detector.profiler.peak_rss_mb
        sys_ram = detector.profiler.get_system_ram_gb()
        model_size_mb = os.path.getsize(detector.weights_path) / (1024 ** 2) if os.path.exists(detector.weights_path) else 0.0

        print("\n" + "=" * 70)
        print(f"      {MODEL_NAME} — RAM CONSUMPTION & PERFORMANCE REPORT")
        print("=" * 70)
        print(f"  Weights File           : {detector.weights_path}")
        print(f"  Model File Size        : {model_size_mb:.2f} MB")
        print(f"  Inference Device       : {args.device.upper()}")
        print(f"  Image Resolution       : {args.imgsz}x{args.imgsz}")
        print("-" * 70)
        print(f"  Baseline RAM (Pre-Load): {detector.ram_before_load_mb:.2f} MB")
        print(f"  Loaded RAM (Post-Load) : {detector.ram_after_load_mb:.2f} MB")
        print(f"  Model RAM Impact       : +{detector.model_ram_delta_mb:.2f} MB")
        print(f"  Runtime Average RAM    : {avg_ram:.2f} MB")
        print(f"  Runtime Peak RAM (RSS) : {peak_ram:.2f} MB")
        print(f"  System RAM Usage       : {sys_ram['used_gb']} GB / {sys_ram['total_gb']} GB ({sys_ram['percent_used']}%)")
        print("-" * 70)
        print(f"  Average Latency        : {avg_lat:.2f} ms")
        print(f"  Average FPS            : {1000.0 / avg_lat:.1f} FPS (Inference)")
        print(f"  Total Processed Frames : {frame_count}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
