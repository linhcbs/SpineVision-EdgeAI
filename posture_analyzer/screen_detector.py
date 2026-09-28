"""
Screen Device Detection Module.

Detects electronic screen devices (phone, tablet, laptop, monitor, TV) in the webcam
frame using YOLO11n object detection. Falls back to a lightweight heuristic webcam-only
mode when YOLO is unavailable.

Supports up to N detected devices per frame (configurable, default = 3, includes webcam).
Real-time updates: devices are added/removed each frame based on current detection.

User orientation contexts:
  - FRONTAL (user faces camera): Screens likely in front — webcam is in user's gaze
  - OBLIQUE/PROFILE (user faces sideways): Primary screen is likely NOT the webcam
    (e.g. user stares at laptop while surveillance webcam is to their side)

All config keys are read from configs/posture_analyzer_config.json under 'screen_detection'.
"""

import math
import os
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from .types import DetectedDevice, DeviceType, ViewMode, UnifiedKeypoints, NormalizedKeypoint
from .config import get_posture_config

# ─── YOLO class IDs that correspond to screens ─────────────────────────────────
# COCO80 class list subset relevant to screens / electronics:
#   63 = laptop, 66 = keyboard, 67 = cell phone, 72 = tv, 65 = remote,
# We handle laptop(63), cell_phone(67), tv(72) as screen devices.
# Tablets are often classified as cell phone or laptop by open-vocabulary YOLO.
_YOLO_SCREEN_CLASSES: Dict[int, DeviceType] = {
    63: DeviceType.LAPTOP,
    67: DeviceType.PHONE,
    72: DeviceType.TV,
}
_YOLO_SCREEN_CLASS_IDS = set(_YOLO_SCREEN_CLASSES.keys())

# Physical reference sizes (cm), used with pixel width for distance estimation
_DEVICE_REAL_WIDTH_CM: Dict[DeviceType, float] = {
    DeviceType.PHONE:   7.5,   # Avg smartphone width
    DeviceType.TABLET:  20.0,  # Avg tablet width
    DeviceType.LAPTOP:  35.0,  # Avg 15-inch laptop screen width
    DeviceType.MONITOR: 50.0,  # Avg 24-inch monitor width
    DeviceType.TV:      80.0,  # Avg 40-inch TV width
    DeviceType.WEBCAM:  0.0,   # Computed via IPD instead
    DeviceType.UNKNOWN: 30.0,  # Conservative default
}

# BGR colors per device type for HUD
_DEVICE_COLORS: Dict[DeviceType, Tuple[int, int, int]] = {
    DeviceType.WEBCAM:  (80, 220, 80),     # Green
    DeviceType.LAPTOP:  (0, 200, 255),     # Yellow
    DeviceType.PHONE:   (255, 140, 0),     # Blue
    DeviceType.TABLET:  (255, 80, 200),    # Magenta
    DeviceType.MONITOR: (0, 255, 200),     # Cyan-green
    DeviceType.TV:      (180, 100, 255),   # Purple
    DeviceType.UNKNOWN: (160, 160, 160),   # Gray
}


def get_device_color(dtype: DeviceType) -> Tuple[int, int, int]:
    return _DEVICE_COLORS.get(dtype, (160, 160, 160))


class ScreenDetector:
    """
    Detects electronic screens in a video frame and estimates eye-to-screen distance
    for each detected device independently.

    Detection strategy:
      1. YOLO11n object detector (primary) — detects laptop, phone, TV
      2. Webcam heuristic (always appended) — the camera itself is always a screen
         whose distance is computed via the IPD pinhole model

    Args:
        model_path: Path to YOLO11n .pt weights file.
        max_devices: Maximum screens to track (including webcam). Default 3.
        min_confidence: Minimum YOLO box confidence. Default 0.40.
        iou_threshold: NMS IoU threshold. Default 0.50.
        focal_length_px: Fallback focal length when not calibrated.
        ipd_real_cm: Physical interpupillary distance in cm (default 6.3).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_devices: Optional[int] = None,
        min_confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        focal_length_px: Optional[float] = None,
        ipd_real_cm: Optional[float] = None,
    ):
        cfg = get_posture_config().get("screen_detection", {})
        metric_cfg = get_posture_config().get("metrics", {}).get("eye_to_screen_distance", {})

        self.max_devices = max_devices if max_devices is not None else cfg.get("max_devices", 3)
        self.min_confidence = min_confidence if min_confidence is not None else cfg.get("min_confidence", 0.40)
        self.iou_threshold = iou_threshold if iou_threshold is not None else cfg.get("iou_threshold", 0.50)
        self.focal_length_px = focal_length_px if focal_length_px is not None else metric_cfg.get("default_focal_length_px", 650.0)
        self.ipd_real_cm = ipd_real_cm if ipd_real_cm is not None else metric_cfg.get("average_ipd_cm", 6.3)
        self.enabled = cfg.get("enabled", True)
        self.draw_boxes = cfg.get("draw_boxes", True)
        self.box_opacity = cfg.get("box_opacity", 0.18)

        # Resolve model path
        if model_path is None:
            model_path = cfg.get("model_path", "")
        if not model_path or not os.path.exists(model_path):
            # Try common locations relative to this file
            _here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(_here, "..", "models", "weights", "yolo11n.pt"),
                os.path.join(_here, "..", "models", "yolo11n.pt"),
                os.path.join(_here, "..", "yolo11n.pt"),
                "models/weights/yolo11n.pt",
                "yolo11n.pt",
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)

        self._yolo = None
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(model_path, verbose=False)
                print(f"[ScreenDetector] YOLO11n loaded from: {model_path}")
            except Exception as e:
                print(f"[ScreenDetector] YOLO load failed ({e}). Using heuristic mode.")
        else:
            print("[ScreenDetector] No YOLO weights found. Using heuristic (webcam-only) mode.")

        self._focal_calibrated: bool = False
        # Smoothing: keep last N detections per label for temporal stability
        self._prev_devices: List[DetectedDevice] = []

    # ────────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────────

    def calibrate_focal_length(self, ipd_pixels: float, known_distance_cm: float):
        """Update focal length from calibration data (same as DistanceEstimator)."""
        if ipd_pixels > 1.0:
            self.focal_length_px = (ipd_pixels * known_distance_cm) / self.ipd_real_cm
            self._focal_calibrated = True

    def detect(
        self,
        frame: np.ndarray,
        kps: Optional[UnifiedKeypoints] = None,
        view_mode: ViewMode = ViewMode.FRONTAL,
    ) -> List[DetectedDevice]:
        """
        Detect electronic screens and compute per-device eye-to-screen distances.

        Args:
            frame: BGR video frame.
            kps: Current pose keypoints (used for IPD distance and eye position).
            view_mode: Current user orientation relative to camera.

        Returns:
            List of DetectedDevice, at most `max_devices` items, always including
            the webcam entry at position 0.
        """
        if not self.enabled:
            return []

        h, w = frame.shape[:2]
        devices: List[DetectedDevice] = []

        # ── 1. IPD-based eye distance (for webcam / frontal distance) ──────────
        ipd_px = self._measure_ipd(kps)
        eye_y = self._get_eye_y(kps)
        webcam_dist = -1.0
        if ipd_px is not None and ipd_px > 3.0:
            webcam_dist = round(
                max(15.0, min(200.0, (self.focal_length_px * self.ipd_real_cm) / ipd_px)), 1
            )

        # ── 2. YOLO detection ─────────────────────────────────────────────────
        yolo_devices: List[DetectedDevice] = []
        if self._yolo is not None:
            try:
                yolo_devices = self._run_yolo(frame, kps, ipd_px, h, w)
            except Exception as e:
                print(f"[ScreenDetector] YOLO inference error: {e}")

        # ── 3. Build user-context-aware device list ───────────────────────────
        # Webcam is always included as an implicit "screen"
        # Its distance = IPD-estimated frontal distance (reliable in FRONTAL mode)
        webcam_label = "Webcam"
        webcam_entry = DetectedDevice(
            device_type=DeviceType.WEBCAM,
            bbox=None,
            center=(w / 2.0, 0.0),   # Camera is at the top center of the frame
            confidence=1.0,
            distance_cm=webcam_dist,
            label=webcam_label,
            source="fixed",
        )
        devices.append(webcam_entry)

        # Add YOLO-detected screens (sorted by confidence), up to max_devices-1 slots
        yolo_devices.sort(key=lambda d: d.confidence, reverse=True)
        for dev in yolo_devices:
            if len(devices) >= self.max_devices:
                break
            devices.append(dev)

        # Keep list order stable: webcam first, then others by confidence
        self._prev_devices = devices
        return devices

    def draw_devices(self, frame: np.ndarray, devices: List[DetectedDevice]):
        """
        Draw bounding boxes and distance labels for detected devices on the frame.
        Only YOLO-detected devices get boxes; webcam gets a small corner marker.
        """
        if not self.draw_boxes:
            return

        h, w = frame.shape[:2]

        for dev in devices:
            color = get_device_color(dev.device_type)

            if dev.bbox is not None:
                x1, y1, x2, y2 = dev.bbox
                # Semi-transparent fill
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, self.box_opacity, frame, 1.0 - self.box_opacity, 0, frame)
                # Solid border
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                # Corner brackets
                blen = 12
                for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(frame, (cx, cy), (cx + blen * dx, cy), color, 2)
                    cv2.line(frame, (cx, cy), (cx, cy + blen * dy), color, 2)

                # Label background
                dist_str = f"{dev.distance_cm:.0f}cm" if dev.distance_cm > 0 else "?cm"
                label_text = f"{dev.label} | {dist_str}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                lx, ly = x1, max(y1 - 6, th + 4)
                cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 8, ly + 2), (20, 20, 20), -1)
                cv2.putText(frame, label_text, (lx + 4, ly - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            else:
                # Webcam: draw a small camera icon at top-center
                cam_cx = w // 2
                cam_cy = 18
                cv2.circle(frame, (cam_cx, cam_cy), 10, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cam_cx, cam_cy), 4, color, -1, cv2.LINE_AA)
                dist_str = f"{dev.distance_cm:.0f}cm" if dev.distance_cm > 0 else "?cm"
                cv2.putText(frame, f"Webcam {dist_str}", (cam_cx + 14, cam_cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────────────────

    def _run_yolo(
        self,
        frame: np.ndarray,
        kps: Optional[UnifiedKeypoints],
        ipd_px: Optional[float],
        h: int, w: int,
    ) -> List[DetectedDevice]:
        """Run YOLO11n, filter screen-class boxes, compute per-device distances."""
        results = self._yolo(frame, classes=list(_YOLO_SCREEN_CLASS_IDS),
                              conf=self.min_confidence, iou=self.iou_threshold,
                              verbose=False, stream=False)
        devices: List[DetectedDevice] = []
        type_counters: Dict[DeviceType, int] = {}

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                dtype = _YOLO_SCREEN_CLASSES.get(cls_id, DeviceType.UNKNOWN)
                type_counters[dtype] = type_counters.get(dtype, 0) + 1
                idx = type_counters[dtype]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box_w = max(x2 - x1, 1)

                # Distance via pinhole: D = (f * W_real) / W_pixel
                real_w = _DEVICE_REAL_WIDTH_CM.get(dtype, 30.0)
                dist_cm = -1.0
                if real_w > 0 and ipd_px is not None and ipd_px > 3.0:
                    # Use IPD-derived focal for consistency
                    dist_cm = round(
                        max(10.0, min(400.0, (self.focal_length_px * real_w) / box_w)), 1
                    )

                # Generate a human-readable label
                type_name = dtype.value.capitalize()
                label = f"{type_name} #{idx}" if idx > 1 else type_name

                devices.append(DetectedDevice(
                    device_type=dtype,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    confidence=conf,
                    distance_cm=dist_cm,
                    label=label,
                    source="yolo",
                ))

        return devices

    def _measure_ipd(self, kps: Optional[UnifiedKeypoints]) -> Optional[float]:
        """Measure IPD in pixels from eye keypoints."""
        if kps is None:
            return None
        l_eye = kps.get("left_eye")
        r_eye = kps.get("right_eye")
        if l_eye is None or r_eye is None:
            return None
        if l_eye.score < 0.25 or r_eye.score < 0.25:
            return None
        dx = l_eye.x - r_eye.x
        dy = l_eye.y - r_eye.y
        ipd = math.sqrt(dx * dx + dy * dy)
        return ipd if ipd > 3.0 else None

    def _get_eye_y(self, kps: Optional[UnifiedKeypoints]) -> Optional[float]:
        """Get eye midpoint y in pixels."""
        if kps is None:
            return None
        l = kps.get("left_eye")
        r = kps.get("right_eye")
        if l and r and l.score >= 0.25 and r.score >= 0.25:
            return (l.y + r.y) / 2.0
        nose = kps.get("nose")
        if nose and nose.score >= 0.25:
            return nose.y
        return None
