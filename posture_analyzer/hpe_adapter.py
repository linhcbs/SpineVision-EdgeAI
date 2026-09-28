"""
Universal Human Pose Estimation (HPE) Model Adapter & Factory.
SpineVision EdgeAI — Plug-and-Play HPE Architecture

Supports all 7 core HPE models:
  1. MediaPipe Pose LITE   (33 3D landmarks, float16, ultra-fast)
  2. MediaPipe Pose FULL   (33 3D landmarks, float16, high accuracy)
  3. MoveNet Lightning     (17 COCO keypoints, 192x192, quantized)
  4. MoveNet Thunder       (17 COCO keypoints, 256x256, quantized)
  5. YOLO26n-pose (Nano)   (17 COCO keypoints, multi-person edge)
  6. YOLO26m-pose (Medium) (17 COCO keypoints, balanced multi-person)
  7. YOLO26x-pose (XLarge) (17 COCO keypoints, maximum accuracy)

All adapters convert raw detector outputs into standardized `UnifiedKeypoints`
so that downstream modules (biometrics, calibration, classification, risk
evaluation, HUD) remain 100% model-agnostic.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Local imports
from .types import (
    KeypointFormat,
    UnifiedKeypoints,
    convert_mediapipe_to_unified,
    convert_coco_to_unified,
)

# Benchmark models directory resolution
BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
MODELS_DIR = os.path.join(BENCHMARKS_DIR, "models")
WEIGHTS_DIR = os.path.join(BENCHMARKS_DIR, "weights")

if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)


def _resolve_weight(filename: str) -> str:
    """Resolve absolute path to model weights file."""
    candidates = [
        os.path.join(WEIGHTS_DIR, filename),
        os.path.join(MODELS_DIR, filename),
        os.path.join(os.path.dirname(WEIGHTS_DIR), filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(WEIGHTS_DIR, filename)


# ==============================================================================
# BASE HPE ADAPTER INTERFACE
# ==============================================================================

class BaseHPEAdapter(ABC):
    """Abstract Base Class for Plug-and-Play Pose Estimation Adapters."""

    def __init__(self, model_key: str, model_name: str, keypoint_format: KeypointFormat, has_3d_depth: bool = False):
        self.model_key = model_key
        self.model_name = model_name
        self.keypoint_format = keypoint_format
        self.has_3d_depth = has_3d_depth

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Tuple[Any, float]:
        """
        Run inference on a single BGR video frame.
        
        Args:
            frame_bgr: BGR video frame as uint8 numpy array.
            timestamp_ms: Optional monotonic timestamp in milliseconds.
            
        Returns:
            Tuple of (raw_result, latency_ms).
        """
        pass

    @abstractmethod
    def to_unified(self, raw_result: Any, frame_width: int, frame_height: int) -> Optional[UnifiedKeypoints]:
        """
        Convert raw model output to standardized UnifiedKeypoints.
        
        Args:
            raw_result: The raw output returned by infer().
            frame_width: Width of video frame in pixels.
            frame_height: Height of video frame in pixels.
            
        Returns:
            UnifiedKeypoints or None if no pose was detected.
        """
        pass

    @abstractmethod
    def draw(self, frame_bgr: np.ndarray, raw_result: Any) -> int:
        """
        Draw skeleton landmarks on frame_bgr in-place.
        
        Returns:
            Number of detected persons.
        """
        pass

    def close(self):
        """Release detector resources."""
        pass


# ==============================================================================
# 1 & 2. MEDIAPIPE POSE ADAPTER (Lite & Full)
# ==============================================================================

class MediaPipePoseAdapter(BaseHPEAdapter):
    """Adapter for MediaPipe Pose (Lite / Full) — 33 3D Landmarks."""

    def __init__(self, model_path: Optional[str] = None, is_lite: bool = False, num_poses: int = 1):
        weight_file = "pose_landmarker_lite.task" if is_lite else "pose_landmarker_full.task"
        resolved_path = model_path or _resolve_weight(weight_file)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"MediaPipe weight file not found: {resolved_path}")

        key = "mediapipe_pose_lite" if is_lite else "mediapipe_pose_full"
        name = "MediaPipe Pose LITE (33 3D)" if is_lite else "MediaPipe Pose FULL (33 3D)"
        super().__init__(model_key=key, model_name=name, keypoint_format=KeypointFormat.MEDIAPIPE_33, has_3d_depth=True)

        try:
            from benchmarks.models.mediapipe_pose_full import MediaPipePoseDetector
        except ImportError:
            from models.mediapipe_pose_full import MediaPipePoseDetector

        self.detector = MediaPipePoseDetector(model_path=resolved_path, num_poses=num_poses)
        self._last_ts_ms = 0

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Tuple[Any, float]:
        now_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms

        result, latency_ms = self.detector.infer(frame_bgr, timestamp_ms=now_ms)
        return result, latency_ms

    def to_unified(self, raw_result: Any, frame_width: int, frame_height: int) -> Optional[UnifiedKeypoints]:
        if raw_result and getattr(raw_result, 'pose_landmarks', None) and len(raw_result.pose_landmarks) > 0:
            landmarks = raw_result.pose_landmarks[0]
            return convert_mediapipe_to_unified(landmarks, frame_width, frame_height)
        return None

    def draw(self, frame_bgr: np.ndarray, raw_result: Any) -> int:
        if raw_result and getattr(raw_result, 'pose_landmarks', None) and len(raw_result.pose_landmarks) > 0:
            self.detector.draw(frame_bgr, raw_result, vis_thresh=0.40, radius_min=3, radius_max=6)
            return len(raw_result.pose_landmarks)
        return 0

    def close(self):
        self.detector.close()


# ==============================================================================
# 3 & 4. MOVENET ADAPTER (Lightning & Thunder)
# ==============================================================================

class MoveNetPoseAdapter(BaseHPEAdapter):
    """Adapter for MoveNet (Lightning 192x192 / Thunder 256x256) — 17 COCO Keypoints."""

    def __init__(self, model_path: Optional[str] = None, is_thunder: bool = False):
        weight_file = "movenet_thunder.tflite" if is_thunder else "movenet_lightning.tflite"
        resolved_path = model_path or _resolve_weight(weight_file)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"MoveNet weight file not found: {resolved_path}")

        input_size = 256 if is_thunder else 192
        key = "movenet_thunder" if is_thunder else "movenet_lightning"
        name = "MoveNet Thunder (256x256)" if is_thunder else "MoveNet Lightning (192x192)"
        super().__init__(model_key=key, model_name=name, keypoint_format=KeypointFormat.COCO_17, has_3d_depth=False)

        try:
            from benchmarks.models.movenet_lightning import MoveNetPoseDetector
        except ImportError:
            from models.movenet_lightning import MoveNetPoseDetector

        self.detector = MoveNetPoseDetector(model_path=resolved_path, input_size=input_size)

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Tuple[Any, float]:
        keypoints, latency_ms = self.detector.infer(frame_bgr)
        return keypoints, latency_ms

    def to_unified(self, raw_result: Any, frame_width: int, frame_height: int) -> Optional[UnifiedKeypoints]:
        if raw_result is not None and len(raw_result) == 17:
            # Check minimum confidence across keypoints
            valid_pts = sum(1 for kp in raw_result if float(kp[2]) > 0.15)
            if valid_pts >= 3:
                return convert_coco_to_unified(
                    raw_result,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    is_normalized=True,
                    is_yx_order=True,
                )
        return None

    def draw(self, frame_bgr: np.ndarray, raw_result: Any) -> int:
        return self.detector.draw(frame_bgr, raw_result, score_thresh=0.25)

    def close(self):
        self.detector.close()


# ==============================================================================
# 5, 6 & 7. YOLO-POSE ADAPTER (Nano, Medium, XLarge, v8n)
# ==============================================================================

class YOLOPoseAdapter(BaseHPEAdapter):
    """Adapter for YOLO-Pose (yolo26n, yolo26m, yolo26x, yolov8n) — 17 COCO Keypoints."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "nano",
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        variant_map = {
            "nano": ("yolo26n_pose", "yolo26n-pose.pt", "YOLO-Pose Nano (17 COCO)"),
            "medium": ("yolo26m_pose", "yolo26m-pose.pt", "YOLO-Pose Medium (17 COCO)"),
            "xlarge": ("yolo26x_pose", "yolo26x-pose.pt", "YOLO-Pose XLarge (17 COCO)"),
            "v8n": ("yolov8n_pose", "yolov8n-pose.pt", "YOLOv8n-Pose (17 COCO)"),
        }
        key, def_weight, name = variant_map.get(variant.lower(), variant_map["nano"])
        resolved_path = model_path or _resolve_weight(def_weight)
        if not os.path.exists(resolved_path):
            # Fallback to yolov8n-pose or yolo26n-pose
            resolved_path = _resolve_weight("yolo26n-pose.pt")
            if not os.path.exists(resolved_path):
                resolved_path = _resolve_weight("yolov8n-pose.pt")

        super().__init__(model_key=key, model_name=name, keypoint_format=KeypointFormat.COCO_17, has_3d_depth=False)

        try:
            from benchmarks.models.yolo26n_pose import YOLOPoseDetector
        except ImportError:
            from models.yolo26n_pose import YOLOPoseDetector

        self.detector = YOLOPoseDetector(
            weights_path=resolved_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=device,
            imgsz=imgsz,
        )

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Tuple[Any, float]:
        results, all_persons_kps, latency_ms = self.detector.infer(frame_bgr)
        # Package (results, all_persons_kps) as raw_result
        return (results, all_persons_kps), latency_ms

    def to_unified(self, raw_result: Any, frame_width: int, frame_height: int) -> Optional[UnifiedKeypoints]:
        if raw_result is None:
            return None
        results, all_persons_kps = raw_result
        if all_persons_kps and len(all_persons_kps) > 0:
            # Primary subject (first detected person)
            person_kps = all_persons_kps[0]
            if len(person_kps) == 17:
                return convert_coco_to_unified(
                    person_kps,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    is_normalized=False,
                    is_yx_order=False,
                )
        return None

    def draw(self, frame_bgr: np.ndarray, raw_result: Any) -> int:
        if raw_result is None:
            return 0
        _, all_persons_kps = raw_result
        return self.detector.draw(frame_bgr, all_persons_kps, conf_thresh=0.35)

    def close(self):
        self.detector.close()


# ==============================================================================
# HPE MODEL REGISTRY & FACTORY FUNCTION
# ==============================================================================

SUPPORTED_HPE_MODELS: Dict[str, Dict[str, Any]] = {
    # 1. MediaPipe Pose Lite
    "mediapipe_pose_lite": {
        "name": "MediaPipe Pose LITE",
        "description": "BlazePose Lite (complexity=0), 33 3D landmarks, ultra-fast CPU inference",
        "keypoints": 33,
        "has_depth": True,
        "weight_file": "pose_landmarker_lite.task",
        "aliases": ["mp_lite", "lite", "mediapipe_lite"],
    },
    # 2. MediaPipe Pose Full
    "mediapipe_pose_full": {
        "name": "MediaPipe Pose FULL",
        "description": "BlazePose Full (complexity=1), 33 3D landmarks, high ergonomic precision",
        "keypoints": 33,
        "has_depth": True,
        "weight_file": "pose_landmarker_full.task",
        "aliases": ["mp_full", "full", "mediapipe", "mediapipe_full"],
    },
    # 3. MoveNet Lightning
    "movenet_lightning": {
        "name": "MoveNet Lightning",
        "description": "MoveNet Lightning (192x192), 17 COCO keypoints, quantized low-latency",
        "keypoints": 17,
        "has_depth": False,
        "weight_file": "movenet_lightning.tflite",
        "aliases": ["lightning", "movenet_light", "movenet192"],
    },
    # 4. MoveNet Thunder
    "movenet_thunder": {
        "name": "MoveNet Thunder",
        "description": "MoveNet Thunder (256x256), 17 COCO keypoints, high-accuracy single person",
        "keypoints": 17,
        "has_depth": False,
        "weight_file": "movenet_thunder.tflite",
        "aliases": ["thunder", "movenet_thun", "movenet256"],
    },
    # 5. YOLO-Pose Nano
    "yolo26n_pose": {
        "name": "YOLO26n-pose (Nano)",
        "description": "YOLO-Pose Nano architecture, 17 COCO keypoints, multi-person edge monitoring",
        "keypoints": 17,
        "has_depth": False,
        "weight_file": "yolo26n-pose.pt",
        "aliases": ["yolo_nano", "yolo_n", "yolon", "yolov8n_pose", "yolo11n_pose"],
    },
    # 6. YOLO-Pose Medium
    "yolo26m_pose": {
        "name": "YOLO26m-pose (Medium)",
        "description": "YOLO-Pose Medium architecture, 17 COCO keypoints, balanced speed & accuracy",
        "keypoints": 17,
        "has_depth": False,
        "weight_file": "yolo26m-pose.pt",
        "aliases": ["yolo_medium", "yolo_m", "yolom", "yolo11m_pose"],
    },
    # 7. YOLO-Pose XLarge
    "yolo26x_pose": {
        "name": "YOLO26x-pose (XLarge)",
        "description": "YOLO-Pose Extra-Large architecture, 17 COCO keypoints, maximum precision",
        "keypoints": 17,
        "has_depth": False,
        "weight_file": "yolo26x-pose.pt",
        "aliases": ["yolo_xlarge", "yolo_x", "yolox"],
    },
}


def list_supported_hpe_models() -> List[Dict[str, Any]]:
    """Return a list of metadata for all 7 supported HPE models."""
    models = []
    for key, info in SUPPORTED_HPE_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "keypoints": info["keypoints"],
            "has_depth": info["has_depth"],
            "aliases": info["aliases"],
        })
    return models


def canonical_model_key(name_or_alias: str) -> str:
    """Resolve any model name, key, or alias to canonical model_key."""
    query = str(name_or_alias).lower().strip().replace("-", "_")
    if query in SUPPORTED_HPE_MODELS:
        return query
    for canon_key, meta in SUPPORTED_HPE_MODELS.items():
        if query == canon_key or query == canon_key.replace("_pose", ""):
            return canon_key
        if query in [a.lower().replace("-", "_") for a in meta.get("aliases", [])]:
            return canon_key
    return "mediapipe_pose_full"


def create_hpe_detector(
    model_name_or_key: str = "mediapipe_pose_full",
    weights_path: Optional[str] = None,
    device: str = "cpu",
    conf_thresh: Optional[float] = None,
    **kwargs,
) -> BaseHPEAdapter:
    """
    Factory function: Instantiates any of the 7 supported HPE models as a plug-and-play adapter.
    
    Args:
        model_name_or_key: Canonical model key or alias (e.g., 'mediapipe_pose_full',
                           'movenet_lightning', 'yolo26n_pose', 'lite', 'thunder', etc.)
        weights_path: Optional custom weight file path override.
        device: Device for YOLO inference ('cpu' or 'cuda').
        conf_thresh: Keypoint/detection confidence threshold.
        
    Returns:
        BaseHPEAdapter instance ready for infer() and to_unified().
    """
    canon_key = canonical_model_key(model_name_or_key)

    if canon_key == "mediapipe_pose_lite":
        return MediaPipePoseAdapter(model_path=weights_path, is_lite=True, **kwargs)

    elif canon_key == "mediapipe_pose_full":
        return MediaPipePoseAdapter(model_path=weights_path, is_lite=False, **kwargs)

    elif canon_key == "movenet_lightning":
        return MoveNetPoseAdapter(model_path=weights_path, is_thunder=False)

    elif canon_key == "movenet_thunder":
        return MoveNetPoseAdapter(model_path=weights_path, is_thunder=True)

    elif canon_key == "yolo26n_pose":
        ct = conf_thresh if conf_thresh is not None else 0.4
        return YOLOPoseAdapter(model_path=weights_path, variant="nano", conf_thresh=ct, device=device, **kwargs)

    elif canon_key == "yolo26m_pose":
        ct = conf_thresh if conf_thresh is not None else 0.4
        return YOLOPoseAdapter(model_path=weights_path, variant="medium", conf_thresh=ct, device=device, **kwargs)

    elif canon_key == "yolo26x_pose":
        ct = conf_thresh if conf_thresh is not None else 0.4
        return YOLOPoseAdapter(model_path=weights_path, variant="xlarge", conf_thresh=ct, device=device, **kwargs)

    else:
        # Default fallback
        return MediaPipePoseAdapter(model_path=weights_path, is_lite=False)
