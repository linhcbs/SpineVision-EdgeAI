"""
SpineVision EdgeAI - Posture Analyzer Module
=============================================
Model-agnostic posture analysis engine that works with any Human Pose Estimation
backend (MediaPipe Pose, MoveNet, YOLO-Pose).

Converts raw keypoint coordinates into ergonomic biometric indices:
  - Craniovertebral Angle (CVA) — Forward Head / Text Neck detection
  - Shoulder Tilt Angle — Lateral spinal asymmetry detection
  - Trunk / Spine Angle — Slouching / Kyphosis detection
  - Eye-to-Screen Distance — Myopia risk estimation

Includes:
  - One-Euro Filter for temporal landmark smoothing
  - Auto-Calibration system for personalized baselines
  - Rule-based + ML-ready Posture Classifier
  - Workspace surface and screen detection
"""

from .types import (
    PostureMetrics,
    PostureState,
    PostureStatus,
    AlertLevel,
    ViewMode,
    CalibrationData,
    KeypointFormat,
    UnifiedKeypoints,
    NormalizedKeypoint,
    DetectedDevice,
    DeviceType,
    RiskAssessment,
    RiskSeverity,
)
from .filter_utils import OneEuroFilter, LandmarkSmoother
from .posture_metrics import PostureMetricsCalculator
from .distance_estimator import DistanceEstimator
from .calibrator import PostureCalibrator
from .posture_classifier import PostureClassifier
from .workspace_detector import WorkspaceDetector
from .posture_engine import PostureAnalysisEngine

from .hpe_adapter import (
    BaseHPEAdapter,
    MediaPipePoseAdapter,
    MoveNetPoseAdapter,
    YOLOPoseAdapter,
    create_hpe_detector,
    list_supported_hpe_models,
    SUPPORTED_HPE_MODELS,
    canonical_model_key,
)

__all__ = [
    "PostureMetrics",
    "PostureState",
    "PostureStatus",
    "AlertLevel",
    "ViewMode",
    "CalibrationData",
    "KeypointFormat",
    "UnifiedKeypoints",
    "NormalizedKeypoint",
    "DetectedDevice",
    "DeviceType",
    "RiskAssessment",
    "RiskSeverity",
    "OneEuroFilter",
    "LandmarkSmoother",
    "PostureMetricsCalculator",
    "DistanceEstimator",
    "PostureCalibrator",
    "PostureClassifier",
    "WorkspaceDetector",
    "PostureAnalysisEngine",
    "BaseHPEAdapter",
    "MediaPipePoseAdapter",
    "MoveNetPoseAdapter",
    "YOLOPoseAdapter",
    "create_hpe_detector",
    "list_supported_hpe_models",
    "SUPPORTED_HPE_MODELS",
    "canonical_model_key",
]

