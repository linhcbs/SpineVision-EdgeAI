"""
Data Types and Structures for the Posture Analyzer Module.
All types are model-agnostic — they work with any HPE backend.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np




class KeypointFormat(Enum):
    """Supported keypoint format standards."""
    MEDIAPIPE_33 = "mediapipe_33"   # 33 landmarks with (x, y, z, visibility)
    COCO_17 = "coco_17"             # 17 keypoints with (x, y, score) or (y, x, score)


class PostureStatus(Enum):
    """Classified posture state labels."""
    GOOD = "GOOD"
    FORWARD_HEAD = "FORWARD_HEAD"       # Forward Head / Text Neck
    SLOUCHED = "SLOUCHED"               # Slouching / Kyphosis
    SHOULDER_TILTED = "SHOULDER_TILTED" # Lateral shoulder asymmetry (generic)
    LEANING_LEFT = "LEANING_LEFT"       # Lateral tilt / lean to the left
    LEANING_RIGHT = "LEANING_RIGHT"     # Lateral tilt / lean to the right
    TOO_CLOSE = "TOO_CLOSE"             # Too close to screen
    TOO_CLOSE_DESK = "TOO_CLOSE_DESK"   # Too close to desk surface
    COMBINED = "COMBINED"               # Multiple simultaneous violations
    UNKNOWN = "UNKNOWN"                 # Undetermined / detecting


class RiskSeverity(Enum):
    """Ergonomic risk and severity categories."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    SEVERE = "Severe"


@dataclass
class RiskAssessment:
    """Quantitative risk and severity assessment for ergonomic disorders."""
    # Kyphosis (Gù lưng)
    kyphosis_risk_pct: float = 0.0              # Instantaneous Kyphosis Risk (0 - 100%)
    prolonged_kyphosis_risk_pct: float = 0.0    # Prolonged / cumulative exposure risk (0 - 100%)
    kyphosis_severity: str = "Low"              # Low | Moderate | High | Severe

    # Myopia (Cận thị)
    myopia_risk_pct: float = 0.0                # Instantaneous Myopia Risk (0 - 100%)
    prolonged_myopia_risk_pct: float = 0.0      # Prolonged / cumulative exposure risk (0 - 100%)
    myopia_severity: str = "Low"                # Low | Moderate | High | Severe

    # Sitting Posture Description
    posture_description: str = "Upright / Straight"  # English descriptive label
    bad_posture_duration_s: float = 0.0         # Continuous duration in bad posture (seconds)
    near_screen_duration_s: float = 0.0         # Continuous duration close to screen (seconds)
    classifier_mode: str = "hybrid"             # rule_based | ml | hybrid


class AlertLevel(Enum):
    """Alert severity levels."""
    SAFE = 0          # Optimal posture, no alert
    MILD = 1          # Mild reminder (visual overlay)
    WARNING = 2       # Warning (caution indicator)
    CRITICAL = 3      # Severe violation


class DeviceType(Enum):
    """Type of detected electronic screen device."""
    WEBCAM      = "webcam"     # The monitoring webcam itself (always present)
    LAPTOP      = "laptop"     # Laptop / notebook screen
    PHONE       = "phone"      # Smartphone
    TABLET      = "tablet"     # Tablet / iPad
    MONITOR     = "monitor"    # External desktop monitor
    TV          = "tv"         # Television
    UNKNOWN     = "unknown"    # Detected but type unclear


@dataclass
class DetectedDevice:
    """A single detected electronic-screen device in the current frame."""
    device_type: DeviceType = DeviceType.UNKNOWN
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) pixels, None = webcam
    center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))  # (cx, cy) pixels
    confidence: float = 1.0      # Detection confidence [0-1]
    distance_cm: float = -1.0    # Eye-to-device distance in cm, -1 = not computable
    label: str = ""              # Display label, e.g. "Phone #1", "Webcam"
    source: str = "yolo"         # Detection source: 'yolo' | 'heuristic' | 'fixed'


class ViewMode(Enum):
    """User viewpoint orientation relative to the camera."""
    FRONTAL = "FRONTAL"   # Sitting facing the camera directly (< ~25°)
    OBLIQUE = "OBLIQUE"   # Angled/quarter view (~25° to ~65°)
    PROFILE = "PROFILE"   # Side/profile view (> ~65°)


# ==========================================================================
# Unified Keypoint Mapping
# ==========================================================================
# MediaPipe Pose 33-landmark indices
MP_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# COCO 17-keypoint indices (MoveNet, YOLO-Pose)
COCO_LANDMARKS = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


@dataclass
class NormalizedKeypoint:
    """A single keypoint in pixel coordinates with optional depth."""
    x: float          # x pixel coordinate
    y: float          # y pixel coordinate
    z: float = 0.0    # depth (MediaPipe only, 0.0 for 2D models)
    score: float = 1.0 # visibility / confidence score


@dataclass
class UnifiedKeypoints:
    """
    Model-agnostic keypoint representation.
    All coordinates are in PIXEL space (not normalized).
    """
    format: KeypointFormat
    frame_width: int
    frame_height: int
    keypoints: Dict[str, NormalizedKeypoint] = field(default_factory=dict)

    @property
    def has_depth(self) -> bool:
        return self.format == KeypointFormat.MEDIAPIPE_33

    def get(self, name: str) -> Optional[NormalizedKeypoint]:
        """Get a keypoint by semantic name (e.g., 'left_shoulder')."""
        return self.keypoints.get(name)

    def get_midpoint(self, name_a: str, name_b: str) -> Optional[NormalizedKeypoint]:
        """Compute the midpoint between two named keypoints."""
        a = self.get(name_a)
        b = self.get(name_b)
        if a is None or b is None:
            return None
        return NormalizedKeypoint(
            x=(a.x + b.x) / 2.0,
            y=(a.y + b.y) / 2.0,
            z=(a.z + b.z) / 2.0,
            score=min(a.score, b.score),
        )


@dataclass
class PostureMetrics:
    """Computed ergonomic biometric indices for a single frame."""
    neck_cva_deg: float = 0.0              # Craniovertebral Angle (degrees)
    shoulder_tilt_deg: float = 0.0         # Raw shoulder tilt deviation (degrees)
    shoulder_level_deg: float = 180.0      # Scaled shoulder angle [180° = level, 90° = vertical]
    lateral_tilt_deg: float = 0.0          # Signed lateral tilt (deg): <0 = tilt left, >0 = tilt right
    lateral_spine_offset: float = 0.0      # Signed horizontal offset (mid_shoulder.x - mid_hip.x) / torso_len
    trunk_angle_deg: float = 0.0           # Trunk / Spine Slump Angle (degrees)
    eye_distance_cm: float = -1.0          # Eye-to-Screen distance (cm), -1 = unavailable (primary / webcam)
    eye_to_desk_cm: float = -1.0           # Eye-to-Desk distance (cm), -1 = unavailable
    yaw_deg: float = 0.0                   # Estimated torso/view yaw angle (degrees)
    view_mode: ViewMode = ViewMode.FRONTAL # Viewpoint classification (FRONTAL, OBLIQUE, PROFILE)
    is_valid: bool = False                 # Whether enough keypoints were visible for computation
    device_distances: List["DetectedDevice"] = field(default_factory=list)  # Per-device distances


@dataclass
class CalibrationData:
    """Stores the user's personal baseline posture metrics."""
    baseline_neck_cva: float = 0.0
    baseline_shoulder_tilt: float = 0.0
    baseline_shoulder_level: float = 180.0
    baseline_trunk_angle: float = 0.0
    baseline_eye_distance: float = 60.0    # cm (Eye to Screen)
    baseline_eye_to_desk: float = 45.0     # cm (Eye to Desk)
    baseline_ipd_pixels: float = 0.0       # Interpupillary Distance in pixels at calibrated distance
    n_samples: int = 0
    is_calibrated: bool = False
    focal_length_px: float = 0.0           # Estimated focal length for distance calculation



@dataclass
class PostureState:
    """Complete posture analysis result for a single frame."""
    status: PostureStatus = PostureStatus.UNKNOWN
    violations: List[PostureStatus] = field(default_factory=list)
    metrics: PostureMetrics = field(default_factory=PostureMetrics)
    deviations: Dict[str, float] = field(default_factory=dict)  # Deviation from baseline
    confidence: float = 0.0               # Overall analysis confidence (0.0 - 1.0)
    alert_level: AlertLevel = AlertLevel.SAFE
    is_calibrated: bool = False
    timestamp_ms: float = 0.0

    # Ergonomic risk & diagnostic assessment
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    kyphosis_risk_pct: float = 0.0              # Direct access to instant kyphosis risk
    prolonged_kyphosis_risk_pct: float = 0.0    # Direct access to prolonged kyphosis risk
    myopia_risk_pct: float = 0.0                # Direct access to instant myopia risk
    prolonged_myopia_risk_pct: float = 0.0      # Direct access to prolonged myopia risk
    posture_description: str = "Upright / Straight"  # English sitting posture description

    # Workspace detection results
    desk_line_y: int = -1                  # Detected desk surface y-coordinate (-1 = not detected)
    screen_region: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) of detected screen

    # Multi-device screen detection results
    detected_devices: List["DetectedDevice"] = field(default_factory=list)  # All screens detected this frame


def convert_mediapipe_to_unified(landmarks, frame_width: int, frame_height: int) -> UnifiedKeypoints:
    """
    Convert MediaPipe Pose landmarks to UnifiedKeypoints.
    
    Args:
        landmarks: MediaPipe PoseLandmarker result landmarks (list of NormalizedLandmark)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        UnifiedKeypoints with all 33 landmarks mapped to semantic names
    """
    unified = UnifiedKeypoints(
        format=KeypointFormat.MEDIAPIPE_33,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    index_to_name = {v: k for k, v in MP_LANDMARKS.items()}

    for idx, lm in enumerate(landmarks):
        name = index_to_name.get(idx)
        if name is None:
            continue
        unified.keypoints[name] = NormalizedKeypoint(
            x=lm.x * frame_width,
            y=lm.y * frame_height,
            z=getattr(lm, 'z', 0.0),
            score=getattr(lm, 'visibility', 1.0),
        )

    return unified


def convert_coco_to_unified(
    keypoints,
    frame_width: int,
    frame_height: int,
    is_normalized: bool = False,
    is_yx_order: bool = False,
) -> UnifiedKeypoints:
    """
    Convert COCO 17-keypoint array to UnifiedKeypoints.
    Handles both MoveNet (y,x,score normalized) and YOLO (x,y,score pixel) formats.

    Args:
        keypoints: Array-like of shape (17, 3)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        is_normalized: True if coordinates are in [0, 1] range
        is_yx_order: True if format is (y, x, score) like MoveNet
    
    Returns:
        UnifiedKeypoints with 17 COCO landmarks mapped
    """
    unified = UnifiedKeypoints(
        format=KeypointFormat.COCO_17,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    index_to_name = {v: k for k, v in COCO_LANDMARKS.items()}

    for idx, kp in enumerate(keypoints):
        name = index_to_name.get(idx)
        if name is None:
            continue

        if len(kp) >= 3:
            p1, p2, score = float(kp[0]), float(kp[1]), float(kp[2])
        elif len(kp) == 2:
            p1, p2, score = float(kp[0]), float(kp[1]), 1.0
        else:
            continue

        if is_yx_order:
            # MoveNet format: (y_norm, x_norm, score)
            y_val, x_val = p1, p2
        else:
            x_val, y_val = p1, p2

        if is_normalized or (0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0 and not (x_val > 1.0 or y_val > 1.0)):
            # Heuristic: if values are all <= 1.0, treat as normalized
            # But also check: MoveNet is always normalized; YOLO is pixel coords
            if is_normalized or (x_val <= 1.0 and y_val <= 1.0 and score > 0):
                x_px = x_val * frame_width
                y_px = y_val * frame_height
            else:
                x_px = x_val
                y_px = y_val
        else:
            x_px = x_val
            y_px = y_val

        unified.keypoints[name] = NormalizedKeypoint(
            x=x_px,
            y=y_px,
            z=0.0,  # COCO models don't provide depth
            score=score,
        )

    return unified
