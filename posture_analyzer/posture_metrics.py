"""
Posture Biometric Metrics Calculator.
Computes ergonomic indices from unified keypoints (model-agnostic).

Indices computed:
  1. Craniovertebral Angle (CVA) — Forward Head / Text Neck
  2. Shoulder Tilt Angle — Lateral spinal asymmetry
  3. Trunk / Spine Angle — Slouching / Kyphosis
  
All angles are returned in DEGREES.
"""

import math
import numpy as np
from typing import Optional, Tuple

from .types import (
    UnifiedKeypoints,
    NormalizedKeypoint,
    PostureMetrics,
    KeypointFormat,
    ViewMode,
)
from .config import get_posture_config


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================


def _angle_between_points_horizontal(p_lower: NormalizedKeypoint, p_upper: NormalizedKeypoint) -> float:
    """
    Compute the angle of the vector from p_lower to p_upper relative to the
    horizontal axis (pointing right).
    
    Returns angle in degrees [0, 180].
    For CVA: p_lower = mid_shoulder, p_upper = ear/tragus.
    """
    dx = p_upper.x - p_lower.x
    dy = p_lower.y - p_upper.y   # Inverted because image y increases downward
    angle_rad = math.atan2(dy, abs(dx)) if abs(dx) > 1e-6 else math.pi / 2
    return abs(math.degrees(angle_rad))


def _angle_from_vertical(p_top: NormalizedKeypoint, p_bottom: NormalizedKeypoint) -> float:
    """
    Compute the angle between the vector (p_bottom -> p_top) and the vertical axis
    (straight up). Used for trunk/spine angle.
    
    Returns angle in degrees [0, 180]. 
    0° = perfectly vertical (good posture).
    """
    dx = p_top.x - p_bottom.x
    dy = p_bottom.y - p_top.y  # Inverted: image y down, we want up as positive
    
    # Vertical reference vector: (0, 1) pointing up
    # cos(theta) = dot(v, vertical) / (|v| * |vertical|)
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return 0.0
    
    cos_theta = dy / mag  # dot product with (0, 1)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _horizontal_tilt_angle(p_left: NormalizedKeypoint, p_right: NormalizedKeypoint) -> float:
    """
    Compute the tilt angle of the line connecting two horizontally-paired
    keypoints (e.g., left shoulder — right shoulder) relative to horizontal.
    
    Uses abs(dx) so the result is always the deviation from horizontal,
    regardless of which keypoint has a higher x-coordinate in image space.
    (MediaPipe's 'left_shoulder' appears on the RIGHT side of the image.)
    
    Returns angle in degrees [0, 90]. 0° = perfectly level.
    """
    dx = p_right.x - p_left.x
    dy = p_left.y - p_right.y  # Inverted for image coords
    if abs(dx) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dy, abs(dx))))


class PostureMetricsCalculator:
    """
    Computes ergonomic posture metrics from a UnifiedKeypoints object.
    Works with both MediaPipe (33 landmarks) and COCO (17 keypoints).
    
    Parameters:
        min_confidence: Minimum keypoint confidence to use in calculations.
                        Keypoints below this threshold are treated as missing.
    """

    def __init__(self, min_confidence: Optional[float] = None):
        cfg = get_posture_config()
        default_conf = cfg.get("keypoints", {}).get("min_confidence", 0.3)
        self.min_confidence = min_confidence if min_confidence is not None else default_conf

        # View mode adaptive rule parameters
        vm_cfg = cfg.get("view_modes", {})
        self.view_modes_enabled: bool = vm_cfg.get("enabled", True)
        self.hysteresis_deg: float = vm_cfg.get("hysteresis_deg", 5.0)
        self.yaw_smoothing_alpha: float = vm_cfg.get("yaw_smoothing_alpha", 0.25)
        self.frontal_max_yaw: float = vm_cfg.get("frontal", {}).get("max_yaw_deg", 25.0)
        self.oblique_max_yaw: float = vm_cfg.get("oblique", {}).get("max_yaw_deg", 65.0)

        # Internal state tracking for view mode and yaw
        self._current_view_mode: ViewMode = ViewMode.FRONTAL
        self._smoothed_yaw: float = 0.0
        self._has_init_yaw: bool = False


    def _is_valid(self, kp: Optional[NormalizedKeypoint]) -> bool:
        """Check if a keypoint exists and has sufficient confidence."""
        return kp is not None and kp.score >= self.min_confidence

    def compute_neck_cva(self, kps: UnifiedKeypoints) -> Optional[float]:
        """
        Compute the Craniovertebral Angle (CVA).
        
        Measures the angle from the midpoint of both shoulders (C7 approximation)
        to the ear (tragus), relative to the horizontal plane.
        
        For MediaPipe: Uses left_ear (7), right_ear (8), left_shoulder (11), right_shoulder (12).
        For COCO: Uses left_ear (3), right_ear (4), left_shoulder (5), right_shoulder (6).
        
        Clinical thresholds:
            >= 50°: Normal posture
            40° - 49°: Mild Forward Head  
            < 40°: Severe Forward Head (cervical load increases ~300%)
        
        Returns:
            Angle in degrees, or None if keypoints insufficient.
        """
        l_shoulder = kps.get("left_shoulder")
        r_shoulder = kps.get("right_shoulder")
        l_ear = kps.get("left_ear")
        r_ear = kps.get("right_ear")

        # Must have at least one shoulder pair visible
        if not (self._is_valid(l_shoulder) and self._is_valid(r_shoulder)):
            return None

        mid_shoulder = kps.get_midpoint("left_shoulder", "right_shoulder")

        # Use the best-visible ear, or average both
        if self._is_valid(l_ear) and self._is_valid(r_ear):
            ear = NormalizedKeypoint(
                x=(l_ear.x + r_ear.x) / 2.0,
                y=(l_ear.y + r_ear.y) / 2.0,
                z=(l_ear.z + r_ear.z) / 2.0,
                score=min(l_ear.score, r_ear.score),
            )
        elif self._is_valid(l_ear):
            ear = l_ear
        elif self._is_valid(r_ear):
            ear = r_ear
        else:
            # Fallback: use nose if no ears visible
            nose = kps.get("nose")
            if self._is_valid(nose):
                ear = nose
            else:
                return None

        return _angle_between_points_horizontal(mid_shoulder, ear)

    def compute_shoulder_tilt(self, kps: UnifiedKeypoints) -> Optional[float]:
        """
        Compute the Shoulder Tilt Angle.
        
        Measures the angular deviation of the shoulder line from horizontal.
        
        Clinical thresholds:
            <= 5°: Balanced (normal)
            > 8°: Significant lateral asymmetry (scoliosis risk)
        
        Returns:
            Absolute angle in degrees, or None if keypoints insufficient.
        """
        l_shoulder = kps.get("left_shoulder")
        r_shoulder = kps.get("right_shoulder")

        if not (self._is_valid(l_shoulder) and self._is_valid(r_shoulder)):
            return None

        return abs(_horizontal_tilt_angle(l_shoulder, r_shoulder))

    def compute_trunk_angle(self, kps: UnifiedKeypoints) -> Optional[float]:
        """
        Compute the Trunk / Spine Slump Angle.
        
        Measures the angle between the spine vector (midpoint of shoulders
        to midpoint of hips) and the vertical axis.
        
        Clinical thresholds:
            <= 10°: Good upright posture
            > 18°: Slouching / Kyphotic posture
        
        Returns:
            Angle in degrees from vertical, or None if keypoints insufficient.
        """
        mid_shoulder = kps.get_midpoint("left_shoulder", "right_shoulder")
        mid_hip = kps.get_midpoint("left_hip", "right_hip")

        if mid_shoulder is None or mid_hip is None:
            return None
        if mid_shoulder.score < self.min_confidence or mid_hip.score < self.min_confidence:
            return None

        return _angle_from_vertical(mid_shoulder, mid_hip)

    def estimate_view_angle(self, kps: UnifiedKeypoints) -> Tuple[float, ViewMode]:
        """
        Estimate user viewpoint yaw angle relative to camera and classify ViewMode.
        
        Uses 3D depth disparity (MediaPipe) or 2D facial asymmetry and shoulder
        foreshortening (COCO / 2D backends).
        Applies exponential moving average (EMA) smoothing and Schmitt-trigger
        hysteresis to prevent boundary fluttering.
        
        Returns:
            Tuple of (smoothed_yaw_deg, ViewMode)
        """
        if not self.view_modes_enabled:
            return 0.0, ViewMode.FRONTAL

        l_shoulder = kps.get("left_shoulder")
        r_shoulder = kps.get("right_shoulder")

        # Fallback if shoulders not sufficiently detected
        if not (self._is_valid(l_shoulder) and self._is_valid(r_shoulder)):
            return self._smoothed_yaw, self._current_view_mode

        raw_yaw = 0.0

        # Method 1: MediaPipe 3D Landmark depth disparity
        has_depth_info = kps.has_depth and (abs(l_shoulder.z) > 1e-4 or abs(r_shoulder.z) > 1e-4)
        if has_depth_info:
            dx = (r_shoulder.x - l_shoulder.x) / max(kps.frame_width, 1)
            dz = r_shoulder.z - l_shoulder.z
            raw_yaw = math.degrees(math.atan2(abs(dz), max(abs(dx), 1e-4)))
        else:
            # Method 2: 2D Facial asymmetry + Ear visibility (works for COCO/MoveNet/YOLO)
            nose = kps.get("nose")
            l_ear = kps.get("left_ear")
            r_ear = kps.get("right_ear")

            valid_l_ear = self._is_valid(l_ear)
            valid_r_ear = self._is_valid(r_ear)
            valid_nose = self._is_valid(nose)

            if valid_nose and valid_l_ear and valid_r_ear:
                d_left = abs(nose.x - l_ear.x)
                d_right = abs(nose.x - r_ear.x)
                asym = abs(d_left - d_right) / max(d_left + d_right, 1e-5)
                raw_yaw = min(90.0, asym * 90.0)
            elif valid_nose and (valid_l_ear != valid_r_ear):
                # Severe occlusion: only one ear is visible
                raw_yaw = 75.0
            else:
                # Method 3: Shoulder width foreshortening vs torso length
                mid_hip = kps.get_midpoint("left_hip", "right_hip")
                mid_shoulder = kps.get_midpoint("left_shoulder", "right_shoulder")
                if mid_hip is not None and mid_shoulder is not None and self._is_valid(mid_hip):
                    w_sh = math.sqrt((r_shoulder.x - l_shoulder.x) ** 2 + (r_shoulder.y - l_shoulder.y) ** 2)
                    h_torso = math.sqrt((mid_shoulder.x - mid_hip.x) ** 2 + (mid_shoulder.y - mid_hip.y) ** 2)
                    ratio = w_sh / max(h_torso, 1e-4)
                    cos_val = min(1.0, max(0.0, ratio / 1.05))
                    raw_yaw = math.degrees(math.acos(cos_val))
                else:
                    raw_yaw = 0.0

        # Temporal smoothing
        if not self._has_init_yaw:
            self._smoothed_yaw = raw_yaw
            self._has_init_yaw = True
        else:
            self._smoothed_yaw = (1.0 - self.yaw_smoothing_alpha) * self._smoothed_yaw + self.yaw_smoothing_alpha * raw_yaw

        # Schmitt-trigger hysteresis state transitions
        H = self.hysteresis_deg
        T_front = self.frontal_max_yaw
        T_oblique = self.oblique_max_yaw

        if self._current_view_mode == ViewMode.FRONTAL:
            if self._smoothed_yaw > (T_oblique + H):
                self._current_view_mode = ViewMode.PROFILE
            elif self._smoothed_yaw > (T_front + H):
                self._current_view_mode = ViewMode.OBLIQUE
        elif self._current_view_mode == ViewMode.OBLIQUE:
            if self._smoothed_yaw < (T_front - H):
                self._current_view_mode = ViewMode.FRONTAL
            elif self._smoothed_yaw > (T_oblique + H):
                self._current_view_mode = ViewMode.PROFILE
        elif self._current_view_mode == ViewMode.PROFILE:
            if self._smoothed_yaw < (T_front - H):
                self._current_view_mode = ViewMode.FRONTAL
            elif self._smoothed_yaw < (T_oblique - H):
                self._current_view_mode = ViewMode.OBLIQUE

        return self._smoothed_yaw, self._current_view_mode

    def compute_all(self, kps: UnifiedKeypoints) -> PostureMetrics:
        """
        Compute all posture metrics from a unified keypoint set.
        
        Returns:
            PostureMetrics dataclass with all computed indices.
        """
        yaw_deg, view_mode = self.estimate_view_angle(kps)
        neck_cva = self.compute_neck_cva(kps)
        shoulder_tilt = self.compute_shoulder_tilt(kps)
        trunk_angle = self.compute_trunk_angle(kps)

        is_valid = (neck_cva is not None) or (shoulder_tilt is not None) or (trunk_angle is not None)

        raw_tilt = shoulder_tilt if shoulder_tilt is not None else 0.0
        # Scaled shoulder level angle: 180° = level (flat), decreasing to 90° as tilt increases
        shoulder_level = 180.0 - min(90.0, raw_tilt)

        return PostureMetrics(
            neck_cva_deg=neck_cva if neck_cva is not None else 0.0,
            shoulder_tilt_deg=raw_tilt,
            shoulder_level_deg=shoulder_level,
            trunk_angle_deg=trunk_angle if trunk_angle is not None else 0.0,
            eye_distance_cm=-1.0,  # Computed separately by DistanceEstimator
            eye_to_desk_cm=-1.0,   # Computed using desk surface
            yaw_deg=yaw_deg,
            view_mode=view_mode,
            is_valid=is_valid,
        )

