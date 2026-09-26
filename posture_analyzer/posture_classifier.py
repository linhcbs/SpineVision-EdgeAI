"""
Posture Classifier — Hybrid Rule-based + ML-ready Classification.

Classifies the user's posture into discrete states based on ergonomic metrics,
using both absolute clinical thresholds and relative deviations from the
user's personal calibrated baseline.

Classification Pipeline:
    Tier 1 (Rule-based): Fast O(1) check against clinical thresholds.
                         Catches clear violations immediately.
    Tier 2 (ML-ready):   For ambiguous / marginal cases, prepares feature
                         vectors for a lightweight ML classifier (SVM / RF / KNN).
                         Currently uses extended rule logic; ML model can be
                         plugged in later.

Posture States:
    - GOOD: All metrics within safe thresholds
    - FORWARD_HEAD: CVA below threshold (Forward Head Posture / Text Neck)
    - SLOUCHED: Trunk angle exceeds threshold (Kyphotic / Slouching)
    - SHOULDER_TILTED: Shoulder tilt exceeds threshold (Lateral asymmetry)
    - TOO_CLOSE: Eye-to-screen distance below safe minimum
    - COMBINED: Multiple violations detected simultaneously
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .types import (
    PostureMetrics,
    PostureState,
    PostureStatus,
    AlertLevel,
    CalibrationData,
    ViewMode,
)
from .config import get_posture_config


class PostureClassifier:
    """
    Classifies posture based on ergonomic metrics.
    
    When calibrated, uses relative deviations from the user's personal baseline.
    When uncalibrated, falls back to absolute clinical thresholds.
    
    Parameters:
        use_relative: If True and calibration exists, use relative deviation mode.
        sensitivity: Overall sensitivity multiplier (0.5 = lenient, 1.5 = strict).
                     Adjusts all thresholds proportionally.
    """

    def __init__(self, use_relative: Optional[bool] = None, sensitivity: Optional[float] = None):
        cfg = get_posture_config()
        class_cfg = cfg.get("classification", {})
        metrics_cfg = cfg.get("metrics", {})
        calib_cfg = cfg.get("calibration", {}).get("relative_deviations", {})

        self.use_relative = use_relative if use_relative is not None else class_cfg.get("use_relative_when_calibrated", True)
        self.sensitivity = sensitivity if sensitivity is not None else class_cfg.get("default_sensitivity", 1.0)

        # CVA thresholds
        cva_cfg = metrics_cfg.get("neck_cva", {})
        self.cva_normal_min = cva_cfg.get("normal_min", 50.0)
        self.cva_severe_below = cva_cfg.get("severe_below", 40.0)

        # Shoulder Tilt thresholds (raw tilt)
        tilt_cfg = metrics_cfg.get("shoulder_tilt", {}).get("raw_tilt_thresholds", {})
        self.tilt_normal_max = tilt_cfg.get("normal_max", 5.0)
        self.tilt_severe_above = tilt_cfg.get("severe_above", 8.0)

        # Trunk Angle thresholds
        trunk_cfg = metrics_cfg.get("trunk_angle", {})
        self.trunk_normal_max = trunk_cfg.get("normal_max", 10.0)
        self.trunk_severe_above = trunk_cfg.get("severe_above", 18.0)

        # Eye to Screen thresholds
        screen_cfg = metrics_cfg.get("eye_to_screen_distance", {})
        self.screen_safe_min = screen_cfg.get("safe_min", 50.0)
        self.screen_danger_below = screen_cfg.get("danger_below", 35.0)

        # Eye to Desk thresholds
        desk_cfg = metrics_cfg.get("eye_to_desk_distance", {})
        self.desk_safe_min = desk_cfg.get("safe_min", 35.0)
        self.desk_danger_below = desk_cfg.get("danger_below", 25.0)

        # Relative deviation thresholds
        self.cva_rel_warn = calib_cfg.get("neck_cva_warn_deg", 8.0)
        self.cva_rel_severe = calib_cfg.get("neck_cva_severe_deg", 15.0)
        self.tilt_rel_warn = calib_cfg.get("shoulder_tilt_warn_deg", 4.0)
        self.tilt_rel_severe = calib_cfg.get("shoulder_tilt_severe_deg", 8.0)
        self.trunk_rel_warn = calib_cfg.get("trunk_angle_warn_deg", 8.0)
        self.trunk_rel_severe = calib_cfg.get("trunk_angle_severe_deg", 15.0)
        self.screen_rel_warn = calib_cfg.get("screen_distance_warn_cm", 15.0)
        self.screen_rel_severe = calib_cfg.get("screen_distance_severe_cm", 25.0)
        self.desk_rel_warn = calib_cfg.get("desk_distance_warn_cm", 10.0)
        self.desk_rel_severe = calib_cfg.get("desk_distance_severe_cm", 18.0)

        # View modes adaptive configurations
        self.view_modes_cfg = cfg.get("view_modes", {})
        self.view_modes_enabled = self.view_modes_cfg.get("enabled", True)


    def classify(
        self,
        metrics: PostureMetrics,
        calibration: Optional[CalibrationData] = None,
    ) -> PostureState:
        """
        Classify the current posture state.
        
        Args:
            metrics: Computed PostureMetrics from the current frame.
            calibration: Optional calibration data for relative deviation mode.
        
        Returns:
            PostureState with classified status, violations, deviations, and alert level.
        """
        if not metrics.is_valid:
            return PostureState(status=PostureStatus.UNKNOWN, confidence=0.0)

        violations: List[PostureStatus] = []
        deviations: Dict[str, float] = {}
        severity_scores: List[float] = []  # 0.0 = safe, 1.0 = severe

        is_cal = calibration is not None and calibration.is_calibrated and self.use_relative

        # Select adaptive thresholds based on ViewMode
        view_mode = getattr(metrics, "view_mode", ViewMode.FRONTAL)
        mode_key = view_mode.value.lower() if isinstance(view_mode, ViewMode) else str(view_mode).lower()
        mode_cfg = self.view_modes_cfg.get(mode_key, {}) if self.view_modes_enabled else {}

        # Resolve CVA thresholds for current view
        cva_mode_cfg = mode_cfg.get("neck_cva", {})
        cva_norm_min = cva_mode_cfg.get("normal_min", self.cva_normal_min)
        cva_sev_below = cva_mode_cfg.get("severe_below", self.cva_severe_below)

        # Resolve Shoulder Tilt thresholds for current view
        tilt_mode_cfg = mode_cfg.get("shoulder_tilt", {})
        tilt_enabled = tilt_mode_cfg.get("enabled", True) if self.view_modes_enabled else True
        tilt_raw_cfg = tilt_mode_cfg.get("raw_tilt_thresholds", {})
        tilt_norm_max = tilt_raw_cfg.get("normal_max", self.tilt_normal_max)
        tilt_sev_above = tilt_raw_cfg.get("severe_above", self.tilt_severe_above)

        # ----- CVA / Forward Head Analysis -----
        if metrics.neck_cva_deg > 0:
            if is_cal:
                delta_cva = calibration.baseline_neck_cva - metrics.neck_cva_deg
                deviations["neck_cva_delta"] = delta_cva
                warn_thresh = self.cva_rel_warn / self.sensitivity
                severe_thresh = self.cva_rel_severe / self.sensitivity
                if delta_cva > severe_thresh:
                    violations.append(PostureStatus.FORWARD_HEAD)
                    severity_scores.append(1.0)
                elif delta_cva > warn_thresh:
                    violations.append(PostureStatus.FORWARD_HEAD)
                    severity_scores.append(0.5)
            else:
                if metrics.neck_cva_deg < cva_sev_below * self.sensitivity:
                    violations.append(PostureStatus.FORWARD_HEAD)
                    severity_scores.append(1.0)
                elif metrics.neck_cva_deg < cva_norm_min * self.sensitivity:
                    violations.append(PostureStatus.FORWARD_HEAD)
                    severity_scores.append(0.4)

        # ----- Shoulder Tilt Analysis -----
        # In PROFILE mode, shoulder tilt measurement in 2D is degenerate/meaningless, so bypassed
        if tilt_enabled:
            if is_cal:
                delta_tilt = abs(metrics.shoulder_tilt_deg) - abs(calibration.baseline_shoulder_tilt)
                deviations["shoulder_tilt_delta"] = delta_tilt
                warn_thresh = self.tilt_rel_warn / self.sensitivity
                severe_thresh = self.tilt_rel_severe / self.sensitivity
                if delta_tilt > severe_thresh:
                    violations.append(PostureStatus.SHOULDER_TILTED)
                    severity_scores.append(1.0)
                elif delta_tilt > warn_thresh:
                    violations.append(PostureStatus.SHOULDER_TILTED)
                    severity_scores.append(0.5)
            else:
                if metrics.shoulder_tilt_deg > tilt_sev_above / self.sensitivity:
                    violations.append(PostureStatus.SHOULDER_TILTED)
                    severity_scores.append(1.0)
                elif metrics.shoulder_tilt_deg > tilt_norm_max / self.sensitivity:
                    violations.append(PostureStatus.SHOULDER_TILTED)
                    severity_scores.append(0.4)

        # ----- Trunk / Spine Angle Analysis -----
        if is_cal:
            delta_trunk = metrics.trunk_angle_deg - calibration.baseline_trunk_angle
            deviations["trunk_angle_delta"] = delta_trunk
            warn_thresh = self.trunk_rel_warn / self.sensitivity
            severe_thresh = self.trunk_rel_severe / self.sensitivity
            if delta_trunk > severe_thresh:
                violations.append(PostureStatus.SLOUCHED)
                severity_scores.append(1.0)
            elif delta_trunk > warn_thresh:
                violations.append(PostureStatus.SLOUCHED)
                severity_scores.append(0.5)
        else:
            if metrics.trunk_angle_deg > self.trunk_severe_above / self.sensitivity:
                violations.append(PostureStatus.SLOUCHED)
                severity_scores.append(1.0)
            elif metrics.trunk_angle_deg > self.trunk_normal_max / self.sensitivity:
                violations.append(PostureStatus.SLOUCHED)
                severity_scores.append(0.4)

        # ----- Eye-to-Screen Distance Analysis -----
        if metrics.eye_distance_cm > 0:
            if is_cal:
                delta_dist = calibration.baseline_eye_distance - metrics.eye_distance_cm
                deviations["eye_distance_delta"] = delta_dist
                warn_thresh = self.screen_rel_warn / self.sensitivity
                severe_thresh = self.screen_rel_severe / self.sensitivity
                if delta_dist > severe_thresh:
                    violations.append(PostureStatus.TOO_CLOSE)
                    severity_scores.append(1.0)
                elif delta_dist > warn_thresh:
                    violations.append(PostureStatus.TOO_CLOSE)
                    severity_scores.append(0.5)
            else:
                if metrics.eye_distance_cm < self.screen_danger_below * self.sensitivity:
                    violations.append(PostureStatus.TOO_CLOSE)
                    severity_scores.append(1.0)
                elif metrics.eye_distance_cm < self.screen_safe_min * self.sensitivity:
                    violations.append(PostureStatus.TOO_CLOSE)
                    severity_scores.append(0.4)

        # ----- Eye-to-Desk Distance Analysis -----
        if metrics.eye_to_desk_cm > 0:
            if is_cal:
                delta_desk = calibration.baseline_eye_to_desk - metrics.eye_to_desk_cm
                deviations["eye_to_desk_delta"] = delta_desk
                warn_thresh = self.desk_rel_warn / self.sensitivity
                severe_thresh = self.desk_rel_severe / self.sensitivity
                if delta_desk > severe_thresh:
                    violations.append(PostureStatus.TOO_CLOSE_DESK)
                    severity_scores.append(1.0)
                elif delta_desk > warn_thresh:
                    violations.append(PostureStatus.TOO_CLOSE_DESK)
                    severity_scores.append(0.5)
            else:
                if metrics.eye_to_desk_cm < self.desk_danger_below * self.sensitivity:
                    violations.append(PostureStatus.TOO_CLOSE_DESK)
                    severity_scores.append(1.0)
                elif metrics.eye_to_desk_cm < self.desk_safe_min * self.sensitivity:
                    violations.append(PostureStatus.TOO_CLOSE_DESK)
                    severity_scores.append(0.4)


        # ----- Determine final status and alert level -----
        if len(violations) == 0:
            status = PostureStatus.GOOD
            alert_level = AlertLevel.SAFE
        elif len(violations) == 1:
            status = violations[0]
            max_sev = max(severity_scores) if severity_scores else 0.0
            if max_sev >= 0.8:
                alert_level = AlertLevel.CRITICAL
            elif max_sev >= 0.4:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.MILD
        else:
            status = PostureStatus.COMBINED
            max_sev = max(severity_scores) if severity_scores else 0.0
            if max_sev >= 0.8 or len(violations) >= 3:
                alert_level = AlertLevel.CRITICAL
            elif max_sev >= 0.4:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.MILD

        # Confidence: based on how many metrics were computable
        n_computed = sum([
            metrics.neck_cva_deg > 0,
            True,  # shoulder tilt is always computed if shoulders visible
            True,  # trunk angle
            metrics.eye_distance_cm > 0,
        ])
        confidence = min(1.0, n_computed / 4.0)

        return PostureState(
            status=status,
            violations=violations,
            metrics=metrics,
            deviations=deviations,
            confidence=confidence,
            alert_level=alert_level,
            is_calibrated=is_cal,
        )
