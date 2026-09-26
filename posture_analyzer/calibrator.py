"""
Auto-Calibration Module for Personalized Posture Baselines.

Every user has different body proportions, camera angles, and sitting heights.
This module captures a personalized "good posture" baseline during a 3-5 second
calibration phase, then all subsequent posture analysis uses relative deviation
from this baseline instead of fixed absolute thresholds.

Calibration Workflow:
    1. User clicks "Start Calibration" and sits in correct upright posture.
    2. System collects N frames (default: ~90 frames at 30 FPS = 3 seconds).
    3. System computes mean baseline metrics and stores them.
    4. During analysis, deviations from baseline trigger alerts.
"""

import time
import numpy as np
from typing import List, Optional

from .types import (
    CalibrationData,
    PostureMetrics,
    UnifiedKeypoints,
)
from .posture_metrics import PostureMetricsCalculator
from .distance_estimator import DistanceEstimator
from .config import get_posture_config


class PostureCalibrator:
    """
    Handles the personalized auto-calibration workflow.
    
    Parameters:
        target_samples: Number of frames to collect during calibration (default: 90).
        calibration_distance_cm: Known distance the user sits at during calibration.
        metrics_calculator: PostureMetricsCalculator instance.
        distance_estimator: DistanceEstimator instance.
    """

    def __init__(
        self,
        target_samples: Optional[int] = None,
        calibration_distance_cm: Optional[float] = None,
        metrics_calculator: Optional[PostureMetricsCalculator] = None,
        distance_estimator: Optional[DistanceEstimator] = None,
    ):
        cfg = get_posture_config().get("calibration", {})
        self.target_samples = target_samples if target_samples is not None else cfg.get("target_samples", 90)
        self.calibration_distance_cm = calibration_distance_cm if calibration_distance_cm is not None else cfg.get("calibration_distance_cm", 60.0)
        self.metrics_calc = metrics_calculator or PostureMetricsCalculator()
        self.dist_estimator = distance_estimator or DistanceEstimator()

        self._samples: List[PostureMetrics] = []
        self._ipd_samples: List[float] = []
        self._desk_dist_samples: List[float] = []
        self._is_collecting = False
        self._calibration_data = CalibrationData()
        self._start_time: float = 0.0


    @property
    def is_calibrating(self) -> bool:
        """Whether calibration is currently in progress."""
        return self._is_collecting

    @property
    def is_calibrated(self) -> bool:
        """Whether a valid calibration baseline exists."""
        return self._calibration_data.is_calibrated

    @property
    def calibration(self) -> CalibrationData:
        """Get the current calibration data."""
        return self._calibration_data

    @property
    def progress(self) -> float:
        """Calibration progress (0.0 to 1.0)."""
        if not self._is_collecting:
            return 1.0 if self.is_calibrated else 0.0
        return min(1.0, len(self._samples) / max(1, self.target_samples))

    @property
    def remaining_seconds(self) -> float:
        """Estimated seconds remaining in calibration."""
        if not self._is_collecting:
            return 0.0
        elapsed = time.time() - self._start_time
        if len(self._samples) == 0:
            return self.target_samples / 30.0  # Assume 30 FPS
        fps_estimate = len(self._samples) / max(elapsed, 0.01)
        remaining_frames = self.target_samples - len(self._samples)
        return max(0.0, remaining_frames / max(fps_estimate, 1.0))

    def start_calibration(self):
        """Begin the calibration collection phase."""
        self._samples.clear()
        self._ipd_samples.clear()
        self._is_collecting = True
        self._start_time = time.time()
        self.dist_estimator.reset()
        print("[Calibrator] Calibration started. Please sit upright and look at the screen.")

    def feed_frame(self, kps: UnifiedKeypoints, desk_y: int = -1) -> bool:
        """
        Feed a single frame's keypoints during calibration.
        
        Args:
            kps: Unified keypoints from the current frame.
            desk_y: Optional detected desk surface y coordinate.
        
        Returns:
            True if calibration has completed after this frame.
        """
        if not self._is_collecting:
            return False

        metrics = self.metrics_calc.compute_all(kps)
        if metrics.is_valid:
            self._samples.append(metrics)

        # Collect IPD measurements for focal length calibration
        ipd_px = self.dist_estimator.measure_ipd_pixels(kps)
        if ipd_px is not None:
            self._ipd_samples.append(ipd_px)

        # Collect eye-to-desk distance samples if desk detected
        if desk_y > 0:
            desk_dist = self.dist_estimator.estimate_eye_to_desk(kps, desk_y)
            if desk_dist > 0:
                self._desk_dist_samples.append(desk_dist)

        # Check if we have enough samples
        if len(self._samples) >= self.target_samples:
            self._finalize_calibration()
            return True

        return False

    def _finalize_calibration(self):
        """Compute and store baseline metrics from collected samples."""
        if len(self._samples) == 0:
            print("[Calibrator] ERROR: No valid samples collected. Calibration failed.")
            self._is_collecting = False
            return

        # Compute mean baseline for each metric
        neck_vals = [s.neck_cva_deg for s in self._samples if s.neck_cva_deg > 0]
        tilt_vals = [s.shoulder_tilt_deg for s in self._samples]
        trunk_vals = [s.trunk_angle_deg for s in self._samples]

        avg_tilt = float(np.mean(tilt_vals)) if tilt_vals else 0.0
        avg_desk = float(np.mean(self._desk_dist_samples)) if self._desk_dist_samples else 45.0

        self._calibration_data = CalibrationData(
            baseline_neck_cva=float(np.mean(neck_vals)) if neck_vals else 50.0,
            baseline_shoulder_tilt=avg_tilt,
            baseline_shoulder_level=180.0 - min(90.0, avg_tilt),
            baseline_trunk_angle=float(np.mean(trunk_vals)) if trunk_vals else 5.0,
            baseline_eye_distance=self.calibration_distance_cm,
            baseline_eye_to_desk=avg_desk,
            n_samples=len(self._samples),
            is_calibrated=True,
        )

        # Calibrate focal length for distance estimation
        if self._ipd_samples:
            avg_ipd_px = float(np.mean(self._ipd_samples))
            self._calibration_data.baseline_ipd_pixels = avg_ipd_px
            self.dist_estimator.calibrate_focal_length(avg_ipd_px, self.calibration_distance_cm)
            self._calibration_data.focal_length_px = self.dist_estimator.focal_length_px

        self._is_collecting = False
        print(
            f"[Calibrator] Calibration complete! Baseline:"
            f" CVA={self._calibration_data.baseline_neck_cva:.1f}°,"
            f" Shoulder Level={self._calibration_data.baseline_shoulder_level:.1f}°,"
            f" Trunk={self._calibration_data.baseline_trunk_angle:.1f}°,"
            f" Eye-Screen={self._calibration_data.baseline_eye_distance:.0f} cm,"
            f" Eye-Desk={self._calibration_data.baseline_eye_to_desk:.0f} cm"
        )


    def cancel_calibration(self):
        """Cancel an in-progress calibration."""
        self._is_collecting = False
        self._samples.clear()
        self._ipd_samples.clear()
        print("[Calibrator] Calibration cancelled.")

    def reset(self):
        """Reset all calibration data."""
        self._calibration_data = CalibrationData()
        self._samples.clear()
        self._ipd_samples.clear()
        self._is_collecting = False
        self.dist_estimator.reset()
