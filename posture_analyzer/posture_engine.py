"""
Posture Analysis Engine.

The central coordinator that integrates all posture analysis sub-modules:
  - Landmark smoothing (One-Euro Filter)
  - Biometric metrics calculation (CVA, Shoulder Tilt, Trunk Angle)
  - Eye-to-screen distance estimation (Pinhole Camera Model)
  - Workspace surface and screen detection
  - Personalized baseline auto-calibration
  - Hybrid posture classification and alert generation
  - HUD visualization rendering

Supports both MediaPipe Pose (33 landmarks) and COCO formats (MoveNet, YOLO-Pose).
"""

import time
import numpy as np
from typing import Optional, Tuple, Any

from .types import (
    UnifiedKeypoints,
    PostureState,
    PostureStatus,
    CalibrationData,
    KeypointFormat,
    convert_mediapipe_to_unified,
    convert_coco_to_unified,
)
from .filter_utils import LandmarkSmoother
from .posture_metrics import PostureMetricsCalculator
from .distance_estimator import DistanceEstimator
from .calibrator import PostureCalibrator
from .posture_classifier import PostureClassifier
from .workspace_detector import WorkspaceDetector
from .posture_hud import draw_posture_hud
from .config import get_posture_config, load_posture_config


class PostureAnalysisEngine:
    """
    Unified high-level posture analysis engine.

    Coordinates landmark preprocessing, ergonomic metric computation,
    personalized calibration, classification, and HUD visualization.
    Loads defaults from configs/posture_analyzer_config.json.
    """

    def __init__(
        self,
        use_smoothing: Optional[bool] = None,
        one_euro_min_cutoff: Optional[float] = None,
        one_euro_beta: Optional[float] = None,
        use_relative_calibration: Optional[bool] = None,
        classifier_sensitivity: Optional[float] = None,
        target_calibration_samples: Optional[int] = None,
        calibration_distance_cm: Optional[float] = None,
        enable_workspace_detection: Optional[bool] = None,
        config_path: Optional[str] = None,
    ):
        if config_path is not None:
            load_posture_config(config_path)
        cfg = get_posture_config()
        smooth_cfg = cfg.get("smoothing", {})
        oe_cfg = smooth_cfg.get("one_euro", {})
        calib_cfg = cfg.get("calibration", {})
        class_cfg = cfg.get("classification", {})
        ws_cfg = cfg.get("workspace_detection", {})
        hud_cfg = cfg.get("visualization_hud", {})

        self.use_smoothing = use_smoothing if use_smoothing is not None else smooth_cfg.get("enabled", True)
        self.enable_workspace_detection = enable_workspace_detection if enable_workspace_detection is not None else ws_cfg.get("enabled", True)
        self.hud_panel_width = hud_cfg.get("panel_width", 310)

        # 1. Landmark Smoother
        self.smoother = LandmarkSmoother(
            min_cutoff=one_euro_min_cutoff if one_euro_min_cutoff is not None else oe_cfg.get("min_cutoff", 1.0),
            beta=one_euro_beta if one_euro_beta is not None else oe_cfg.get("beta", 0.005),
        )

        # 2. Metrics Calculator
        self.metrics_calc = PostureMetricsCalculator()

        # 3. Distance Estimator
        self.distance_estimator = DistanceEstimator()

        # 4. Calibrator
        self.calibrator = PostureCalibrator(
            target_samples=target_calibration_samples if target_calibration_samples is not None else calib_cfg.get("target_samples", 90),
            calibration_distance_cm=calibration_distance_cm if calibration_distance_cm is not None else calib_cfg.get("calibration_distance_cm", 60.0),
            metrics_calculator=self.metrics_calc,
            distance_estimator=self.distance_estimator,
        )

        # 5. Classifier
        self.classifier = PostureClassifier(
            use_relative=use_relative_calibration if use_relative_calibration is not None else class_cfg.get("use_relative_when_calibrated", True),
            sensitivity=classifier_sensitivity if classifier_sensitivity is not None else class_cfg.get("default_sensitivity", 1.0),
        )

        # 6. Workspace Detector
        self.workspace_detector = WorkspaceDetector()

    # --------------------------------------------------------------------------
    # Core Processing Pipeline
    # --------------------------------------------------------------------------

    def process_keypoints(
        self,
        kps: UnifiedKeypoints,
        timestamp_s: Optional[float] = None,
    ) -> PostureState:
        """
        Process a single frame's unified keypoints through the full analysis pipeline.

        Args:
            kps: UnifiedKeypoints object with normalized pixel coordinates.
            timestamp_s: Timestamp in seconds (float). If None, uses current time.

        Returns:
            PostureState containing computed metrics, classification, and workspace details.
        """
        if timestamp_s is None:
            timestamp_s = time.time()

        # 1. Apply temporal smoothing (One-Euro Filter)
        if self.use_smoothing:
            smoothed_kps_dict = self.smoother.smooth(kps.keypoints, t=timestamp_s)
            kps = UnifiedKeypoints(
                format=kps.format,
                frame_width=kps.frame_width,
                frame_height=kps.frame_height,
                keypoints=smoothed_kps_dict,
            )

        # 2. Detect workspace surfaces (desk, screen) first so desk_y is available
        desk_y = -1
        screen_region = None
        if self.enable_workspace_detection:
            desk_y, screen_region = self.workspace_detector.detect(kps)

        # 3. Feed calibration if active
        if self.calibrator.is_calibrating:
            self.calibrator.feed_frame(kps, desk_y=desk_y)

        # 4. Compute ergonomic posture metrics (CVA, Shoulder Tilt / Level, Trunk Angle)
        metrics = self.metrics_calc.compute_all(kps)

        # 5. Estimate eye-to-screen distance
        metrics.eye_distance_cm = self.distance_estimator.estimate_distance(kps)

        # 6. Estimate eye-to-desk distance
        metrics.eye_to_desk_cm = self.distance_estimator.estimate_eye_to_desk(kps, desk_y)

        # 7. Classify posture state (absolute or relative to calibration)
        calibration_data = self.calibrator.calibration if self.calibrator.is_calibrated else None
        state = self.classifier.classify(metrics, calibration=calibration_data)

        # 8. Enrich state
        state.desk_line_y = desk_y
        state.screen_region = screen_region
        state.timestamp_ms = timestamp_s * 1000.0

        return state


    def process_mediapipe(
        self,
        landmarks: Any,
        frame_width: int,
        frame_height: int,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[PostureState, UnifiedKeypoints]:
        """
        Process MediaPipe Pose landmarks.

        Args:
            landmarks: MediaPipe PoseLandmarker result landmark list.
            frame_width: Width of the input frame.
            frame_height: Height of the input frame.
            timestamp_s: Optional timestamp in seconds.

        Returns:
            Tuple of (PostureState, UnifiedKeypoints).
        """
        kps = convert_mediapipe_to_unified(landmarks, frame_width, frame_height)
        state = self.process_keypoints(kps, timestamp_s=timestamp_s)
        return state, kps

    def process_coco(
        self,
        keypoints: Any,
        frame_width: int,
        frame_height: int,
        is_normalized: bool = False,
        is_yx_order: bool = False,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[PostureState, UnifiedKeypoints]:
        """
        Process COCO 17-keypoint outputs (MoveNet, YOLO-Pose).

        Args:
            keypoints: Array-like keypoints shape (17, 3) or (17, 2).
            frame_width: Width of the input frame.
            frame_height: Height of the input frame.
            is_normalized: True if coordinates are in [0, 1] range.
            is_yx_order: True for MoveNet format (y, x, score).
            timestamp_s: Optional timestamp in seconds.

        Returns:
            Tuple of (PostureState, UnifiedKeypoints).
        """
        kps = convert_coco_to_unified(
            keypoints,
            frame_width,
            frame_height,
            is_normalized=is_normalized,
            is_yx_order=is_yx_order,
        )
        state = self.process_keypoints(kps, timestamp_s=timestamp_s)
        return state, kps

    # --------------------------------------------------------------------------
    # HUD & Visualization
    # --------------------------------------------------------------------------

    def render_hud(
        self,
        frame: np.ndarray,
        state: PostureState,
        kps: Optional[UnifiedKeypoints] = None,
        draw_workspace: bool = True,
        panel_width: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw the posture analysis HUD and workspace visualization onto a video frame.

        Args:
            frame: BGR video frame (modified in-place).
            state: Current PostureState.
            kps: Optional UnifiedKeypoints for drawing skeleton angle arcs.
            draw_workspace: Whether to draw desk line and screen region.
            panel_width: Width of the right-hand HUD metrics panel.

        Returns:
            The modified frame.
        """
        panel_w = panel_width if panel_width is not None else self.hud_panel_width

        # Draw workspace surface / screen
        if draw_workspace and (state.desk_line_y > 0 or state.screen_region is not None):
            self.workspace_detector.draw_workspace(
                frame,
                desk_y=state.desk_line_y,
                screen_region=state.screen_region,
            )

        # Draw posture HUD panel and angle arcs
        draw_posture_hud(
            frame=frame,
            state=state,
            kps=kps,
            is_calibrating=self.calibrator.is_calibrating,
            calib_progress=self.calibrator.progress,
            panel_width=panel_w,
        )

        return frame


    # --------------------------------------------------------------------------
    # Calibration Controls
    # --------------------------------------------------------------------------

    def start_calibration(self):
        """Start auto-calibration for personalized posture baseline."""
        self.calibrator.start_calibration()

    def cancel_calibration(self):
        """Cancel ongoing auto-calibration."""
        self.calibrator.cancel_calibration()

    def reset_calibration(self):
        """Reset calibration back to uncalibrated absolute threshold mode."""
        self.calibrator.reset()

    @property
    def is_calibrating(self) -> bool:
        """Whether auto-calibration is currently collecting frames."""
        return self.calibrator.is_calibrating

    @property
    def is_calibrated(self) -> bool:
        """Whether a personalized calibration baseline is active."""
        return self.calibrator.is_calibrated

    @property
    def calibration_progress(self) -> float:
        """Auto-calibration progress (0.0 to 1.0)."""
        return self.calibrator.progress

    @property
    def calibration_data(self) -> CalibrationData:
        """Current calibration baseline data."""
        return self.calibrator.calibration

    def reset(self):
        """Reset all stateful components (filters, workspace tracker, calibration)."""
        self.smoother.reset()
        self.workspace_detector.reset()
        self.calibrator.reset()
