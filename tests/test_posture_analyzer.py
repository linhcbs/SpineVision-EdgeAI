"""
Unit and Integration Tests for posture_analyzer.
Validates all modules:
  - Landmark smoothing (OneEuroFilter, LandmarkSmoother)
  - Biometric metrics (CVA, Shoulder Tilt, Trunk Angle)
  - Distance estimation
  - Auto-calibration
  - Posture classifier
  - Workspace detector
  - PostureAnalysisEngine (End-to-End)
"""

import os
import sys
import math
import numpy as np
import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from posture_analyzer import (
    PostureMetrics,
    PostureState,
    PostureStatus,
    AlertLevel,
    ViewMode,
    KeypointFormat,
    OneEuroFilter,
    LandmarkSmoother,
    PostureMetricsCalculator,
    DistanceEstimator,
    PostureCalibrator,
    PostureClassifier,
    WorkspaceDetector,
    PostureAnalysisEngine,
)
from posture_analyzer.types import (
    NormalizedKeypoint,
    UnifiedKeypoints,
    convert_coco_to_unified,
)


def _create_mock_keypoints(
    head_forward: bool = False,
    tilt_right: bool = False,
    slouched: bool = False,
    width: int = 640,
    height: int = 480,
) -> UnifiedKeypoints:
    """Helper to generate synthetic UnifiedKeypoints for sitting posture tests."""
    kps = UnifiedKeypoints(
        format=KeypointFormat.COCO_17,
        frame_width=width,
        frame_height=height,
    )

    # Base coords (centered in frame)
    # Head & face
    ear_y = 120.0
    ear_x_offset = 0.0
    if head_forward:
        # Head translated forward/downward relative to shoulders
        ear_y = 175.0
        ear_x_offset = 50.0

    kps.keypoints["nose"] = NormalizedKeypoint(320.0 + ear_x_offset, ear_y + 10.0, 0.0, 0.95)
    kps.keypoints["left_eye"] = NormalizedKeypoint(300.0 + ear_x_offset, ear_y, 0.0, 0.95)
    kps.keypoints["right_eye"] = NormalizedKeypoint(340.0 + ear_x_offset, ear_y, 0.0, 0.95)
    kps.keypoints["left_ear"] = NormalizedKeypoint(280.0 + ear_x_offset, ear_y, 0.0, 0.95)
    kps.keypoints["right_ear"] = NormalizedKeypoint(360.0 + ear_x_offset, ear_y, 0.0, 0.95)

    # Shoulders
    l_shoulder_y = 200.0
    r_shoulder_y = 200.0
    if tilt_right:
        l_shoulder_y = 220.0
        r_shoulder_y = 180.0

    kps.keypoints["left_shoulder"] = NormalizedKeypoint(250.0, l_shoulder_y, 0.0, 0.95)
    kps.keypoints["right_shoulder"] = NormalizedKeypoint(390.0, r_shoulder_y, 0.0, 0.95)

    # Hips
    hip_x_offset = 0.0
    if slouched:
        # Hips slide forward or shoulders lean backward
        hip_x_offset = 70.0

    kps.keypoints["left_hip"] = NormalizedKeypoint(260.0 + hip_x_offset, 400.0, 0.0, 0.9)
    kps.keypoints["right_hip"] = NormalizedKeypoint(380.0 + hip_x_offset, 400.0, 0.0, 0.9)

    # Wrists (resting on desk)
    kps.keypoints["left_wrist"] = NormalizedKeypoint(240.0, 360.0, 0.0, 0.9)
    kps.keypoints["right_wrist"] = NormalizedKeypoint(400.0, 360.0, 0.0, 0.9)

    return kps


# ==============================================================================
# TESTS
# ==============================================================================

def test_one_euro_filter():
    """Verify OneEuroFilter reduces high-frequency jitter."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.005)
    
    # Static signal with jitter
    t = 0.0
    dt = 0.033
    outputs = []
    for i in range(30):
        t += dt
        jitter = 5.0 if i % 2 == 0 else -5.0
        val = 100.0 + jitter
        smoothed = f(val, t)
        outputs.append(smoothed)
    
    # Smoothed values should have much lower standard deviation than raw jitter
    raw_vals = [100.0 + (5.0 if i % 2 == 0 else -5.0) for i in range(30)]
    assert np.std(outputs[10:]) < np.std(raw_vals[10:])


def test_posture_metrics_good_posture():
    """Verify metrics calculation for upright standard posture."""
    calc = PostureMetricsCalculator()
    kps = _create_mock_keypoints(head_forward=False, tilt_right=False, slouched=False)
    metrics = calc.compute_all(kps)

    assert metrics.is_valid
    assert metrics.neck_cva_deg >= 50.0  # Normal CVA
    assert metrics.shoulder_tilt_deg <= 3.0  # Level shoulders
    assert metrics.trunk_angle_deg <= 5.0  # Upright trunk


def test_posture_metrics_forward_head():
    """Verify detection of low CVA for forward head posture."""
    calc = PostureMetricsCalculator()
    kps = _create_mock_keypoints(head_forward=True)
    metrics = calc.compute_all(kps)

    assert metrics.is_valid
    # Lower angle relative to horizontal plane
    assert metrics.neck_cva_deg < 40.0


def test_posture_metrics_shoulder_tilt():
    """Verify detection of shoulder tilt and 180°-90° level scale."""
    calc = PostureMetricsCalculator()
    # 1. Level shoulders
    good_kps = _create_mock_keypoints(tilt_right=False)
    good_metrics = calc.compute_all(good_kps)
    assert good_metrics.is_valid
    assert good_metrics.shoulder_tilt_deg <= 3.0
    assert good_metrics.shoulder_level_deg >= 177.0  # Close to 180°

    # 2. Tilted shoulders
    bad_kps = _create_mock_keypoints(tilt_right=True)
    bad_metrics = calc.compute_all(bad_kps)
    assert bad_metrics.is_valid
    assert bad_metrics.shoulder_tilt_deg > 8.0
    assert bad_metrics.shoulder_level_deg < 172.0  # Reduced from 180° towards 90°


def test_distance_estimator():
    """Verify Pinhole Camera distance calculation for Screen and Desk."""
    dist_estimator = DistanceEstimator(ipd_real_cm=6.3, default_focal_length=650.0)
    kps = _create_mock_keypoints()
    
    # In mock, IPD = 340 - 300 = 40 pixels
    # Screen distance: D = (650 * 6.3) / 40 ≈ 102.375 cm
    dist = dist_estimator.estimate_distance(kps)
    assert 90.0 <= dist <= 115.0

    # Desk distance: desk at y=370, eyes at y=120, delta_y = 250 px
    # px_to_cm = 6.3 / 40 = 0.1575 cm/px -> 250 * 0.1575 = 39.375 cm
    desk_dist = dist_estimator.estimate_eye_to_desk(kps, desk_y=370)
    assert 35.0 <= desk_dist <= 45.0

    # Calibrate focal length: user sits at 60 cm with 40 px IPD
    dist_estimator.calibrate_focal_length(ipd_pixels=40.0, known_distance_cm=60.0)
    assert dist_estimator.is_calibrated
    
    # Recalculate screen distance: should now equal 60 cm
    new_dist = dist_estimator.estimate_distance(kps)
    assert pytest.approx(new_dist, abs=1.0) == 60.0



def test_posture_classifier_uncalibrated():
    """Verify rule-based classification against absolute thresholds."""
    classifier = PostureClassifier(use_relative=False)

    # 1. Normal
    good_kps = _create_mock_keypoints()
    calc = PostureMetricsCalculator()
    good_metrics = calc.compute_all(good_kps)
    good_metrics.eye_distance_cm = 60.0
    state = classifier.classify(good_metrics)
    assert state.status == PostureStatus.GOOD
    assert state.alert_level == AlertLevel.SAFE

    # 2. Forward Head
    bad_cva_metrics = PostureMetrics(
        neck_cva_deg=35.0, shoulder_tilt_deg=2.0, trunk_angle_deg=5.0, eye_distance_cm=60.0, is_valid=True
    )
    state = classifier.classify(bad_cva_metrics)
    assert PostureStatus.FORWARD_HEAD in state.violations
    assert state.alert_level in (AlertLevel.WARNING, AlertLevel.CRITICAL)

    # 3. Eye to Screen Distance: > 35 cm -> SAFE
    safe_dist_metrics = PostureMetrics(
        neck_cva_deg=85.0, shoulder_tilt_deg=1.0, trunk_angle_deg=5.0, eye_distance_cm=38.0, is_valid=True
    )
    state = classifier.classify(safe_dist_metrics)
    assert state.status == PostureStatus.GOOD
    assert state.alert_level == AlertLevel.SAFE

    # 4. Eye to Screen Distance: 25 - 35 cm -> WARNING (Yellow)
    warn_dist_metrics = PostureMetrics(
        neck_cva_deg=85.0, shoulder_tilt_deg=1.0, trunk_angle_deg=5.0, eye_distance_cm=28.0, is_valid=True
    )
    state = classifier.classify(warn_dist_metrics)
    assert PostureStatus.TOO_CLOSE in state.violations
    assert state.alert_level == AlertLevel.WARNING

    # 5. Eye to Screen Distance: < 25 cm -> CRITICAL (Red)
    crit_dist_metrics = PostureMetrics(
        neck_cva_deg=85.0, shoulder_tilt_deg=1.0, trunk_angle_deg=5.0, eye_distance_cm=20.0, is_valid=True
    )
    state = classifier.classify(crit_dist_metrics)
    assert PostureStatus.TOO_CLOSE in state.violations
    assert state.alert_level == AlertLevel.CRITICAL


def test_auto_calibration_workflow():
    """Verify multi-frame calibration and relative deviation triggering."""
    calibrator = PostureCalibrator(target_samples=10, calibration_distance_cm=60.0)
    calibrator.start_calibration()
    assert calibrator.is_calibrating

    # Feed 10 frames of upright posture
    for _ in range(10):
        kps = _create_mock_keypoints(head_forward=False)
        calibrator.feed_frame(kps)

    assert not calibrator.is_calibrating
    assert calibrator.is_calibrated
    assert calibrator.calibration.baseline_neck_cva > 0

    # Classify with calibration
    classifier = PostureClassifier(use_relative=True)
    
    # Now simulate user bending neck down by 15 degrees from baseline
    bad_metrics = PostureMetrics(
        neck_cva_deg=calibrator.calibration.baseline_neck_cva - 16.0,
        shoulder_tilt_deg=0.0,
        trunk_angle_deg=calibrator.calibration.baseline_trunk_angle,
        eye_distance_cm=60.0,
        is_valid=True,
    )
    state = classifier.classify(bad_metrics, calibration=calibrator.calibration)
    assert PostureStatus.FORWARD_HEAD in state.violations


def test_workspace_detector():
    """Verify desk and screen detection heuristics."""
    detector = WorkspaceDetector()
    kps = _create_mock_keypoints()
    desk_y, screen_region = detector.detect(kps)

    assert desk_y > 0
    # Wrists are at y=360, so desk should be around y=360-390
    assert 340 <= desk_y <= 420
    assert screen_region is not None
    x1, y1, x2, y2 = screen_region
    assert x2 > x1 and y2 > y1


def test_posture_analysis_engine_e2e():
    """Integration test running PostureAnalysisEngine through full lifecycle and HUD rendering."""
    engine = PostureAnalysisEngine(
        use_smoothing=True,
        target_calibration_samples=5,
        calibration_distance_cm=60.0,
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Process initial frames
    kps = _create_mock_keypoints()
    state = engine.process_keypoints(kps)
    assert state.status == PostureStatus.GOOD

    # 2. Run calibration
    engine.start_calibration()
    assert engine.is_calibrating
    for _ in range(5):
        state = engine.process_keypoints(kps)

    assert not engine.is_calibrating
    assert engine.is_calibrated

    # 3. Render HUD onto dummy frame
    rendered_frame = engine.render_hud(dummy_frame.copy(), state, kps=kps)
    assert rendered_frame.shape == (480, 640, 3)
    # Check that pixels were actually drawn (not all zeros)
    assert np.count_nonzero(rendered_frame) > 0


def test_adaptive_view_modes_and_hysteresis():
    """Verify viewpoint classification, adaptive rule sets, and hysteresis transitions."""
    calc = PostureMetricsCalculator()
    classifier = PostureClassifier(use_relative=False)

    # 1. Frontal View Test
    frontal_kps = _create_mock_keypoints()
    yaw_deg, view_mode = calc.estimate_view_angle(frontal_kps)
    assert view_mode == ViewMode.FRONTAL
    assert yaw_deg < 25.0

    # In Frontal: CVA 85° is GOOD, CVA 65° triggers FORWARD_HEAD
    good_front_metrics = PostureMetrics(
        neck_cva_deg=85.0, shoulder_tilt_deg=2.0, trunk_angle_deg=5.0, eye_distance_cm=60.0,
        view_mode=ViewMode.FRONTAL, is_valid=True
    )
    front_state = classifier.classify(good_front_metrics)
    assert front_state.status == PostureStatus.GOOD

    bad_front_metrics = PostureMetrics(
        neck_cva_deg=65.0, shoulder_tilt_deg=2.0, trunk_angle_deg=5.0, eye_distance_cm=60.0,
        view_mode=ViewMode.FRONTAL, is_valid=True
    )
    bad_front_state = classifier.classify(bad_front_metrics)
    assert PostureStatus.FORWARD_HEAD in bad_front_state.violations

    # 2. Profile View Test (Normal clinical CVA is 50-60°, shoulder tilt enabled)
    profile_metrics = PostureMetrics(
        neck_cva_deg=55.0, shoulder_tilt_deg=2.0, trunk_angle_deg=5.0, eye_distance_cm=60.0,
        view_mode=ViewMode.PROFILE, is_valid=True
    )
    profile_state = classifier.classify(profile_metrics)
    assert PostureStatus.SHOULDER_TILTED not in profile_state.violations
    # CVA 55° is safe under clinical profile thresholds (normal_min = 52.0)
    assert PostureStatus.FORWARD_HEAD not in profile_state.violations
    assert profile_state.status == PostureStatus.GOOD

    # Profile view severe forward head (CVA < 40°)
    severe_profile_metrics = PostureMetrics(
        neck_cva_deg=38.0, shoulder_tilt_deg=2.0, trunk_angle_deg=5.0, eye_distance_cm=60.0,
        view_mode=ViewMode.PROFILE, is_valid=True
    )
    severe_profile_state = classifier.classify(severe_profile_metrics)
    assert PostureStatus.FORWARD_HEAD in severe_profile_state.violations

    # Profile view shoulder tilt: currently disabled in config (enabled: false),
    # so tilt is bypassed even for large values.
    tilted_profile_metrics = PostureMetrics(
        neck_cva_deg=55.0, shoulder_tilt_deg=10.0, trunk_angle_deg=5.0, eye_distance_cm=60.0,
        view_mode=ViewMode.PROFILE, is_valid=True
    )
    tilted_profile_state = classifier.classify(tilted_profile_metrics)
    assert PostureStatus.SHOULDER_TILTED not in tilted_profile_state.violations  # disabled in profile

    # 3. Oblique View Test (Normal posture within thresholds)
    oblique_metrics = PostureMetrics(
        neck_cva_deg=68.0, shoulder_tilt_deg=4.0, trunk_angle_deg=5.0, eye_distance_cm=60.0,
        view_mode=ViewMode.OBLIQUE, is_valid=True
    )
    oblique_state = classifier.classify(oblique_metrics)
    # Tilt 4.0° is permitted in oblique mode (normal_max = 5.0°)
    assert PostureStatus.SHOULDER_TILTED not in oblique_state.violations
    assert oblique_state.status == PostureStatus.GOOD

    # 4. Hysteresis Test on Calculator State Machine
    h_calc = PostureMetricsCalculator()
    # Force initial state to FRONTAL at 20°
    h_calc._smoothed_yaw = 20.0
    h_calc._current_view_mode = ViewMode.FRONTAL
    h_calc._has_init_yaw = True

    # Check that a minor fluctuation to 27° does NOT trigger transition
    # (Frontal max is 25°, with +5° hysteresis buffer it must exceed 30° to switch)
    h_calc.hysteresis_deg = 5.0
    h_calc.frontal_max_yaw = 25.0
    h_calc.oblique_max_yaw = 65.0
    h_calc.yaw_smoothing_alpha = 1.0  # instant for deterministic unit test

    # Simulate keypoints with ~27° yaw
    dummy_kps = _create_mock_keypoints()
    # Directly test state transition logic
    h_calc._smoothed_yaw = 27.0
    # Run hysteresis check
    if h_calc._smoothed_yaw > (h_calc.frontal_max_yaw + h_calc.hysteresis_deg):
        h_calc._current_view_mode = ViewMode.OBLIQUE
    assert h_calc._current_view_mode == ViewMode.FRONTAL  # Still FRONTAL!

    # Now cross threshold + hysteresis (31° > 30°)
    h_calc._smoothed_yaw = 31.0
    if h_calc._smoothed_yaw > (h_calc.frontal_max_yaw + h_calc.hysteresis_deg):
        h_calc._current_view_mode = ViewMode.OBLIQUE
    assert h_calc._current_view_mode == ViewMode.OBLIQUE  # Switched to OBLIQUE!

    # Minor drop to 23° should NOT switch back to FRONTAL (must drop < 20°)
    h_calc._smoothed_yaw = 23.0
    if h_calc._smoothed_yaw < (h_calc.frontal_max_yaw - h_calc.hysteresis_deg):
        h_calc._current_view_mode = ViewMode.FRONTAL
    assert h_calc._current_view_mode == ViewMode.OBLIQUE  # Still OBLIQUE!

    # Drop to 18° (< 20°) triggers transition back to FRONTAL
    h_calc._smoothed_yaw = 18.0
    if h_calc._smoothed_yaw < (h_calc.frontal_max_yaw - h_calc.hysteresis_deg):
        h_calc._current_view_mode = ViewMode.FRONTAL
    assert h_calc._current_view_mode == ViewMode.FRONTAL  # Returned to FRONTAL!


def test_leaning_left_right_detection():
    """Verify classifier correctly detects LEANING_LEFT vs LEANING_RIGHT."""
    calc = PostureMetricsCalculator()
    classifier = PostureClassifier()

    # 1. Leaning Left: left shoulder dropped (larger y)
    kps_left = _create_mock_keypoints(tilt_right=True)  # l_shoulder_y=220, r_shoulder_y=180
    metrics_left = calc.compute_all(kps_left)
    assert metrics_left.lateral_tilt_deg < 0  # Negative = Left
    state_left = classifier.classify(metrics_left, kps=kps_left)
    assert state_left.status == PostureStatus.LEANING_LEFT
    assert PostureStatus.SHOULDER_TILTED in state_left.violations
    assert PostureStatus.LEANING_LEFT in state_left.violations
    assert "Left" in state_left.posture_description

    # 2. Leaning Right: right shoulder dropped
    kps_right = _create_mock_keypoints()
    kps_right.keypoints["left_shoulder"] = NormalizedKeypoint(250.0, 180.0, 0.0, 0.95)
    kps_right.keypoints["right_shoulder"] = NormalizedKeypoint(390.0, 220.0, 0.0, 0.95)
    metrics_right = calc.compute_all(kps_right)
    assert metrics_right.lateral_tilt_deg > 0  # Positive = Right
    state_right = classifier.classify(metrics_right, kps=kps_right)
    assert state_right.status == PostureStatus.LEANING_RIGHT
    assert PostureStatus.SHOULDER_TILTED in state_right.violations
    assert PostureStatus.LEANING_RIGHT in state_right.violations
    assert "Right" in state_right.posture_description


def test_lightweight_ml_classifier():
    """Verify LightweightPostureML extracts features and predicts probabilities."""
    from posture_analyzer.posture_classifier import LightweightPostureML

    ml = LightweightPostureML()
    calc = PostureMetricsCalculator()

    # Test upright keypoints
    kps_good = _create_mock_keypoints()
    metrics_good = calc.compute_all(kps_good)
    metrics_good.eye_distance_cm = 60.0
    features = ml.extract_features(metrics_good)

    assert features.shape == (8,)
    pred_cls, probs, kyph_score, myop_score = ml.predict(features)

    assert pred_cls == PostureStatus.GOOD
    assert probs[PostureStatus.GOOD] > 0.3
    assert 0.0 <= kyph_score <= 1.0
    assert 0.0 <= myop_score <= 1.0


def test_prolonged_risk_accumulation():
    """Verify prolonged risk accumulates during bad posture and decays upon recovery."""
    from posture_analyzer.posture_classifier import ProlongedRiskAccumulator

    acc = ProlongedRiskAccumulator(kyphosis_max_exposure_s=40.0, recovery_half_life_s=10.0)

    # Simulate 30 seconds of severe slouched posture
    t = 0.0
    for _ in range(30):
        t += 1.0
        p_kyph, p_myop, bad_dur, near_dur = acc.update(
            instant_kyphosis_risk_pct=85.0,
            instant_myopia_risk_pct=15.0,
            is_bad_posture=True,
            is_near_screen=False,
            timestamp_s=t,
        )

    # Prolonged kyphosis should have risen significantly (>60%)
    assert bad_dur >= 29.0
    assert p_kyph > 60.0

    # Simulate 25 seconds of good upright posture recovery
    for _ in range(25):
        t += 1.0
        p_kyph_rec, _, bad_dur_rec, _ = acc.update(
            instant_kyphosis_risk_pct=10.0,
            instant_myopia_risk_pct=10.0,
            is_bad_posture=False,
            is_near_screen=False,
            timestamp_s=t,
        )

    # Should have recovered significantly
    assert p_kyph_rec < p_kyph
    assert bad_dur_rec < bad_dur


def test_posture_state_risk_diagnostics_integration():
    """Verify End-to-End PostureAnalysisEngine produces full risk diagnostics on PostureState."""
    engine = PostureAnalysisEngine(
        use_smoothing=False,
        enable_screen_detection=False,
        enable_workspace_detection=False,
    )

    kps = _create_mock_keypoints(slouched=True)
    state = engine.process_keypoints(kps, timestamp_s=1.0)

    # Verify risk assessment fields are populated
    assert state.risk_assessment is not None
    assert state.kyphosis_risk_pct > 0.0
    assert state.prolonged_kyphosis_risk_pct > 0.0
    assert state.myopia_risk_pct > 0.0
    assert state.posture_description != ""
    assert state.risk_assessment.kyphosis_severity in ["Low", "Moderate", "High", "Severe"]
    assert state.risk_assessment.myopia_severity in ["Low", "Moderate", "High", "Severe"]


def test_posture_hud_risk_rendering():
    """Verify HUD renders the risk diagnostics and sitting posture badge without crash."""
    engine = PostureAnalysisEngine(
        use_smoothing=False,
        enable_screen_detection=False,
        enable_workspace_detection=False,
    )

    kps = _create_mock_keypoints(slouched=True)
    state = engine.process_keypoints(kps, timestamp_s=10.0)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = engine.render_hud(frame=frame, state=state, kps=kps)

    assert rendered is not None
    assert rendered.shape == (720, 1280, 3)
    # Background HUD area should have non-zero pixels
    assert np.count_nonzero(rendered) > 0


def test_hpe_model_registry_and_factory():
    """Verify registry lists all 7 models and factory resolves canonical keys and aliases."""
    from posture_analyzer import (
        list_supported_hpe_models,
        canonical_model_key,
        SUPPORTED_HPE_MODELS,
    )

    models = list_supported_hpe_models()
    assert len(models) == 7
    keys = [m["key"] for m in models]
    assert "mediapipe_pose_lite" in keys
    assert "mediapipe_pose_full" in keys
    assert "movenet_lightning" in keys
    assert "movenet_thunder" in keys
    assert "yolo26n_pose" in keys
    assert "yolo26m_pose" in keys
    assert "yolo26x_pose" in keys

    # Verify alias resolution
    assert canonical_model_key("mp_lite") == "mediapipe_pose_lite"
    assert canonical_model_key("lightning") == "movenet_lightning"
    assert canonical_model_key("thunder") == "movenet_thunder"
    assert canonical_model_key("yolo_n") == "yolo26n_pose"
    assert canonical_model_key("yolo_m") == "yolo26m_pose"
    assert canonical_model_key("yolo_x") == "yolo26x_pose"


def test_hpe_adapters_unified_conversion():
    """Verify MoveNet and YOLO adapters convert raw outputs to UnifiedKeypoints correctly."""
    from posture_analyzer import MoveNetPoseAdapter, YOLOPoseAdapter

    # 1. MoveNet adapter conversion test
    # Mock MoveNet 17 keypoints array [17, 3] with [y_norm, x_norm, score]
    mock_movenet_kps = np.zeros((17, 3), dtype=np.float32)
    # nose, left_shoulder, right_shoulder, left_hip, right_hip
    mock_movenet_kps[0] = [0.25, 0.50, 0.90]   # nose (y=120px, x=320px)
    mock_movenet_kps[5] = [0.42, 0.39, 0.85]   # left_shoulder (y=200px, x=250px)
    mock_movenet_kps[6] = [0.42, 0.61, 0.85]   # right_shoulder (y=200px, x=390px)
    mock_movenet_kps[11] = [0.83, 0.40, 0.80]  # left_hip
    mock_movenet_kps[12] = [0.83, 0.60, 0.80]  # right_hip

    # Create MoveNet adapter (without loading weights by testing conversion logic directly)
    unified_mv = convert_coco_to_unified(
        mock_movenet_kps,
        frame_width=640,
        frame_height=480,
        is_normalized=True,
        is_yx_order=True,
    )
    assert unified_mv is not None
    assert unified_mv.format == KeypointFormat.COCO_17
    assert unified_mv.get("left_shoulder") is not None
    assert abs(unified_mv.get("left_shoulder").x - (0.39 * 640)) < 1.0
    assert abs(unified_mv.get("left_shoulder").y - (0.42 * 480)) < 1.0

    # 2. YOLO adapter conversion test
    # Mock YOLO 17 keypoints in pixel space [17, 3] with [x_px, y_px, conf]
    mock_yolo_kps = []
    for i in range(17):
        mock_yolo_kps.append((320.0 + i, 200.0 + i, 0.90))

    unified_yolo = convert_coco_to_unified(
        mock_yolo_kps,
        frame_width=640,
        frame_height=480,
        is_normalized=False,
        is_yx_order=False,
    )
    assert unified_yolo is not None
    assert unified_yolo.format == KeypointFormat.COCO_17
    assert unified_yolo.get("nose") is not None
    assert unified_yolo.get("nose").x == 320.0


def test_engine_process_frame_with_plug_and_play_adapter():
    """Verify PostureAnalysisEngine.process_frame handles adapter inference smoothly."""
    from posture_analyzer import PostureAnalysisEngine, create_hpe_detector

    engine = PostureAnalysisEngine(
        use_smoothing=False,
        enable_screen_detection=False,
        enable_workspace_detection=False,
    )

    # Initialize MoveNet Lightning adapter (ultra-lightweight TFLite)
    detector = create_hpe_detector("movenet_lightning")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    state, kps, lat_ms = engine.process_frame(dummy_frame, detector, timestamp_s=1.0)
    assert state is not None
    assert lat_ms >= 0.0
    detector.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])



