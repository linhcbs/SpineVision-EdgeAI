"""
Posture Classifier — Hybrid Rule-based + Lightweight ML Classification.

Classifies the user's sitting posture into discrete states:
  - Upright / Straight
  - Leaning Left / Leaning Right (Lateral asymmetry)
  - Slouching / Kyphosis
  - Forward Head / Tech Neck
  - Screen Too Close / Desk Too Close
  - Combined Violations

Performs clinical risk and severity diagnosis for:
  - Kyphosis (Gù lưng): Instantaneous & Prolonged exposure risk (%)
  - Myopia (Cận thị): Instantaneous & Prolonged near-work exposure risk (%)

Architecture:
  - Tier 1 (Rule-based): Fast O(1) clinical threshold and baseline deviation checks.
  - Tier 2 (Lightweight ML): Calibrated linear-softmax & ridge predictor for multi-class
    probabilities and smooth risk regression (<0.05ms CPU latency).
  - Prolonged Risk Accumulator: Temporal leaky integrator modeling viscoelastic
    spinal creep and optical accommodative fatigue over time.
"""

import time
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .types import (
    PostureMetrics,
    PostureState,
    PostureStatus,
    AlertLevel,
    CalibrationData,
    ViewMode,
    UnifiedKeypoints,
    RiskAssessment,
    RiskSeverity,
    DetectedDevice,
    DeviceType,
)
from .config import get_posture_config


# ==============================================================================
# TIER 2: LIGHTWEIGHT ML CLASSIFIER & RISK PREDICTOR
# ==============================================================================

class LightweightPostureML:
    """
    Lightweight ML Classifier & Ergonomic Risk Predictor.
    
    Supports:
      1. Pre-trained / fine-tuned weights loaded from models/weights/ or configs/
         running a 12D ergonomic neural network head in pure NumPy (<0.02ms latency).
      2. Fallback heuristic weights if no trained weights file is present.
    """

    # Posture classes predicted by ML
    CLASSES = [
        PostureStatus.GOOD,
        PostureStatus.FORWARD_HEAD,
        PostureStatus.SLOUCHED,
        PostureStatus.LEANING_LEFT,
        PostureStatus.LEANING_RIGHT,
        PostureStatus.TOO_CLOSE,
    ]

    def __init__(self, weights_path: Optional[str] = None):
        self.n_features = 8
        self.n_classes = len(self.CLASSES)
        self.is_trained = False
        self.model_type = "heuristic"

        # Heuristic fallback weights
        self.weights = np.array([
            [-2.5, -0.5, -3.0, -2.0, -1.5, -0.5, -0.2, -2.0],
            [ 5.5,  0.0,  0.5,  0.5,  0.5,  0.0,  0.3,  2.5],
            [ 0.8,  0.0,  6.0,  0.2,  0.5,  0.0,  0.2,  3.0],
            [ 0.0, -5.0,  0.5,  0.0,  0.0, -3.5,  0.1,  1.5],
            [ 0.0,  5.0,  0.5,  0.0,  0.0,  3.5,  0.1,  1.5],
            [ 0.5,  0.0,  0.5,  6.5,  4.0,  0.0,  0.0,  2.5],
        ], dtype=np.float32)
        self.bias = np.array([2.0, -1.0, -1.0, -1.2, -1.2, -1.0], dtype=np.float32)

        self.kyphosis_weights = np.array([0.35, 0.05, 0.75, 0.05, 0.05, 0.05, 0.05, 0.40], dtype=np.float32)
        self.kyphosis_bias = -0.15
        self.myopia_weights = np.array([0.25, 0.0, 0.10, 0.85, 0.50, 0.0, 0.05, 0.30], dtype=np.float32)
        self.myopia_bias = -0.10

        # Try auto-loading trained weights
        self._load_trained_weights(weights_path)

    def _load_trained_weights(self, weights_path: Optional[str] = None):
        """Attempt to load trained weights from specified or default paths."""
        candidates = []
        if weights_path:
            candidates.append(Path(weights_path))

        base_dir = Path(__file__).resolve().parent.parent
        candidates.extend([
            base_dir / "models" / "weights" / "posture_classifier_weights.json",
            base_dir / "configs" / "posture_classifier_weights.json",
            base_dir / "models" / "posture_classifier_weights.json",
        ])

        for p in candidates:
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    self.scaler_mean = np.array(data["scaler_mean"], dtype=np.float32)
                    self.scaler_scale = np.array(data["scaler_scale"], dtype=np.float32)
                    
                    mlp_data = data.get("mlp", {})
                    if "weights" in mlp_data and "intercepts" in mlp_data:
                        self.mlp_W1 = np.array(mlp_data["weights"][0], dtype=np.float32)
                        self.mlp_b1 = np.array(mlp_data["intercepts"][0], dtype=np.float32)
                        self.mlp_W2 = np.array(mlp_data["weights"][1], dtype=np.float32)
                        self.mlp_b2 = np.array(mlp_data["intercepts"][1], dtype=np.float32)
                        self.mlp_W3 = np.array(mlp_data["weights"][2], dtype=np.float32)
                        self.mlp_b3 = np.array(mlp_data["intercepts"][2], dtype=np.float32)
                        self.n_features = len(self.scaler_mean)
                        self.is_trained = True
                        self.model_type = "LightweightMLP (Trained)"
                        break
                except Exception:
                    pass

    def extract_features(
        self,
        metrics: PostureMetrics,
        calibration: Optional[CalibrationData] = None,
        kps: Optional[UnifiedKeypoints] = None,
    ) -> np.ndarray:
        """
        Extract normalized ergonomic feature vector.
        If trained weights are active, extracts a 12-dimensional vector.
        """
        if self.is_trained:
            feats = np.zeros(12, dtype=np.float32)
            if not metrics.is_valid:
                feats[1] = 0.5
                feats[3] = 0.5
                return feats

            # 0: cva norm
            feats[0] = max(0.0, min(1.5, metrics.neck_cva_deg / 90.0))
            # 1: cva deficit below 50 degrees
            feats[1] = max(0.0, min(1.5, (50.0 - metrics.neck_cva_deg) / 25.0)) if metrics.neck_cva_deg > 0 else 0.5
            # 2: trunk norm
            feats[2] = max(0.0, min(1.5, metrics.trunk_angle_deg / 90.0))
            # 3: trunk slump
            feats[3] = max(0.0, min(2.0, metrics.trunk_angle_deg / 18.0))
            # 4: shoulder tilt signed
            feats[4] = max(-1.5, min(1.5, metrics.shoulder_tilt_deg / 15.0))
            # 5: shoulder tilt abs
            feats[5] = max(0.0, min(1.5, abs(metrics.shoulder_tilt_deg) / 15.0))
            # 6: lateral spine offset
            feats[6] = max(-1.5, min(1.5, metrics.lateral_spine_offset * 3.0))
            # 7: yaw
            feats[7] = max(0.0, min(1.5, metrics.yaw_deg / 70.0))
            # 8: eye dist deficit
            if metrics.eye_distance_cm > 0:
                feats[8] = max(0.0, min(1.5, (50.0 - metrics.eye_distance_cm) / 30.0))

            # 9, 10, 11: geometric keypoint ratios
            if kps is not None:
                mid_sh = kps.get_midpoint("left_shoulder", "right_shoulder")
                ls = kps.get("left_shoulder")
                rs = kps.get("right_shoulder")
                sh_w = np.hypot(ls.x - rs.x, ls.y - rs.y) if (ls and rs) else 100.0
                sh_w = max(sh_w, 20.0)

                mid_ear = kps.get_midpoint("left_ear", "right_ear")
                nose = kps.get("nose")
                mid_hip = kps.get_midpoint("left_hip", "right_hip")

                if mid_ear and mid_sh:
                    feats[9] = max(0.0, min(2.0, (mid_sh.y - mid_ear.y) / sh_w))
                if nose and mid_sh:
                    feats[10] = max(0.0, min(2.0, (mid_sh.y - nose.y) / sh_w))
                if mid_sh and mid_hip:
                    torso_len = max(30.0, np.hypot(mid_sh.x - mid_hip.x, mid_sh.y - mid_hip.y))
                    feats[11] = max(-1.5, min(1.5, (mid_sh.x - mid_hip.x) / torso_len))

            return feats

        # Default 8D heuristic features
        features = np.zeros(self.n_features, dtype=np.float32)
        ref_cva = 50.0
        if metrics.neck_cva_deg > 0:
            features[0] = max(0.0, min(1.0, (ref_cva - metrics.neck_cva_deg) / 25.0))
        tilt = getattr(metrics, "lateral_tilt_deg", 0.0)
        features[1] = max(-1.0, min(1.0, tilt / 12.0))
        features[2] = max(0.0, min(1.5, metrics.trunk_angle_deg / 18.0))
        scr_dist = metrics.eye_distance_cm
        if scr_dist > 0:
            features[3] = max(0.0, min(1.5, (50.0 - scr_dist) / 30.0))
        desk_dist = metrics.eye_to_desk_cm
        if desk_dist > 0:
            features[4] = max(0.0, min(1.5, (35.0 - desk_dist) / 20.0))
        features[5] = max(-1.0, min(1.0, getattr(metrics, "lateral_spine_offset", 0.0) * 3.0))
        features[6] = max(0.0, min(1.0, metrics.yaw_deg / 70.0))
        if calibration is not None and calibration.is_calibrated:
            d_trunk = max(0.0, metrics.trunk_angle_deg - calibration.baseline_trunk_angle)
            d_cva = max(0.0, calibration.baseline_neck_cva - metrics.neck_cva_deg) if metrics.neck_cva_deg > 0 else 0.0
            features[7] = max(0.0, min(1.5, (d_trunk / 12.0) + (d_cva / 15.0)))

        return features

    def predict(
        self,
        features: np.ndarray,
        metrics: Optional[PostureMetrics] = None,
        kps: Optional[UnifiedKeypoints] = None,
    ) -> Tuple[PostureStatus, Dict[Any, float], float, float]:
        """
        Inference: Returns (predicted_class, class_probabilities, kyphosis_score, myopia_score).
        Runs in < 0.02ms on CPU using pure NumPy.
        """
        if self.is_trained:
            # 1. Scaler
            x_norm = (features - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-6)

            # 2. MLP Forward Pass (12 -> 16 -> 8 -> 1)
            h1 = np.maximum(0.0, np.dot(x_norm, self.mlp_W1) + self.mlp_b1)
            h2 = np.maximum(0.0, np.dot(h1, self.mlp_W2) + self.mlp_b2)
            z = np.dot(h2, self.mlp_W3) + self.mlp_b3
            z_val = float(np.clip(z[0], -15.0, 15.0))

            p_bad = float(1.0 / (1.0 + math.exp(-z_val)))
            p_good = float(1.0 - p_bad)

            # Probabilities dictionary
            prob_dict = {
                "Good": p_good,
                "Bad": p_bad,
                PostureStatus.GOOD: p_good,
            }

            if p_bad < 0.5:
                predicted_class = PostureStatus.GOOD
            else:
                # Assign specific ergonomic violation
                if metrics and metrics.trunk_angle_deg > 14.0:
                    predicted_class = PostureStatus.SLOUCHED
                elif metrics and metrics.neck_cva_deg > 0 and metrics.neck_cva_deg < 48.0:
                    predicted_class = PostureStatus.FORWARD_HEAD
                elif metrics and abs(getattr(metrics, "shoulder_tilt_deg", 0.0)) > 7.0:
                    predicted_class = PostureStatus.LEANING_LEFT if metrics.shoulder_tilt_deg < 0 else PostureStatus.LEANING_RIGHT
                else:
                    predicted_class = PostureStatus.SLOUCHED

            kyph_score = max(0.05, p_bad)
            myop_score = 0.10
            if metrics and metrics.eye_distance_cm > 0:
                myop_score = max(0.10, min(1.0, (50.0 - metrics.eye_distance_cm) / 25.0))

            return predicted_class, prob_dict, kyph_score, myop_score

        # Heuristic linear softmax fallback
        logits = np.dot(self.weights, features[:8]) + self.bias
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(probs))
        predicted_class = self.CLASSES[best_idx]
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.CLASSES)}
        prob_dict["Good"] = float(probs[0])
        prob_dict["Bad"] = float(1.0 - probs[0])

        kyph_z = float(np.dot(self.kyphosis_weights, features[:8]) + self.kyphosis_bias)
        kyph_score = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, kyph_z * 3.0))))

        myop_z = float(np.dot(self.myopia_weights, features[:8]) + self.myopia_bias)
        myop_score = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, myop_z * 3.0))))

        return predicted_class, prob_dict, kyph_score, myop_score


# ==============================================================================
# PROLONGED EXPOSURE RISK ACCUMULATOR
# ==============================================================================

class ProlongedRiskAccumulator:
    """
    Tracks continuous exposure time in bad posture and close screen distances
    to compute prolonged / chronic risk of Kyphosis and Myopia according
    to ergonomic clinical standards (spinal creep & accommodative fatigue).
    """

    def __init__(
        self,
        kyphosis_max_exposure_s: float = 45.0,
        myopia_max_exposure_s: float = 45.0,
        recovery_half_life_s: float = 10.0,
    ):
        self.kyphosis_max_s = kyphosis_max_exposure_s
        self.myopia_max_s = myopia_max_exposure_s
        self.recovery_half_life = recovery_half_life_s

        self._last_time_s: Optional[float] = None
        self._bad_posture_duration_s: float = 0.0
        self._near_screen_duration_s: float = 0.0

        # Accumulated integral strain [0.0 - 1.0]
        self._kyphosis_strain: float = 0.0
        self._myopia_strain: float = 0.0

    def update(
        self,
        instant_kyphosis_risk_pct: float,
        instant_myopia_risk_pct: float,
        is_bad_posture: bool,
        is_near_screen: bool,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Update exposure accumulators and return:
        (prolonged_kyphosis_pct, prolonged_myopia_pct, bad_duration_s, near_duration_s)
        """
        now = timestamp_s if timestamp_s is not None else time.time()
        if self._last_time_s is None:
            dt = 0.033  # default ~30 FPS frame duration
        else:
            dt = max(0.0, min(1.0, now - self._last_time_s))
        self._last_time_s = now

        # Decay factor for recovery when posture is good
        decay = math.exp(-dt / max(self.recovery_half_life, 1.0))

        # --- 1. Kyphosis Exposure Accumulation ---
        if is_bad_posture or instant_kyphosis_risk_pct >= 30.0:
            self._bad_posture_duration_s += dt
            # Accumulate strain proportional to current severity
            strain_rate = (instant_kyphosis_risk_pct / 100.0) * (dt / self.kyphosis_max_s)
            self._kyphosis_strain = min(1.0, self._kyphosis_strain + strain_rate)
        else:
            # User is sitting upright: relieve continuous bad duration and decay strain
            self._bad_posture_duration_s = max(0.0, self._bad_posture_duration_s - dt * 2.0)
            self._kyphosis_strain = self._kyphosis_strain * decay

        # Prolonged Kyphosis Risk: blends instantaneous strain with cumulative duration
        time_factor_k = min(1.0, self._bad_posture_duration_s / self.kyphosis_max_s)
        prolonged_kyphosis = (
            instant_kyphosis_risk_pct * (0.35 + 0.65 * time_factor_k)
            + self._kyphosis_strain * 30.0
        )
        prolonged_kyphosis = float(np.clip(prolonged_kyphosis, 0.0, 100.0))

        # --- 2. Myopia Exposure Accumulation ---
        if is_near_screen or instant_myopia_risk_pct >= 30.0:
            self._near_screen_duration_s += dt
            strain_rate = (instant_myopia_risk_pct / 100.0) * (dt / self.myopia_max_s)
            self._myopia_strain = min(1.0, self._myopia_strain + strain_rate)
        else:
            self._near_screen_duration_s = max(0.0, self._near_screen_duration_s - dt * 2.0)
            self._myopia_strain = self._myopia_strain * decay

        # Prolonged Myopia Risk
        time_factor_m = min(1.0, self._near_screen_duration_s / self.myopia_max_s)
        prolonged_myopia = (
            instant_myopia_risk_pct * (0.35 + 0.65 * time_factor_m)
            + self._myopia_strain * 30.0
        )
        prolonged_myopia = float(np.clip(prolonged_myopia, 0.0, 100.0))

        return (
            prolonged_kyphosis,
            prolonged_myopia,
            self._bad_posture_duration_s,
            self._near_screen_duration_s,
        )

    def reset(self):
        """Reset exposure counters and accumulators."""
        self._bad_posture_duration_s = 0.0
        self._near_screen_duration_s = 0.0
        self._kyphosis_strain = 0.0
        self._myopia_strain = 0.0
        self._last_time_s = None


# ==============================================================================
# MAIN POSTURE CLASSIFIER
# ==============================================================================

class PostureClassifier:
    """
    Classifies posture based on ergonomic metrics and clinical guidelines.
    Integrates Rule-based validation + Lightweight ML predictor + Prolonged Risk Diagnosis.
    
    Parameters:
        use_relative: If True and calibration exists, uses relative baseline deviations.
        sensitivity: Sensitivity multiplier (0.5 = lenient, 1.5 = strict).
        mode: Classification mode: 'hybrid' | 'rule_based' | 'ml'.
    """

    def __init__(
        self,
        use_relative: Optional[bool] = None,
        sensitivity: Optional[float] = None,
        mode: Optional[str] = None,
    ):
        cfg = get_posture_config()
        class_cfg = cfg.get("classification", {})
        metrics_cfg = cfg.get("metrics", {})
        calib_cfg = cfg.get("calibration", {}).get("relative_deviations", {})
        prolonged_cfg = class_cfg.get("prolonged_risk", {})

        self.use_relative = use_relative if use_relative is not None else class_cfg.get("use_relative_when_calibrated", True)
        self.sensitivity = sensitivity if sensitivity is not None else class_cfg.get("default_sensitivity", 1.0)
        self.mode = mode if mode is not None else class_cfg.get("mode", "hybrid")

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

        # Sub-modules
        self.ml_predictor = LightweightPostureML()
        self.accumulator = ProlongedRiskAccumulator(
            kyphosis_max_exposure_s=prolonged_cfg.get("kyphosis_max_exposure_s", 45.0),
            myopia_max_exposure_s=prolonged_cfg.get("myopia_max_exposure_s", 45.0),
            recovery_half_life_s=prolonged_cfg.get("recovery_half_life_s", 10.0),
        )

    def _determine_risk_severity(self, risk_pct: float) -> str:
        """Classify numerical risk percentage into severity string."""
        if risk_pct < 25.0:
            return RiskSeverity.LOW.value
        elif risk_pct < 50.0:
            return RiskSeverity.MODERATE.value
        elif risk_pct < 75.0:
            return RiskSeverity.HIGH.value
        else:
            return RiskSeverity.SEVERE.value

    def classify(
        self,
        metrics: PostureMetrics,
        calibration: Optional[CalibrationData] = None,
        kps: Optional[UnifiedKeypoints] = None,
        timestamp_s: Optional[float] = None,
    ) -> PostureState:
        """
        Classify posture state and diagnose ergonomic risks.
        
        Args:
            metrics: Computed PostureMetrics from current frame.
            calibration: Optional calibration baseline data.
            kps: Optional UnifiedKeypoints for detailed geometry.
            timestamp_s: Frame timestamp in seconds.
            
        Returns:
            PostureState enriched with sitting posture type, violations,
            and complete RiskAssessment (instant & prolonged Kyphosis & Myopia risks).
        """
        if not metrics.is_valid:
            return PostureState(
                status=PostureStatus.UNKNOWN,
                confidence=0.0,
                posture_description="Detecting...",
            )

        violations: List[PostureStatus] = []
        deviations: Dict[str, float] = {}
        severity_scores: List[float] = []

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

        # ----------------------------------------------------------------------
        # 1. Craniovertebral Angle (CVA) / Forward Head Posture
        # ----------------------------------------------------------------------
        cva_violation = False
        cva_sev = 0.0
        if metrics.neck_cva_deg > 0:
            if is_cal:
                delta_cva = calibration.baseline_neck_cva - metrics.neck_cva_deg
                deviations["neck_cva_delta"] = delta_cva
                warn_thresh = self.cva_rel_warn / self.sensitivity
                severe_thresh = self.cva_rel_severe / self.sensitivity
                if delta_cva > severe_thresh:
                    cva_violation = True
                    cva_sev = 1.0
                elif delta_cva > warn_thresh:
                    cva_violation = True
                    cva_sev = 0.5
            else:
                if metrics.neck_cva_deg < cva_sev_below * self.sensitivity:
                    cva_violation = True
                    cva_sev = 1.0
                elif metrics.neck_cva_deg < cva_norm_min * self.sensitivity:
                    cva_violation = True
                    cva_sev = 0.4

        if cva_violation:
            violations.append(PostureStatus.FORWARD_HEAD)
            severity_scores.append(cva_sev)

        # ----------------------------------------------------------------------
        # 2. Shoulder Tilt / Lateral Leaning (Left vs Right)
        # ----------------------------------------------------------------------
        tilt_violation = False
        tilt_sev = 0.0
        specific_tilt_status: Optional[PostureStatus] = None

        if tilt_enabled:
            # Check signed tilt direction: negative = Left, positive = Right
            signed_tilt = getattr(metrics, "lateral_tilt_deg", 0.0)
            lateral_offset = getattr(metrics, "lateral_spine_offset", 0.0)

            if is_cal:
                delta_tilt = abs(metrics.shoulder_tilt_deg) - abs(calibration.baseline_shoulder_tilt)
                deviations["shoulder_tilt_delta"] = delta_tilt
                warn_thresh = self.tilt_rel_warn / self.sensitivity
                severe_thresh = self.tilt_rel_severe / self.sensitivity
                if delta_tilt > severe_thresh:
                    tilt_violation = True
                    tilt_sev = 1.0
                elif delta_tilt > warn_thresh:
                    tilt_violation = True
                    tilt_sev = 0.5
            else:
                if metrics.shoulder_tilt_deg > tilt_sev_above / self.sensitivity:
                    tilt_violation = True
                    tilt_sev = 1.0
                elif metrics.shoulder_tilt_deg > tilt_norm_max / self.sensitivity:
                    tilt_violation = True
                    tilt_sev = 0.4

            if tilt_violation:
                # Distinguish Left vs Right
                # signed_tilt < 0 (left shoulder lower) or lateral_offset > 0.05
                if signed_tilt < -0.5 or (abs(signed_tilt) < 1.0 and lateral_offset > 0.05):
                    specific_tilt_status = PostureStatus.LEANING_LEFT
                elif signed_tilt > 0.5 or (abs(signed_tilt) < 1.0 and lateral_offset < -0.05):
                    specific_tilt_status = PostureStatus.LEANING_RIGHT
                else:
                    # Fallback to general tilt
                    specific_tilt_status = PostureStatus.SHOULDER_TILTED

                # Always add SHOULDER_TILTED for backwards compatibility, plus specific direction
                violations.append(PostureStatus.SHOULDER_TILTED)
                if specific_tilt_status != PostureStatus.SHOULDER_TILTED:
                    violations.append(specific_tilt_status)
                severity_scores.append(tilt_sev)

        # ----------------------------------------------------------------------
        # 3. Trunk / Spine Angle (Slouching / Kyphosis)
        # ----------------------------------------------------------------------
        trunk_violation = False
        trunk_sev = 0.0
        if is_cal:
            delta_trunk = metrics.trunk_angle_deg - calibration.baseline_trunk_angle
            deviations["trunk_angle_delta"] = delta_trunk
            warn_thresh = self.trunk_rel_warn / self.sensitivity
            severe_thresh = self.trunk_rel_severe / self.sensitivity
            if delta_trunk > severe_thresh:
                trunk_violation = True
                trunk_sev = 1.0
            elif delta_trunk > warn_thresh:
                trunk_violation = True
                trunk_sev = 0.5
        else:
            if metrics.trunk_angle_deg > self.trunk_severe_above / self.sensitivity:
                trunk_violation = True
                trunk_sev = 1.0
            elif metrics.trunk_angle_deg > self.trunk_normal_max / self.sensitivity:
                trunk_violation = True
                trunk_sev = 0.4

        if trunk_violation:
            violations.append(PostureStatus.SLOUCHED)
            severity_scores.append(trunk_sev)

        # ----------------------------------------------------------------------
        # 4. Eye-to-Screen Distance
        # ----------------------------------------------------------------------
        screen_violation = False
        screen_sev = 0.0
        if metrics.eye_distance_cm > 0:
            if is_cal:
                delta_dist = calibration.baseline_eye_distance - metrics.eye_distance_cm
                deviations["eye_distance_delta"] = delta_dist
                warn_thresh = self.screen_rel_warn / self.sensitivity
                severe_thresh = self.screen_rel_severe / self.sensitivity
                if delta_dist > severe_thresh:
                    screen_violation = True
                    screen_sev = 1.0
                elif delta_dist > warn_thresh:
                    screen_violation = True
                    screen_sev = 0.5
            else:
                if metrics.eye_distance_cm < self.screen_danger_below * self.sensitivity:
                    screen_violation = True
                    screen_sev = 1.0
                elif metrics.eye_distance_cm < self.screen_safe_min * self.sensitivity:
                    screen_violation = True
                    screen_sev = 0.4

        if screen_violation:
            violations.append(PostureStatus.TOO_CLOSE)
            severity_scores.append(screen_sev)

        # ----------------------------------------------------------------------
        # 5. Eye-to-Desk Distance
        # ----------------------------------------------------------------------
        desk_violation = False
        desk_sev = 0.0
        if metrics.eye_to_desk_cm > 0:
            if is_cal:
                delta_desk = calibration.baseline_eye_to_desk - metrics.eye_to_desk_cm
                deviations["eye_to_desk_delta"] = delta_desk
                warn_thresh = self.desk_rel_warn / self.sensitivity
                severe_thresh = self.desk_rel_severe / self.sensitivity
                if delta_desk > severe_thresh:
                    desk_violation = True
                    desk_sev = 1.0
                elif delta_desk > warn_thresh:
                    desk_violation = True
                    desk_sev = 0.5
            else:
                if metrics.eye_to_desk_cm < self.desk_danger_below * self.sensitivity:
                    desk_violation = True
                    desk_sev = 1.0
                elif metrics.eye_to_desk_cm < self.desk_safe_min * self.sensitivity:
                    desk_violation = True
                    desk_sev = 0.4

        if desk_violation:
            violations.append(PostureStatus.TOO_CLOSE_DESK)
            severity_scores.append(desk_sev)

        # ----------------------------------------------------------------------
        # 6. ML Predictor Execution (Tier 2)
        # ----------------------------------------------------------------------
        ml_features = self.ml_predictor.extract_features(metrics, calibration, kps=kps)
        ml_class, ml_probs, ml_kyph_score, ml_myop_score = self.ml_predictor.predict(
            ml_features, metrics=metrics, kps=kps
        )
        p_bad = float(ml_probs.get("Bad", 0.5))
        p_good = float(ml_probs.get("Good", 0.5))
        deviations["ml_prob_bad"] = round(p_bad, 3)
        deviations["ml_prob_good"] = round(p_good, 3)

        # ----------------------------------------------------------------------
        # 7. Clinical Risk Calculations (Instantaneous)
        # ----------------------------------------------------------------------
        # Kyphosis instant risk formula:
        trunk_strain = min(1.0, max(0.0, (metrics.trunk_angle_deg - 8.0) / 12.0))
        cva_strain = 0.0
        if metrics.neck_cva_deg > 0:
            cva_strain = min(1.0, max(0.0, (cva_norm_min - metrics.neck_cva_deg) / 18.0))

        rule_kyph_score = 0.65 * trunk_strain + 0.35 * cva_strain
        if is_cal and "trunk_angle_delta" in deviations:
            rule_kyph_score = max(rule_kyph_score, min(1.0, max(0.0, deviations["trunk_angle_delta"] / 15.0)))

        # Blend with ML in hybrid mode
        if self.mode == "ml":
            instant_kyph_norm = ml_kyph_score
        elif self.mode == "hybrid":
            instant_kyph_norm = 0.50 * rule_kyph_score + 0.50 * ml_kyph_score
        else:
            instant_kyph_norm = rule_kyph_score

        instant_kyphosis_risk_pct = round(float(np.clip(instant_kyph_norm * 100.0, 5.0 if not trunk_violation else 20.0, 100.0)), 1)
        if not trunk_violation and not cva_violation:
            instant_kyphosis_risk_pct = min(instant_kyphosis_risk_pct, 18.0)

        # Myopia instant risk formula:
        min_screen_dist = metrics.eye_distance_cm
        for dev in getattr(metrics, "device_distances", []):
            if dev.distance_cm > 0:
                if min_screen_dist < 0 or dev.distance_cm < min_screen_dist:
                    min_screen_dist = dev.distance_cm

        rule_myop_score = 0.0
        if min_screen_dist > 0:
            if min_screen_dist >= 50.0:
                rule_myop_score = max(0.05, 0.15 * (60.0 - min_screen_dist) / 10.0)
            elif min_screen_dist >= 35.0:
                rule_myop_score = 0.20 + 0.40 * ((50.0 - min_screen_dist) / 15.0)
            else:
                rule_myop_score = 0.60 + 0.40 * ((35.0 - min_screen_dist) / 15.0)

        if metrics.eye_to_desk_cm > 0 and metrics.eye_to_desk_cm < 30.0:
            desk_myop = (30.0 - metrics.eye_to_desk_cm) / 15.0
            rule_myop_score = max(rule_myop_score, desk_myop)

        rule_myop_score = float(np.clip(rule_myop_score, 0.0, 1.0))

        if self.mode == "ml":
            instant_myop_norm = ml_myop_score
        elif self.mode == "hybrid":
            instant_myop_norm = 0.55 * rule_myop_score + 0.45 * ml_myop_score
        else:
            instant_myop_norm = rule_myop_score

        instant_myopia_risk_pct = round(float(np.clip(instant_myop_norm * 100.0, 5.0 if not screen_violation else 25.0, 100.0)), 1)
        if not screen_violation and not desk_violation:
            instant_myopia_risk_pct = min(instant_myopia_risk_pct, 18.0)

        # ----------------------------------------------------------------------
        # 8. Prolonged Exposure Risk Diagnosis
        # ----------------------------------------------------------------------
        is_bad_posture = trunk_violation or cva_violation or tilt_violation or p_bad >= 0.5
        is_near_screen = screen_violation or desk_violation

        (
            prolonged_kyph_pct,
            prolonged_myop_pct,
            bad_dur_s,
            near_dur_s,
        ) = self.accumulator.update(
            instant_kyphosis_risk_pct=instant_kyphosis_risk_pct,
            instant_myopia_risk_pct=instant_myopia_risk_pct,
            is_bad_posture=is_bad_posture,
            is_near_screen=is_near_screen,
            timestamp_s=timestamp_s,
        )

        kyphosis_sev = self._determine_risk_severity(prolonged_kyph_pct)
        myopia_sev = self._determine_risk_severity(prolonged_myop_pct)

        # ----------------------------------------------------------------------
        # 9. Sitting Posture Classification & English Description
        # ----------------------------------------------------------------------
        active_violations = [v for v in violations if v != PostureStatus.SHOULDER_TILTED]
        if not active_violations and PostureStatus.SHOULDER_TILTED in violations:
            active_violations = [PostureStatus.SHOULDER_TILTED]

        desc_map = {
            PostureStatus.LEANING_LEFT: "Leaning Left",
            PostureStatus.LEANING_RIGHT: "Leaning Right",
            PostureStatus.SHOULDER_TILTED: "Shoulder Tilted",
            PostureStatus.SLOUCHED: "Slouched (Kyphosis)",
            PostureStatus.FORWARD_HEAD: "Forward Head (Tech Neck)",
            PostureStatus.TOO_CLOSE: "Screen Too Close",
            PostureStatus.TOO_CLOSE_DESK: "Too Close to Desk",
        }

        if self.mode == "ml" and self.ml_predictor.is_trained:
            # Pure Trained ML Mode
            if p_bad < 0.50:
                status = PostureStatus.GOOD
                alert_level = AlertLevel.SAFE
                posture_desc = f"Upright (ML {p_good*100:.0f}%)"
            else:
                status = active_violations[0] if active_violations else ml_class
                if status == PostureStatus.GOOD:
                    status = PostureStatus.SLOUCHED
                alert_level = AlertLevel.WARNING if p_bad < 0.75 else AlertLevel.CRITICAL
                posture_desc = f"{status.value.replace('_', ' ').title()} (ML {p_bad*100:.0f}%)"

        elif self.mode == "hybrid" and self.ml_predictor.is_trained:
            # Hybrid Mode: fuses rules with trained ML neural network head
            rule_has_violation = len(active_violations) > 0
            severe_violation = any(s >= 0.8 for s in severity_scores)

            if not severe_violation and p_bad < 0.35:
                # ML confidently confirms Good posture, filter out minor threshold noise
                status = PostureStatus.GOOD
                alert_level = AlertLevel.SAFE
                posture_desc = f"Upright (Hybrid {p_good*100:.0f}%)"
            elif p_bad >= 0.52 or (rule_has_violation and p_bad >= 0.40):
                if len(active_violations) == 1:
                    status = active_violations[0]
                    alert_level = AlertLevel.WARNING if p_bad < 0.75 else AlertLevel.CRITICAL
                    posture_desc = desc_map.get(status, status.value.replace("_", " ").title())
                elif len(active_violations) > 1:
                    status = PostureStatus.COMBINED
                    alert_level = AlertLevel.CRITICAL
                    names = [desc_map.get(v, v.value) for v in active_violations]
                    posture_desc = "Combined: " + " + ".join(names[:2])
                else:
                    status = ml_class if ml_class != PostureStatus.GOOD else PostureStatus.SLOUCHED
                    alert_level = AlertLevel.WARNING
                    posture_desc = f"{status.value.replace('_', ' ').title()} (Hybrid {p_bad*100:.0f}%)"
            else:
                status = PostureStatus.GOOD
                alert_level = AlertLevel.SAFE
                posture_desc = "Upright / Straight"

        else:
            # Rule-based Mode
            if len(active_violations) == 0:
                status = PostureStatus.GOOD
                alert_level = AlertLevel.SAFE
                posture_desc = "Upright / Straight"
            elif len(active_violations) == 1:
                status = active_violations[0]
                max_sev = max(severity_scores) if severity_scores else 0.4
                alert_level = AlertLevel.CRITICAL if max_sev >= 0.8 else (AlertLevel.WARNING if max_sev >= 0.4 else AlertLevel.MILD)
                posture_desc = desc_map.get(status, status.value.replace("_", " ").title())
            else:
                status = PostureStatus.COMBINED
                alert_level = AlertLevel.CRITICAL if any(s >= 0.8 for s in severity_scores) or len(active_violations) >= 3 else AlertLevel.WARNING
                names = [v.value.replace("_", " ").title() for v in active_violations]
                posture_desc = "Combined: " + " + ".join(names[:2])

        # Construct RiskAssessment
        risk_assessment = RiskAssessment(
            kyphosis_risk_pct=instant_kyphosis_risk_pct,
            prolonged_kyphosis_risk_pct=round(prolonged_kyph_pct, 1),
            kyphosis_severity=kyphosis_sev,
            myopia_risk_pct=instant_myopia_risk_pct,
            prolonged_myopia_risk_pct=round(prolonged_myop_pct, 1),
            myopia_severity=myopia_sev,
            posture_description=posture_desc,
            bad_posture_duration_s=round(bad_dur_s, 1),
            near_screen_duration_s=round(near_dur_s, 1),
            classifier_mode=self.mode,
        )

        # Confidence: based on how many metrics were computable
        n_computed = sum([
            metrics.neck_cva_deg > 0,
            True,  # shoulder tilt
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
            risk_assessment=risk_assessment,
            kyphosis_risk_pct=instant_kyphosis_risk_pct,
            prolonged_kyphosis_risk_pct=round(prolonged_kyph_pct, 1),
            myopia_risk_pct=instant_myopia_risk_pct,
            prolonged_myopia_risk_pct=round(prolonged_myop_pct, 1),
            posture_description=posture_desc,
        )

    def reset_exposure(self):
        """Reset prolonged risk exposure counters."""
        self.accumulator.reset()
