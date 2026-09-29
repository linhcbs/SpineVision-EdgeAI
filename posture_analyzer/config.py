"""
Configuration Manager for Posture Analyzer.
Loads and caches configuration from configs/posture_analyzer_config.json.
"""

import os
import json
from typing import Dict, Any, Optional

_CACHED_CONFIG: Optional[Dict[str, Any]] = None


def find_config_file(custom_path: Optional[str] = None) -> str:
    """Find the path to posture_analyzer_config.json."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Search candidates
    here = os.path.dirname(os.path.abspath(__file__))  # code/posture_analyzer
    code_root = os.path.dirname(here)                 # code

    candidates = [
        os.path.join(code_root, "configs", "posture_analyzer_config.json"),
        os.path.join(code_root, "benchmarks", "configs", "posture_analyzer_config.json"),
        os.path.join(os.getcwd(), "configs", "posture_analyzer_config.json"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    # Fallback to default path in code_root/configs
    return candidates[0]


def get_default_config() -> Dict[str, Any]:
    """Default fallback configuration dictionary."""
    return {
        "camera": {
            "camera_index": 0,
            "fallback_indices": [1, 2, 3, 4],
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
        "smoothing": {
            "enabled": True,
            "filter_type": "one_euro",
            "one_euro": {"min_cutoff": 1.0, "beta": 0.005, "d_cutoff": 1.0},
            "ema": {"alpha": 0.4},
        },
        "keypoints": {
            "min_confidence": 0.3,
        },
        "metrics": {
            "neck_cva": {
                "name": "Craniovertebral Angle",
                "abbreviation": "CVA",
                "unit": "deg",
                "normal_min": 50.0,
                "mild_min": 40.0,
                "severe_below": 40.0,
            },
            "shoulder_tilt": {
                "name": "Shoulder Level Angle",
                "abbreviation": "Shoulder Level",
                "unit": "deg",
                "display_scale": "180_to_90",
                "normal_min": 175.0,
                "warning_min": 172.0,
                "severe_below": 172.0,
                "raw_tilt_thresholds": {
                    "normal_max": 5.0,
                    "warning_max": 8.0,
                    "severe_above": 8.0,
                },
            },
            "trunk_angle": {
                "name": "Trunk Slump Angle",
                "abbreviation": "Trunk Angle",
                "unit": "deg",
                "normal_max": 10.0,
                "warning_max": 18.0,
                "severe_above": 18.0,
            },
            "eye_to_screen_distance": {
                "name": "Eye to Screen Distance",
                "abbreviation": "Eye - Screen",
                "unit": "cm",
                "safe_min": 35.0,
                "warning_min": 25.0,
                "danger_below": 25.0,
                "average_ipd_cm": 6.3,
                "default_focal_length_px": 650.0,
                "z_depth_dampening": 0.3,
            },
            "eye_to_desk_distance": {
                "name": "Eye to Desk Distance",
                "abbreviation": "Eye - Desk",
                "unit": "cm",
                "safe_min": 35.0,
                "warning_min": 25.0,
                "danger_below": 25.0,
            },
        },
        "view_modes": {
            "enabled": True,
            "hysteresis_deg": 5.0,
            "yaw_smoothing_alpha": 0.25,
            "frontal": {
                "max_yaw_deg": 25.0,
                "neck_cva": {
                    "normal_min": 80.0,
                    "mild_min": 70.0,
                    "severe_below": 70.0,
                },
                "shoulder_tilt": {
                    "enabled": True,
                    "normal_min": 175.0,
                    "warning_min": 172.0,
                    "severe_below": 172.0,
                    "raw_tilt_thresholds": {
                        "normal_max": 5.0,
                        "warning_max": 8.0,
                        "severe_above": 8.0,
                    },
                },
            },
            "oblique": {
                "max_yaw_deg": 65.0,
                "neck_cva": {
                    "normal_min": 65.0,
                    "mild_min": 55.0,
                    "severe_below": 50.0,
                },
                "shoulder_tilt": {
                    "enabled": True,
                    "normal_min": 165.0,
                    "warning_min": 158.0,
                    "severe_below": 158.0,
                    "raw_tilt_thresholds": {
                        "normal_max": 15.0,
                        "warning_max": 22.0,
                        "severe_above": 22.0,
                    },
                },
            },
            "profile": {
                "neck_cva": {
                    "normal_min": 52.0,
                    "mild_min": 45.0,
                    "severe_below": 40.0,
                },
                "shoulder_tilt": {
                    "enabled": True,
                    "normal_min": 175.0,
                    "warning_min": 172.0,
                    "severe_below": 172.0,
                    "raw_tilt_thresholds": {
                        "normal_max": 5.0,
                        "warning_max": 8.0,
                        "severe_above": 8.0,
                    },
                },
            },
        },
        "calibration": {
            "target_samples": 90,
            "calibration_distance_cm": 60.0,
            "relative_deviations": {
                "neck_cva_warn_deg": 8.0,
                "neck_cva_severe_deg": 15.0,
                "shoulder_tilt_warn_deg": 4.0,
                "shoulder_tilt_severe_deg": 8.0,
                "trunk_angle_warn_deg": 8.0,
                "trunk_angle_severe_deg": 15.0,
                "screen_distance_warn_cm": 15.0,
                "screen_distance_severe_cm": 25.0,
                "desk_distance_warn_cm": 10.0,
                "desk_distance_severe_cm": 18.0,
            },
        },
        "classification": {
            "use_relative_when_calibrated": True,
            "default_sensitivity": 1.0,
            "severity_levels": {
                "mild_threshold": 0.4,
                "critical_threshold": 0.8,
            },
        },
        "workspace_detection": {
            "enabled": True,
            "desk_smoothing": 0.15,
            "min_confidence": 0.3,
            "desk_offset_ratio": 0.05,
            "screen_width_ratio": 0.65,
            "screen_aspect_ratio": 0.5625,
            "opacity": 0.3,
        },
        "screen_detection": {
            "enabled": True,
            "model_path": "models/weights/yolo11n.pt",
            "max_devices": 3,
            "min_confidence": 0.40,
            "iou_threshold": 0.50,
            "draw_boxes": True,
            "box_opacity": 0.18,
        },
        "visualization_hud": {
            "language": "en",
            "show_ram": True,
            "show_fps_latency": True,
            "panel_width": 310,
            "panel_opacity": 0.75,
            "status_labels": {
                "GOOD": "Good Posture",
                "FORWARD_HEAD": "Forward Head",
                "SLOUCHED": "Slouched Spine",
                "SHOULDER_TILTED": "Shoulder Tilted",
                "TOO_CLOSE": "Too Close to Screen",
                "TOO_CLOSE_DESK": "Too Close to Desk",
                "COMBINED": "Multiple Violations",
                "UNKNOWN": "Detecting...",
            },
        },
    }


def load_posture_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON, with deep fallback to defaults."""
    global _CACHED_CONFIG
    resolved_path = find_config_file(config_path)

    default_cfg = get_default_config()

    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                loaded_cfg = json.load(f)
            # Merge loaded with defaults so new keys are always present
            _deep_merge(default_cfg, loaded_cfg)
        except Exception as e:
            print(f"[Warning] Failed to parse config at {resolved_path}: {e}. Using defaults.")

    _CACHED_CONFIG = default_cfg
    return _CACHED_CONFIG


def get_posture_config() -> Dict[str, Any]:
    """Get the cached posture analyzer configuration."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        _CACHED_CONFIG = load_posture_config()
    return _CACHED_CONFIG


def _deep_merge(base: dict, update: dict):
    """Recursively update base dictionary with update dictionary."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
