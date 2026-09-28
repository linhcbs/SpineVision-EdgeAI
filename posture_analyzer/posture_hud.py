"""
Posture Analysis HUD (Heads-Up Display) Renderer.

Draws posture metrics, status indicators, angle visualizations, and
workspace overlays directly onto the webcam video frame.

Design:
  - Right side panel: Ergonomic posture metrics with color-coded status bars
  - Angle arc visualizations drawn on the skeleton
  - Calibration progress indicator
  - Status badge with overall posture classification
  - Real-time RAM consumption and latency indicators
  - All text in English
"""

import cv2
import math
import numpy as np
from typing import Optional, Tuple, List

try:
    import psutil
    _CURRENT_PROCESS = psutil.Process()
except ImportError:
    _CURRENT_PROCESS = None

from .types import (
    PostureState,
    PostureStatus,
    PostureMetrics,
    AlertLevel,
    ViewMode,
    UnifiedKeypoints,
    NormalizedKeypoint,
    DetectedDevice,
    DeviceType,
)
from .config import get_posture_config


# ==============================================================================
# COLOR PALETTE (BGR)
# ==============================================================================
COLOR_GOOD = (0, 200, 80)          # Green
COLOR_MILD = (0, 200, 255)         # Yellow/Orange
COLOR_WARNING = (0, 140, 255)      # Orange
COLOR_CRITICAL = (0, 50, 255)      # Red
COLOR_UNKNOWN = (150, 150, 150)    # Gray
COLOR_TEXT_WHITE = (240, 240, 240)
COLOR_TEXT_DIM = (160, 160, 160)
COLOR_PANEL_BG = (22, 22, 22)
COLOR_CALIBRATING = (255, 200, 0)  # Cyan-gold
COLOR_ARC = (255, 255, 0)          # Cyan for angle arcs
COLOR_DISTANCE_LINE = (255, 150, 50)  # Blue-ish for distance indicator


def _get_ram_usage_mb() -> float:
    """Get current process resident memory in Megabytes."""
    if _CURRENT_PROCESS is not None:
        try:
            return _CURRENT_PROCESS.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


def _status_color(status: PostureStatus) -> Tuple[int, int, int]:
    """Get the display color for a posture status."""
    return {
        PostureStatus.GOOD: COLOR_GOOD,
        PostureStatus.FORWARD_HEAD: COLOR_CRITICAL,
        PostureStatus.SLOUCHED: COLOR_WARNING,
        PostureStatus.SHOULDER_TILTED: COLOR_MILD,
        PostureStatus.LEANING_LEFT: COLOR_WARNING,
        PostureStatus.LEANING_RIGHT: COLOR_WARNING,
        PostureStatus.TOO_CLOSE: COLOR_CRITICAL,
        PostureStatus.TOO_CLOSE_DESK: COLOR_CRITICAL,
        PostureStatus.COMBINED: COLOR_CRITICAL,
        PostureStatus.UNKNOWN: COLOR_UNKNOWN,
    }.get(status, COLOR_UNKNOWN)


def _alert_color(level: AlertLevel) -> Tuple[int, int, int]:
    """Get the display color for an alert level."""
    return {
        AlertLevel.SAFE: COLOR_GOOD,
        AlertLevel.MILD: COLOR_MILD,
        AlertLevel.WARNING: COLOR_WARNING,
        AlertLevel.CRITICAL: COLOR_CRITICAL,
    }.get(level, COLOR_UNKNOWN)


def _risk_color(risk_pct: float) -> Tuple[int, int, int]:
    """Get display color for clinical risk percentage: Green -> Yellow -> Orange -> Red."""
    if risk_pct < 25.0:
        return COLOR_GOOD
    elif risk_pct < 50.0:
        return COLOR_MILD
    elif risk_pct < 75.0:
        return COLOR_WARNING
    else:
        return COLOR_CRITICAL


def _status_label(status: PostureStatus, desc: str = "") -> str:
    """Get English label for posture status."""
    if desc:
        return desc
    labels = get_posture_config().get("visualization_hud", {}).get("status_labels", {})
    return labels.get(status.name, status.name.replace("_", " ").title())


def draw_posture_hud(
    frame: np.ndarray,
    state: PostureState,
    kps: Optional[UnifiedKeypoints] = None,
    is_calibrating: bool = False,
    calib_progress: float = 0.0,
    panel_width: int = 330,
    detected_devices: Optional[List[DetectedDevice]] = None,
):
    """
    Draw the ergonomic posture analysis HUD on the right side of the frame in English.
    
    Args:
        frame: BGR image (modified in-place).
        state: Current PostureState with metrics, classification, and risk assessment.
        kps: Optional UnifiedKeypoints for drawing angle arcs on skeleton.
        is_calibrating: Whether calibration is in progress.
        calib_progress: Calibration progress (0.0 - 1.0).
        panel_width: Width of the right-side metrics panel.
        detected_devices: Optional list of DetectedDevice for multi-screen HUD section.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_small = cv2.FONT_HERSHEY_SIMPLEX

    panel_x = w - panel_width - 10
    panel_y = 8

    # Resolve device list from arg or state
    if detected_devices is None:
        detected_devices = getattr(state, 'detected_devices', [])

    # Dynamic panel height based on device count and risk section
    n_dev = len(detected_devices)
    extra_dev_rows = max(0, n_dev)
    has_risk = getattr(state, "risk_assessment", None) is not None
    base_panel_h = 490 if has_risk else 385
    panel_height = base_panel_h + extra_dev_rows * 20 + (20 if n_dev > 0 else 0)
    if is_calibrating:
        panel_height += 30

    # === Semi-transparent background panel ===
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w - 5, panel_y + panel_height), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    y_cursor = panel_y + 22

    # === Top Header: Status Badge & RAM ===
    ram_mb = _get_ram_usage_mb()

    if is_calibrating:
        status_text = f"CALIBRATING... {calib_progress * 100:.0f}%"
        status_color = COLOR_CALIBRATING
        # Draw progress bar
        bar_x1 = panel_x + 10
        bar_x2 = w - 15
        bar_y = y_cursor + 8
        bar_h = 10
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 50), -1)
        fill_x = int(bar_x1 + (bar_x2 - bar_x1) * calib_progress)
        cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + bar_h), COLOR_CALIBRATING, -1)
        cv2.putText(frame, status_text, (panel_x + 10, y_cursor),
                    font, 0.48, status_color, 1, cv2.LINE_AA)
        if ram_mb > 0:
            cv2.putText(frame, f"RAM: {ram_mb:.0f}MB", (w - 85, y_cursor),
                        font_small, 0.35, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y_cursor += 30
    else:
        status_color = _status_color(state.status)
        desc = getattr(state, "posture_description", "")
        status_text = _status_label(state.status, desc=desc)

        # Status dot & label
        cv2.circle(frame, (panel_x + 16, y_cursor - 5), 6, status_color, -1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (panel_x + 28, y_cursor),
                    font, 0.46, status_color, 1, cv2.LINE_AA)
        if ram_mb > 0:
            cv2.putText(frame, f"RAM: {ram_mb:.0f}MB", (w - 85, y_cursor),
                        font_small, 0.35, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y_cursor += 16

        # View mode indicator
        view_mode = getattr(state.metrics, "view_mode", ViewMode.FRONTAL)
        yaw_val = getattr(state.metrics, "yaw_deg", 0.0)
        view_str = f"View: {view_mode.value.title()} ({yaw_val:.0f} deg)"
        cv2.putText(frame, view_str, (panel_x + 28, y_cursor),
                    font_small, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
        y_cursor += 4

    # === Divider ===
    y_cursor += 10
    cv2.line(frame, (panel_x + 8, y_cursor), (w - 15, y_cursor), (60, 60, 60), 1)
    y_cursor += 14

    # === Ergonomic Clinical Risk Diagnostics Section ===
    risk = getattr(state, "risk_assessment", None)
    if risk is not None:
        cv2.putText(frame, "CLINICAL RISK ASSESSMENT", (panel_x + 10, y_cursor),
                    font_small, 0.38, (120, 210, 255), 1, cv2.LINE_AA)
        y_cursor += 16

        def draw_risk_row(label: str, risk_pct: float, severity: str, dur_s: Optional[float] = None):
            nonlocal y_cursor
            color = _risk_color(risk_pct)

            # Label on left
            cv2.putText(frame, label, (panel_x + 10, y_cursor),
                        font_small, 0.36, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

            # Value & Severity on right (e.g. "45% [Moderate] (12s)")
            dur_str = f" ({dur_s:.0f}s)" if dur_s is not None and dur_s >= 2.0 else ""
            val_text = f"{risk_pct:.0f}% [{severity}]{dur_str}"
            cv2.putText(frame, val_text, (panel_x + 140, y_cursor),
                        font_small, 0.38, color, 1, cv2.LINE_AA)

            # Progress bar
            bar_x1 = panel_x + 10
            bar_x2 = w - 15
            bar_y = y_cursor + 4
            bar_h = 3
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (45, 45, 45), -1)
            fill_ratio = min(1.0, max(0.0, risk_pct / 100.0))
            fill_x = int(bar_x1 + (bar_x2 - bar_x1) * fill_ratio)
            cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + bar_h), color, -1)
            y_cursor += 18

        draw_risk_row("Kyphosis Risk", risk.kyphosis_risk_pct, risk.kyphosis_severity)
        draw_risk_row("Prolonged Kyph", risk.prolonged_kyphosis_risk_pct, risk.kyphosis_severity, risk.bad_posture_duration_s)
        draw_risk_row("Myopia Risk", risk.myopia_risk_pct, risk.myopia_severity)
        draw_risk_row("Prolonged Myop", risk.prolonged_myopia_risk_pct, risk.myopia_severity, risk.near_screen_duration_s)

        # Divider
        y_cursor += 4
        cv2.line(frame, (panel_x + 8, y_cursor), (w - 15, y_cursor), (60, 60, 60), 1)
        y_cursor += 14

    # === Metrics Section ===
    metrics = state.metrics
    cfg = get_posture_config()
    m_cfg = cfg.get("metrics", {})
    vm_cfg = cfg.get("view_modes", {})

    view_mode = getattr(metrics, "view_mode", ViewMode.FRONTAL)
    mode_key = view_mode.value.lower() if isinstance(view_mode, ViewMode) else str(view_mode).lower()
    cur_vm_cfg = vm_cfg.get(mode_key, {})

    cva_cfg = m_cfg.get("neck_cva", {})
    sh_cfg = m_cfg.get("shoulder_tilt", {})
    tr_cfg = m_cfg.get("trunk_angle", {})
    screen_cfg = m_cfg.get("eye_to_screen_distance", {})
    desk_cfg = m_cfg.get("eye_to_desk_distance", {})

    # Adaptive CVA thresholds
    cva_mode_cfg = cur_vm_cfg.get("neck_cva", {})
    cva_good = cva_mode_cfg.get("normal_min", cva_cfg.get("normal_min", 50.0))
    cva_warn = cva_mode_cfg.get("mild_min", cva_cfg.get("mild_min", 40.0))

    # Adaptive Shoulder Tilt thresholds
    tilt_mode_cfg = cur_vm_cfg.get("shoulder_tilt", {})
    sh_enabled = tilt_mode_cfg.get("enabled", True)
    sh_good = tilt_mode_cfg.get("normal_min", sh_cfg.get("normal_min", 175.0))
    sh_warn = tilt_mode_cfg.get("warning_min", sh_cfg.get("warning_min", 172.0))

    tr_good = tr_cfg.get("normal_max", 10.0)
    tr_warn = tr_cfg.get("warning_max", 18.0)

    scr_safe = screen_cfg.get("safe_min", 35.0)
    scr_warn = screen_cfg.get("warning_min", 25.0)

    desk_safe = desk_cfg.get("safe_min", 35.0)
    desk_warn = desk_cfg.get("warning_min", 25.0)

    # Helper to draw a metric row with status bar
    def draw_metric_row(label: str, value: float, unit: str, good_range: Tuple[float, float],
                        warn_range: Tuple[float, float], invert: bool = False):
        nonlocal y_cursor

        # Determine color based on value
        if invert:
            # Higher value = better (CVA, Shoulder Level [180-90], Distances)
            if value >= good_range[0]:
                bar_color = COLOR_GOOD
            elif value >= warn_range[0]:
                bar_color = COLOR_WARNING
            else:
                bar_color = COLOR_CRITICAL
        else:
            # Lower value = better (Trunk slump, Raw Tilt)
            if value <= good_range[1]:
                bar_color = COLOR_GOOD
            elif value <= warn_range[1]:
                bar_color = COLOR_WARNING
            else:
                bar_color = COLOR_CRITICAL

        # Label
        cv2.putText(frame, label, (panel_x + 10, y_cursor),
                    font_small, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Value with unit
        value_text = f"{value:.1f} {unit}"
        cv2.putText(frame, value_text, (panel_x + 160, y_cursor),
                    font_small, 0.46, bar_color, 1, cv2.LINE_AA)

        # Status bar
        bar_x1 = panel_x + 10
        bar_x2 = w - 15
        bar_y = y_cursor + 5
        bar_h = 4
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 50), -1)

        # Bar fill proportional to how "good" the value is
        if invert:
            span = max(good_range[1] - warn_range[0], 1.0)
            fill_ratio = min(1.0, max(0.0, (value - warn_range[0] * 0.7) / span))
        else:
            fill_ratio = min(1.0, max(0.0, 1.0 - value / max(warn_range[1] * 1.5, 1.0)))

        fill_x = int(bar_x1 + (bar_x2 - bar_x1) * fill_ratio)
        cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + bar_h), bar_color, -1)

        y_cursor += 28

    # 1. Neck CVA
    if metrics.neck_cva_deg > 0:
        draw_metric_row(
            "Neck CVA", metrics.neck_cva_deg, "deg",
            good_range=(cva_good, 90.0), warn_range=(cva_warn, cva_good), invert=True,
        )
    else:
        cv2.putText(frame, "Neck CVA: N/A", (panel_x + 10, y_cursor),
                    font_small, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y_cursor += 28

    # 2. Shoulder Level Angle (Scaled 180° - 90°)
    if not sh_enabled:
        cv2.putText(frame, "Shoulder Level", (panel_x + 10, y_cursor),
                    font_small, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, "N/A (Disabled)", (panel_x + 150, y_cursor),
                    font_small, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        bar_x1 = panel_x + 10
        bar_x2 = w - 15
        bar_y = y_cursor + 5
        bar_h = 4
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 50), -1)
        y_cursor += 28
    else:
        draw_metric_row(
            "Shoulder Level", metrics.shoulder_level_deg, "deg",
            good_range=(sh_good, 180.0), warn_range=(sh_warn, sh_good), invert=True,
        )

    # 3. Trunk Angle
    draw_metric_row(
        "Trunk Slump", metrics.trunk_angle_deg, "deg",
        good_range=(0.0, tr_good), warn_range=(0.0, tr_warn), invert=False,
    )

    # 4. Eye to Screen Distance (primary / webcam)
    if metrics.eye_distance_cm > 0:
        draw_metric_row(
            "Eye - Screen", metrics.eye_distance_cm, "cm",
            good_range=(scr_safe, 200.0), warn_range=(scr_warn, scr_safe), invert=True,
        )
    else:
        cv2.putText(frame, "Eye - Screen: N/A", (panel_x + 10, y_cursor),
                    font_small, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y_cursor += 28

    # 4b. Multi-device distances section
    if detected_devices:
        # Try to import color helper; fall back to gray
        try:
            from .screen_detector import get_device_color
        except Exception:
            def get_device_color(_t):
                return (160, 160, 160)

        # Section header
        cv2.putText(frame, "Devices Detected:", (panel_x + 10, y_cursor),
                    font_small, 0.37, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y_cursor += 18

        for dev in detected_devices:
            color = get_device_color(dev.device_type)
            dist_str = f"{dev.distance_cm:.0f} cm" if dev.distance_cm > 0 else "N/A"
            # Status color based on distance
            if dev.distance_cm > 0:
                if dev.distance_cm >= scr_safe:
                    dist_color = COLOR_GOOD
                elif dev.distance_cm >= scr_warn:
                    dist_color = COLOR_WARNING
                else:
                    dist_color = COLOR_CRITICAL
            else:
                dist_color = COLOR_TEXT_DIM

            # Colored dot indicating device type
            cv2.circle(frame, (panel_x + 16, y_cursor - 4), 4, color, -1, cv2.LINE_AA)
            cv2.putText(frame, dev.label, (panel_x + 26, y_cursor),
                        font_small, 0.37, color, 1, cv2.LINE_AA)
            cv2.putText(frame, dist_str, (panel_x + 160, y_cursor),
                        font_small, 0.40, dist_color, 1, cv2.LINE_AA)
            y_cursor += 18

    # 5. Eye to Desk Distance
    if metrics.eye_to_desk_cm > 0:
        draw_metric_row(
            "Eye - Desk", metrics.eye_to_desk_cm, "cm",
            good_range=(desk_safe, 150.0), warn_range=(desk_warn, desk_safe), invert=True,
        )
    else:
        cv2.putText(frame, "Eye - Desk: N/A", (panel_x + 10, y_cursor),
                    font_small, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y_cursor += 28

    # === Divider ===
    cv2.line(frame, (panel_x + 8, y_cursor), (w - 15, y_cursor), (60, 60, 60), 1)
    y_cursor += 12

    # === Calibration status ===
    cal_text = "Calibrated (Personalized)" if state.is_calibrated else "Uncalibrated (Press 'C')"
    cal_color = COLOR_GOOD if state.is_calibrated else COLOR_TEXT_DIM
    cv2.putText(frame, cal_text, (panel_x + 10, y_cursor),
                font_small, 0.38, cal_color, 1, cv2.LINE_AA)
    y_cursor += 18

    # === Alert Level ===
    alert_color = _alert_color(state.alert_level)
    alert_text = f"Alert: {state.alert_level.name}"
    cv2.putText(frame, alert_text, (panel_x + 10, y_cursor),
                font_small, 0.38, alert_color, 1, cv2.LINE_AA)

    # === Draw angle arcs on skeleton (if keypoints available) ===
    if kps is not None and metrics.is_valid:
        _draw_angle_arcs(frame, kps, metrics, desk_y=state.desk_line_y)


def _draw_angle_arcs(
    frame: np.ndarray,
    kps: UnifiedKeypoints,
    metrics: PostureMetrics,
    desk_y: int = -1,
):
    """Draw visual angle arc indicators and measurements on the skeleton."""
    cfg = get_posture_config()
    m_cfg = cfg.get("metrics", {})
    vm_cfg = cfg.get("view_modes", {})

    view_mode = getattr(metrics, "view_mode", ViewMode.FRONTAL)
    mode_key = view_mode.value.lower() if isinstance(view_mode, ViewMode) else str(view_mode).lower()
    cur_vm_cfg = vm_cfg.get(mode_key, {})

    cva_mode_cfg = cur_vm_cfg.get("neck_cva", {})
    cva_good = cva_mode_cfg.get("normal_min", m_cfg.get("neck_cva", {}).get("normal_min", 50.0))

    tilt_mode_cfg = cur_vm_cfg.get("shoulder_tilt", {})
    sh_enabled = tilt_mode_cfg.get("enabled", True)
    sh_good = tilt_mode_cfg.get("normal_min", m_cfg.get("shoulder_tilt", {}).get("normal_min", 175.0))

    tr_good = m_cfg.get("trunk_angle", {}).get("normal_max", 10.0)
    scr_safe = m_cfg.get("eye_to_screen_distance", {}).get("safe_min", 35.0)
    scr_warn = m_cfg.get("eye_to_screen_distance", {}).get("warning_min", 25.0)
    desk_safe = m_cfg.get("eye_to_desk_distance", {}).get("safe_min", 35.0)
    desk_warn = m_cfg.get("eye_to_desk_distance", {}).get("warning_min", 25.0)

    # --- CVA Arc: from mid-shoulder to ear ---
    mid_shoulder = kps.get_midpoint("left_shoulder", "right_shoulder")
    l_ear = kps.get("left_ear")
    r_ear = kps.get("right_ear")

    if mid_shoulder is not None and (l_ear is not None or r_ear is not None):
        if l_ear is not None and r_ear is not None:
            ear = NormalizedKeypoint(
                x=(l_ear.x + r_ear.x) / 2,
                y=(l_ear.y + r_ear.y) / 2,
            )
        elif l_ear is not None:
            ear = l_ear
        else:
            ear = r_ear

        sx, sy = int(mid_shoulder.x), int(mid_shoulder.y)
        ex, ey = int(ear.x), int(ear.y)

        # Draw line from shoulder to ear
        color = COLOR_GOOD if metrics.neck_cva_deg >= cva_good else COLOR_CRITICAL
        cv2.line(frame, (sx, sy), (ex, ey), color, 2, cv2.LINE_AA)

        # Draw horizontal reference line from shoulder
        cv2.line(frame, (sx - 40, sy), (sx + 40, sy), (100, 100, 100), 1, cv2.LINE_AA)

        # Draw small arc
        radius = 30
        angle_start = 0  # horizontal
        angle_end = -int(metrics.neck_cva_deg)
        cv2.ellipse(frame, (sx, sy), (radius, radius), 0, angle_end, angle_start,
                    color, 1, cv2.LINE_AA)

        # Angle label
        label_x = sx + 35
        label_y = sy - 10
        cv2.putText(frame, f"CVA: {metrics.neck_cva_deg:.0f} deg",
                    (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # --- Shoulder Tilt Line (Scaled 180° - 90°) ---
    # Only draw when shoulder tilt is enabled
    if sh_enabled:
        l_shoulder = kps.get("left_shoulder")
        r_shoulder = kps.get("right_shoulder")
        if l_shoulder is not None and r_shoulder is not None:
            color = COLOR_GOOD if metrics.shoulder_level_deg >= sh_good else COLOR_WARNING
            lx, ly = int(l_shoulder.x), int(l_shoulder.y)
            rx, ry = int(r_shoulder.x), int(r_shoulder.y)

            # Extend shoulder line slightly
            cv2.line(frame, (lx - 10, ly), (rx + 10, ry), color, 2, cv2.LINE_AA)

            # Draw horizontal reference
            mid_x = (lx + rx) // 2
            mid_y = (ly + ry) // 2
            cv2.line(frame, (mid_x - 50, mid_y), (mid_x + 50, mid_y),
                     (100, 100, 100), 1, cv2.LINE_AA)

            # Shoulder Level label (scaled 180° - 90°)
            cv2.putText(frame, f"Shoulder: {metrics.shoulder_level_deg:.1f} deg",
                        (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        color, 1, cv2.LINE_AA)

    # --- Trunk Angle: spine line from mid-shoulder to mid-hip ---
    mid_hip = kps.get_midpoint("left_hip", "right_hip")
    if mid_shoulder is not None and mid_hip is not None:
        sx, sy = int(mid_shoulder.x), int(mid_shoulder.y)
        hx, hy = int(mid_hip.x), int(mid_hip.y)

        color = COLOR_GOOD if metrics.trunk_angle_deg <= tr_good else COLOR_WARNING
        cv2.line(frame, (sx, sy), (hx, hy), color, 2, cv2.LINE_AA)

        # Vertical reference from hip
        cv2.line(frame, (hx, hy), (hx, hy - 80), (100, 100, 100), 1, cv2.LINE_AA)

        # Arc at hip
        radius = 25
        cv2.ellipse(frame, (hx, hy), (radius, radius), -90,
                    0, int(metrics.trunk_angle_deg), color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"Trunk: {metrics.trunk_angle_deg:.1f} deg",
                    (hx + 30, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    color, 1, cv2.LINE_AA)

    # --- Eye-to-Screen Distance Indicator ---
    nose = kps.get("nose")
    if metrics.eye_distance_cm > 0 and nose is not None:
        nx, ny = int(nose.x), int(nose.y)
        color = COLOR_GOOD if metrics.eye_distance_cm >= scr_safe else (
            COLOR_WARNING if metrics.eye_distance_cm >= scr_warn else COLOR_CRITICAL
        )

        cv2.putText(frame, f"Screen: {metrics.eye_distance_cm:.0f}cm",
                    (nx - 35, ny - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color, 1, cv2.LINE_AA)
        cv2.arrowedLine(frame, (nx, ny - 12), (nx, ny - 4),
                        color, 1, cv2.LINE_AA, tipLength=0.4)

    # --- Eye-to-Desk Distance Indicator ---
    if metrics.eye_to_desk_cm > 0 and desk_y > 0 and nose is not None:
        nx, ny = int(nose.x), int(nose.y)
        color = COLOR_GOOD if metrics.eye_to_desk_cm >= desk_safe else (
            COLOR_WARNING if metrics.eye_to_desk_cm >= desk_warn else COLOR_CRITICAL
        )
        # Draw dotted or thin line from nose to desk
        guide_x = nx + 40
        cv2.line(frame, (guide_x, ny), (guide_x, desk_y), color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Desk: {metrics.eye_to_desk_cm:.0f}cm",
                    (guide_x + 5, (ny + desk_y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    color, 1, cv2.LINE_AA)
