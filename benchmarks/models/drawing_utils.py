"""
Drawing and Visualization Utilities for Human Pose Estimation.
Supports:
- 3D Depth-aware rendering (MediaPipe Z-coordinate mapped to point size and color intensity)
- Confidence/Visibility score scaling for 2D models (MoveNet, YOLO)
- HUD Overlay (FPS, Latency, RAM, Persons count, Device)
- Safe camera acquisition with fallback indices
- Config loading from pose_models_config.json
"""

import os
import json
import cv2
import numpy as np


# Default Color Palette (BGR)
COLOR_FPS     = (0, 255, 255)    # Yellow
COLOR_LAT     = (0, 255, 128)    # Light Green
COLOR_RAM     = (255, 200, 0)    # Cyan
COLOR_MODEL   = (200, 200, 255)  # Soft Pink/White
COLOR_BONE_MP = (0, 220, 0)      # Green for MediaPipe
COLOR_JOINT_B = (255, 255, 255)  # White border

# COCO Body Part Color Themes
COCO_PART_COLORS = {
    "head":  (0, 255, 255),  # Yellow
    "upper": (0, 200, 255),  # Orange-Yellow
    "torso": (0, 255, 100),  # Bright Green
    "legs":  (100, 0, 255),  # Magenta / Purple
}

COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # Head / Face
    (5, 6),                            # Shoulders
    (5, 7), (7, 9),                    # Left Arm
    (6, 8), (8, 10),                   # Right Arm
    (5, 11), (6, 12),                  # Torso Sides
    (11, 12),                          # Hips
    (11, 13), (13, 15),                # Left Leg
    (12, 14), (14, 16),                # Right Leg
]

COCO_BONE_COLORS = [
    COCO_PART_COLORS["head"],  COCO_PART_COLORS["head"],  COCO_PART_COLORS["head"], COCO_PART_COLORS["head"],
    COCO_PART_COLORS["upper"],
    COCO_PART_COLORS["upper"], COCO_PART_COLORS["upper"],
    COCO_PART_COLORS["upper"], COCO_PART_COLORS["upper"],
    COCO_PART_COLORS["torso"], COCO_PART_COLORS["torso"],
    COCO_PART_COLORS["torso"],
    COCO_PART_COLORS["legs"],  COCO_PART_COLORS["legs"],
    COCO_PART_COLORS["legs"],  COCO_PART_COLORS["legs"],
]


def load_config(model_key=None, config_path=None):
    """Load config from pose_models_config.json with optional model_key slice."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "pose_models_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = json.load(f)
            if model_key and "models" in full_cfg and model_key in full_cfg["models"]:
                return full_cfg["models"][model_key], full_cfg.get("global_settings", {})
            return full_cfg, full_cfg.get("global_settings", {})
    return {}, {}


def resolve_weight_path(filename):
    """Resolve absolute path to weight file, searching weights/ directory first."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_paths = [
        os.path.join(base_dir, "weights", filename),
        os.path.join(base_dir, "models", filename),
        os.path.join(base_dir, filename),
        filename
    ]
    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(base_dir, "weights", filename)



def open_camera(preferred_index=1, fallback_indices=None, width=1280, height=720):
    """Open camera safely with fallback list."""
    if fallback_indices is None:
        fallback_indices = [0, 1, 2, 3, 4, 5]
    search_order = [preferred_index] + [i for i in fallback_indices if i != preferred_index]

    for idx in search_order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if idx != preferred_index:
                    print(f"-> Notice: Preferred camera {preferred_index} unavailable. Connected to camera {idx}")
                return cap, idx
            cap.release()
    return None, -1


def compute_point_style(z=None, score=None, radius_min=3, radius_max=10, base_color=(0, 0, 255)):
    """
    Computes depth-aware or confidence-aware radius and color shading.
    - If z is provided (MediaPipe 3D): z < 0 is closer to screen, z > 0 is farther away.
      Closer landmarks get larger radius and deeper/darker/richer color.
      Farther landmarks get smaller radius and lighter/faint color.
    - If z is None: use confidence score to scale radius and intensity.
    """
    if z is not None:
        # MediaPipe z is normalized roughly in [-0.5, 0.5]
        # Closer to camera = negative z (depth_factor -> 1.0)
        # Farther from camera = positive z (depth_factor -> 0.0)
        depth_factor = np.clip((-z + 0.35) / 0.7, 0.0, 1.0)
    elif score is not None:
        depth_factor = np.clip(score, 0.0, 1.0)
    else:
        depth_factor = 0.5

    # Interpolate radius
    radius = int(radius_min + depth_factor * (radius_max - radius_min))

    # Modulate color: blend between faded tint and deep saturated color
    b, g, r = base_color
    # Closer: deep saturated tone; Farther: softer/lighter tone
    mod_b = int(np.clip(b * (0.4 + 0.6 * depth_factor) + (1 - depth_factor) * 80, 0, 255))
    mod_g = int(np.clip(g * (0.4 + 0.6 * depth_factor) + (1 - depth_factor) * 80, 0, 255))
    mod_r = int(np.clip(r * (0.4 + 0.6 * depth_factor) + (1 - depth_factor) * 80, 0, 255))
    color = (mod_b, mod_g, mod_r)

    return radius, color


def draw_mediapipe_landmarks_3d(frame, landmarks, connections, vis_thresh=0.2,
                                radius_min=3, radius_max=10, base_bone_color=COLOR_BONE_MP):
    """
    Renders MediaPipe skeleton with 3D depth-aware landmark circles:
    - Points closer to camera appear larger and darker/richer in color.
    - Points farther away appear smaller and lighter.
    """
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        z_val = getattr(lm, 'z', 0.0)
        vis   = getattr(lm, 'visibility', 1.0)
        pts.append((x_px, y_px, z_val, vis))

    # Draw bones
    for s, e in connections:
        if s >= len(pts) or e >= len(pts):
            continue
        x1, y1, z1, v1 = pts[s]
        x2, y2, z2, v2 = pts[e]
        if v1 < vis_thresh or v2 < vis_thresh:
            continue
        # Bone thickness can also slightly scale with average depth
        avg_z = (z1 + z2) / 2.0
        depth_factor = np.clip((-avg_z + 0.35) / 0.7, 0.0, 1.0)
        thickness = 3 if depth_factor > 0.6 else 2
        cv2.line(frame, (x1, y1), (x2, y2), base_bone_color, thickness, cv2.LINE_AA)

    # Draw keypoints with 3D depth styling
    for (x, y, z, v) in pts:
        if v < vis_thresh:
            continue
        # Base color: crimson/red for joints, adjust depth
        radius, color = compute_point_style(z=z, radius_min=radius_min, radius_max=radius_max, base_color=(0, 0, 220))
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        border_thick = 2 if radius >= 6 else 1
        cv2.circle(frame, (x, y), radius, COLOR_JOINT_B, border_thick, cv2.LINE_AA)


def draw_coco_landmarks(frame, keypoints, connections=COCO_CONNECTIONS, conf_thresh=0.3,
                        radius_min=3, radius_max=8):
    """
    Renders COCO 17-keypoint skeleton (MoveNet, YOLO):
    - Keypoint size and color intensity scaled by confidence score.
    - Distinct colors per anatomical region (Head, Upper, Torso, Legs).
    """
    h, w = frame.shape[:2]
    pts = []
    # Accept keypoints as (17, 3) with [y, x, score] or [x, y, score]
    for kp in keypoints:
        if len(kp) == 3:
            p1, p2, score = kp
            # MoveNet format: [y_norm, x_norm, score] where p1 and p2 are <= 1.0
            # YOLO format: [x_px, y_px, score] where p1 and p2 are pixel coordinates
            if float(p1) <= 1.0 and float(p2) <= 1.0 and (float(p1) > 0 or float(p2) > 0 or float(score) > 0):
                x_px = int(float(p2) * w)
                y_px = int(float(p1) * h)
            else:
                x_px = int(float(p1))
                y_px = int(float(p2))
            pts.append((x_px, y_px, float(score)))
        elif len(kp) == 2:
            pts.append((int(float(kp[0])), int(float(kp[1])), 1.0))

    # Draw bones
    for idx, (s, e) in enumerate(connections):
        if s >= len(pts) or e >= len(pts):
            continue
        x1, y1, v1 = pts[s]
        x2, y2, v2 = pts[e]
        if v1 < conf_thresh or v2 < conf_thresh:
            continue
        bone_color = COCO_BONE_COLORS[idx] if idx < len(COCO_BONE_COLORS) else (0, 200, 0)
        cv2.line(frame, (x1, y1), (x2, y2), bone_color, 2, cv2.LINE_AA)

    # Draw joints
    for i, (x, y, score) in enumerate(pts):
        if score < conf_thresh:
            continue
        if i < 5:
            base_col = COCO_PART_COLORS["head"]
        elif i < 11:
            base_col = COCO_PART_COLORS["upper"]
        elif i < 13:
            base_col = COCO_PART_COLORS["torso"]
        else:
            base_col = COCO_PART_COLORS["legs"]

        radius, color = compute_point_style(score=score, radius_min=radius_min, radius_max=radius_max, base_color=base_col)
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, COLOR_JOINT_B, 1, cv2.LINE_AA)


def draw_hud(frame, model_name, fps, latency_ms, ram_mb, n_persons, extra_text=""):
    """Renders semi-transparent HUD overlay on the upper left."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (350, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, model_name[:32],                     (14, 30),  font, 0.52, COLOR_MODEL, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS     : {fps:5.1f}",              (14, 58),  font, 0.58, COLOR_FPS,   2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency : {latency_ms:5.1f} ms",    (14, 84),  font, 0.58, COLOR_LAT,   2, cv2.LINE_AA)
    cv2.putText(frame, f"RAM     : {ram_mb:5.0f} MB",        (14, 110), font, 0.58, COLOR_RAM,   2, cv2.LINE_AA)
    
    person_txt = f"Persons : {n_persons}"
    if extra_text:
        person_txt += f" | {extra_text}"
    cv2.putText(frame, person_txt, (14, 136), font, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
