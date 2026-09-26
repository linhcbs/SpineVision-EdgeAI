"""
Workspace Surface & Screen Detection Module.

Detects the desk/floor surface and computer screen/monitor in the webcam frame
by analyzing the spatial distribution of pose keypoints (wrists, hips) and
simple geometric heuristics.

This is a lightweight, keypoint-based approach (no object detection model needed).
It works by:
  1. Estimating desk surface from wrist/hip positions when user is seated.
  2. Estimating screen position from head/eye gaze direction.
  3. Drawing visual indicators on the frame.
"""

import cv2
import math
import numpy as np
from typing import Optional, Tuple, List

from .types import UnifiedKeypoints, NormalizedKeypoint
from .config import get_posture_config


# Colors (BGR)
COLOR_DESK_LINE = (180, 140, 60)      # Warm brown for desk surface
COLOR_DESK_FILL = (180, 140, 60)      # Fill color for desk area
COLOR_SCREEN_RECT = (255, 180, 0)     # Cyan-ish for screen indicator
COLOR_FLOOR_LINE = (100, 100, 100)    # Gray for floor
COLOR_TEXT = (220, 220, 220)


class WorkspaceDetector:
    """
    Detects workspace surfaces (desk, floor) and screen position from pose keypoints.
    
    Strategy:
      - Desk: Estimated from the average y-position of wrists when they are
        near the body (user typing/writing). The desk surface is assumed to
        be at or slightly below wrist level.
      - Screen: Estimated from the direction the user's head is facing.
        The screen center is assumed to be in front of and slightly above
        the eye line.
    
    Parameters:
        desk_smoothing: EMA smoothing factor for desk position (0-1).
        min_confidence: Minimum keypoint confidence to use.
    """

    def __init__(self, desk_smoothing: Optional[float] = None, min_confidence: Optional[float] = None):
        cfg = get_posture_config().get("workspace_detection", {})
        self.desk_smoothing = desk_smoothing if desk_smoothing is not None else cfg.get("desk_smoothing", 0.15)
        self.min_confidence = min_confidence if min_confidence is not None else cfg.get("min_confidence", 0.3)
        self.desk_offset_ratio = cfg.get("desk_offset_ratio", 0.05)
        self.screen_width_ratio = cfg.get("screen_width_ratio", 0.65)
        self.screen_aspect_ratio = cfg.get("screen_aspect_ratio", 0.5625)

        
        self._desk_y: Optional[float] = None            # Smoothed desk y-coordinate
        self._screen_center: Optional[Tuple[float, float]] = None  # Smoothed screen center
        self._detection_count = 0

    def _is_valid(self, kp: Optional[NormalizedKeypoint]) -> bool:
        return kp is not None and kp.score >= self.min_confidence

    def detect(self, kps: UnifiedKeypoints) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
        """
        Detect workspace elements from current keypoints.
        
        Returns:
            (desk_y, screen_region)
            - desk_y: Y-coordinate of detected desk surface (-1 if not detected)
            - screen_region: (x1, y1, x2, y2) of estimated screen area (None if not detected)
        """
        desk_y = self._estimate_desk(kps)
        screen_region = self._estimate_screen(kps)
        self._detection_count += 1
        return desk_y, screen_region

    def _estimate_desk(self, kps: UnifiedKeypoints) -> int:
        """
        Estimate desk surface y-position.
        
        Heuristic: When seated at a desk, the wrists and elbows are typically
        at desk height. The desk is estimated as the average y of visible wrists,
        with a small offset downward.
        """
        l_wrist = kps.get("left_wrist")
        r_wrist = kps.get("right_wrist")
        l_elbow = kps.get("left_elbow")
        r_elbow = kps.get("right_elbow")
        l_hip = kps.get("left_hip")
        r_hip = kps.get("right_hip")

        wrist_ys = []
        if self._is_valid(l_wrist):
            wrist_ys.append(l_wrist.y)
        if self._is_valid(r_wrist):
            wrist_ys.append(r_wrist.y)

        if not wrist_ys:
            # Fallback: use elbows
            if self._is_valid(l_elbow):
                wrist_ys.append(l_elbow.y)
            if self._is_valid(r_elbow):
                wrist_ys.append(r_elbow.y)

        if not wrist_ys:
            return int(self._desk_y) if self._desk_y is not None else -1

        # Current desk estimate: average wrist y + small offset
        raw_desk_y = np.mean(wrist_ys) + 15  # Slightly below wrists

        # Additional validation: desk should be between shoulders and bottom of frame
        mid_shoulder = kps.get_midpoint("left_shoulder", "right_shoulder")
        if mid_shoulder is not None and raw_desk_y < mid_shoulder.y:
            # Wrists are above shoulders — likely raised, not at desk
            # Keep previous estimate
            if self._desk_y is not None:
                return int(self._desk_y)
            return -1

        # Clamp desk to reasonable range
        raw_desk_y = min(raw_desk_y, kps.frame_height * 0.95)

        # Apply EMA smoothing
        if self._desk_y is None:
            self._desk_y = raw_desk_y
        else:
            self._desk_y = (self.desk_smoothing * raw_desk_y +
                           (1.0 - self.desk_smoothing) * self._desk_y)

        return int(self._desk_y)

    def _estimate_screen(self, kps: UnifiedKeypoints) -> Optional[Tuple[int, int, int, int]]:
        """
        Estimate the computer screen/monitor position.
        
        Heuristic: The screen is assumed to be:
          - Horizontally centered on the midpoint between the user's eyes/nose
          - Vertically at or slightly above eye level
          - Width: ~40% of frame width (rough monitor size at typical distance)
        
        Only valid when the user appears to be seated at a desk.
        """
        nose = kps.get("nose")
        l_eye = kps.get("left_eye")
        r_eye = kps.get("right_eye")

        # Need at least nose or both eyes
        if not self._is_valid(nose) and not (self._is_valid(l_eye) and self._is_valid(r_eye)):
            return None

        # Reference point: midpoint of eyes or nose
        if self._is_valid(l_eye) and self._is_valid(r_eye):
            ref_x = (l_eye.x + r_eye.x) / 2.0
            ref_y = (l_eye.y + r_eye.y) / 2.0
        elif self._is_valid(nose):
            ref_x = nose.x
            ref_y = nose.y
        else:
            return None

        # Screen is "behind the camera" — we approximate it as a region
        # above and in front of the user's face direction.
        # Since the camera IS the screen (laptop), the screen center ≈ camera center.
        screen_cx = kps.frame_width / 2.0
        screen_cy = kps.frame_height * 0.15  # Upper portion of frame (where monitor would be)

        # Screen dimensions (approximate)
        screen_w = int(kps.frame_width * 0.35)
        screen_h = int(kps.frame_height * 0.20)

        x1 = int(screen_cx - screen_w / 2)
        y1 = int(screen_cy - screen_h / 2)
        x2 = int(screen_cx + screen_w / 2)
        y2 = int(screen_cy + screen_h / 2)

        # Clamp to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(kps.frame_width, x2)
        y2 = min(kps.frame_height, y2)

        return (x1, y1, x2, y2)

    def draw_workspace(
        self,
        frame: np.ndarray,
        desk_y: int,
        screen_region: Optional[Tuple[int, int, int, int]],
        opacity: float = 0.3,
    ):
        """
        Draw workspace indicators on the frame.
        
        Args:
            frame: BGR image to draw on (modified in-place).
            desk_y: Detected desk surface y-coordinate.
            screen_region: (x1, y1, x2, y2) of screen area.
            opacity: Opacity for filled overlay regions.
        """
        h, w = frame.shape[:2]

        # --- Draw Desk Surface ---
        if desk_y > 0 and desk_y < h:
            # Draw a semi-transparent filled region below the desk line
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, desk_y), (w, h), COLOR_DESK_FILL, -1)
            cv2.addWeighted(overlay, opacity * 0.5, frame, 1.0 - opacity * 0.5, 0, frame)

            # Draw the desk line itself
            cv2.line(frame, (0, desk_y), (w, desk_y), COLOR_DESK_LINE, 2, cv2.LINE_AA)

            # Draw small dashes to indicate "desk surface"
            dash_length = 20
            gap_length = 15
            x = 0
            while x < w:
                x_end = min(x + dash_length, w)
                cv2.line(frame, (x, desk_y), (x_end, desk_y), COLOR_DESK_LINE, 2, cv2.LINE_AA)
                x += dash_length + gap_length

            # Label
            cv2.putText(
                frame, "Desk Surface",
                (10, desk_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DESK_LINE, 1, cv2.LINE_AA,
            )

        # --- Draw Screen Region ---
        if screen_region is not None:
            x1, y1, x2, y2 = screen_region
            # Semi-transparent rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_SCREEN_RECT, -1)
            cv2.addWeighted(overlay, opacity * 0.3, frame, 1.0 - opacity * 0.3, 0, frame)

            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SCREEN_RECT, 1, cv2.LINE_AA)

            # Corner brackets for a "monitor" feel
            bracket_len = 15
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), COLOR_SCREEN_RECT, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), COLOR_SCREEN_RECT, 2)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), COLOR_SCREEN_RECT, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), COLOR_SCREEN_RECT, 2)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), COLOR_SCREEN_RECT, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), COLOR_SCREEN_RECT, 2)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), COLOR_SCREEN_RECT, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), COLOR_SCREEN_RECT, 2)

            # Label
            label = "Screen (est.)"
            cv2.putText(
                frame, label,
                (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_SCREEN_RECT, 1, cv2.LINE_AA,
            )

    def reset(self):
        """Reset all detection state."""
        self._desk_y = None
        self._screen_center = None
        self._detection_count = 0
