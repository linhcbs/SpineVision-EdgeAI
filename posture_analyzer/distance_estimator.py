"""
Eye-to-Screen Distance Estimator.
Uses the Pinhole Camera Model with Interpupillary Distance (IPD) as the known
physical reference to estimate the user's distance from the screen/camera.

For MediaPipe models: Also incorporates the z-coordinate (relative depth) as
a secondary signal to improve robustness.

Clinical thresholds (ISO 9241 / Ergonomics guidelines):
    >= 50 cm : Safe distance for computer screens
    35 - 49 cm : Slightly close (mild eye strain risk)
    < 35 cm : Too close — significant myopia / accommodation spasm risk

Reference:
    Average adult interpupillary distance (IPD) ≈ 63 mm (range: 54-72 mm)
    Source: Dodgson, N.A. (2004). "Variation and Extrema of Human IPD"
"""

import math
import numpy as np
from typing import Optional, Tuple

from .types import UnifiedKeypoints, NormalizedKeypoint, KeypointFormat
from .config import get_posture_config


# Physical constants
AVERAGE_IPD_MM = 63.0          # Average adult interpupillary distance (mm)
AVERAGE_IPD_CM = AVERAGE_IPD_MM / 10.0  # 6.3 cm


class DistanceEstimator:
    """
    Estimates the distance from the user's eyes to the camera/screen
    and to the desk surface using the Pinhole Camera Model.
    
    D = (f_focal * W_real) / W_pixel
    
    Where:
        f_focal: Camera focal length in pixels (estimated during calibration)
        W_real: Known physical width (IPD ≈ 6.3 cm)
        W_pixel: Measured width in pixels (distance between eye landmarks)
    
    Parameters:
        ipd_real_cm: Real-world interpupillary distance in cm (default: 6.3)
        default_focal_length: Fallback focal length if not calibrated.
    """

    def __init__(
        self,
        ipd_real_cm: Optional[float] = None,
        default_focal_length: Optional[float] = None,
    ):
        cfg = get_posture_config().get("metrics", {}).get("eye_to_screen_distance", {})
        self.ipd_real_cm = ipd_real_cm if ipd_real_cm is not None else cfg.get("average_ipd_cm", AVERAGE_IPD_CM)
        self.focal_length_px = default_focal_length if default_focal_length is not None else cfg.get("default_focal_length_px", 650.0)
        self.z_depth_dampening = cfg.get("z_depth_dampening", 0.3)
        self._is_focal_calibrated = False


    def calibrate_focal_length(self, ipd_pixels: float, known_distance_cm: float):
        """
        Calibrate the camera's focal length using a known distance.
        
        During calibration mode, the user sits at a known distance (e.g., 60 cm)
        and the system measures the IPD in pixels. The focal length is then:
        
            f = (ipd_pixels * known_distance_cm) / ipd_real_cm
        
        Args:
            ipd_pixels: Measured IPD in pixels during calibration
            known_distance_cm: Known distance user is sitting at during calibration
        """
        if ipd_pixels > 1.0:
            self.focal_length_px = (ipd_pixels * known_distance_cm) / self.ipd_real_cm
            self._is_focal_calibrated = True

    def measure_ipd_pixels(self, kps: UnifiedKeypoints) -> Optional[float]:
        """
        Measure the Interpupillary Distance (IPD) in pixels from keypoints.
        
        For MediaPipe: Uses left_eye (2) and right_eye (5) landmarks.
        For COCO: Uses left_eye (1) and right_eye (2) landmarks.
        
        Returns:
            IPD in pixels, or None if eye keypoints are not visible.
        """
        l_eye = kps.get("left_eye")
        r_eye = kps.get("right_eye")

        if l_eye is None or r_eye is None:
            return None
        if l_eye.score < 0.3 or r_eye.score < 0.3:
            return None

        dx = l_eye.x - r_eye.x
        dy = l_eye.y - r_eye.y
        ipd_px = math.sqrt(dx * dx + dy * dy)
        
        # Sanity check: IPD should be at least a few pixels
        if ipd_px < 3.0:
            return None

        return ipd_px

    def estimate_distance(self, kps: UnifiedKeypoints) -> float:
        """
        Estimate the eye-to-screen distance in centimeters.
        
        Primary method: Pinhole camera model using IPD.
        Secondary signal (MediaPipe only): z-coordinate of nose/eye.
        
        Returns:
            Estimated distance in cm. Returns -1.0 if estimation fails.
        """
        ipd_px = self.measure_ipd_pixels(kps)
        if ipd_px is None or ipd_px < 3.0:
            return -1.0

        # Primary: Pinhole model
        distance_cm = (self.focal_length_px * self.ipd_real_cm) / ipd_px

        # Secondary: If MediaPipe, use z-coordinate as a correction factor
        if kps.has_depth:
            nose = kps.get("nose")
            if nose is not None and nose.score >= 0.3:
                # MediaPipe z is relative depth: negative = closer to camera
                # Use it as a multiplicative correction (small adjustment)
                z_correction = 1.0 + nose.z * 0.3  # Damped z influence
                z_correction = max(0.5, min(1.5, z_correction))
                distance_cm *= z_correction

        # Clamp to reasonable range (15 cm to 200 cm)
        distance_cm = max(15.0, min(200.0, distance_cm))
        return round(distance_cm, 1)

    def estimate_eye_to_desk(self, kps: UnifiedKeypoints, desk_y: int) -> float:
        """
        Estimate vertical Eye-to-Desk distance in centimeters.
        Uses pixel height delta from eye line to detected desk surface,
        scaled by physical IPD reference (px to cm ratio).
        
        Args:
            kps: Unified keypoints from current frame.
            desk_y: Detected desk surface y-coordinate in pixels.
            
        Returns:
            Distance in cm, or -1.0 if not computable.
        """
        if desk_y <= 0:
            return -1.0

        # Determine eye y position
        l_eye = kps.get("left_eye")
        r_eye = kps.get("right_eye")
        if l_eye is not None and r_eye is not None and l_eye.score >= 0.3 and r_eye.score >= 0.3:
            eye_y = (l_eye.y + r_eye.y) / 2.0
        else:
            nose = kps.get("nose")
            if nose is not None and nose.score >= 0.3:
                eye_y = nose.y
            else:
                return -1.0

        delta_y_px = desk_y - eye_y
        if delta_y_px <= 5.0:
            # Desk is at or above eye level, invalid
            return -1.0

        ipd_px = self.measure_ipd_pixels(kps)
        if ipd_px is None or ipd_px < 3.0:
            return -1.0

        # Centimeters per pixel at the user's distance plane
        px_to_cm = self.ipd_real_cm / ipd_px
        eye_to_desk_cm = delta_y_px * px_to_cm

        # Clamp to reasonable range (10 cm to 150 cm)
        eye_to_desk_cm = max(10.0, min(150.0, eye_to_desk_cm))
        return round(eye_to_desk_cm, 1)

    @property
    def is_calibrated(self) -> bool:
        return self._is_focal_calibrated

    def reset(self):
        """Reset to default uncalibrated state."""
        self.focal_length_px = 650.0
        self._is_focal_calibrated = False

