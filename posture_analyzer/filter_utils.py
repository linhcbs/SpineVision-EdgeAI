"""
Temporal Landmark Smoothing Filters for Pose Estimation.
Eliminates micro-jittering caused by webcam sensor noise, lighting changes,
and micro-movements, resulting in stable angle/distance computations.

Implements:
  - One-Euro Filter (adaptive low-pass filter for HCI applications)
  - Exponential Moving Average (EMA) as a simpler fallback

Reference:
  Casiez, G., Roussel, N., Vogel, D. (2012).
  "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
  CHI '12 Proceedings, pp. 2527-2530.
"""

import math
import time
import numpy as np
from typing import Dict, Optional, Tuple


class OneEuroFilter:
    """
    One-Euro Filter — Adaptive frequency low-pass filter.
    
    When the signal is stable (user sitting still), it applies heavy smoothing
    to eliminate jitter. When the signal changes rapidly (user moving), it
    reduces smoothing to minimize lag.
    
    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing when still.
                    Recommended: 1.0 for pose landmarks.
        beta: Speed coefficient. Higher = less lag when moving fast.
              Recommended: 0.005 for pose landmarks.
        d_cutoff: Cutoff frequency for the derivative filter.
                  Recommended: 1.0 (usually doesn't need tuning).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Compute the exponential smoothing factor alpha from time period and cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a new sample.
        
        Args:
            x: New raw sample value
            t: Timestamp in seconds (uses wall clock if None)
        
        Returns:
            Filtered (smoothed) value
        """
        if t is None:
            t = time.perf_counter()

        if self._t_prev is None:
            # First sample — initialize state
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        # Time elapsed since last sample
        t_e = t - self._t_prev
        if t_e <= 0:
            t_e = 1e-6  # Avoid division by zero

        # --- Filter the derivative (speed of change) ---
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # --- Adaptive cutoff frequency based on speed ---
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # --- Filter the signal ---
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self):
        """Reset filter state for re-calibration or new session."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class EMAFilter:
    """
    Simple Exponential Moving Average filter.
    Suitable when computational simplicity is preferred over adaptive behavior.
    
    Parameters:
        alpha: Smoothing factor (0.0 - 1.0). Lower = more smoothing.
               Recommended: 0.3 - 0.5 for pose landmarks.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._prev: Optional[float] = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if self._prev is None:
            self._prev = x
            return x
        self._prev = self.alpha * x + (1.0 - self.alpha) * self._prev
        return self._prev

    def reset(self):
        self._prev = None


class LandmarkSmoother:
    """
    Multi-channel smoother for a full set of pose landmarks.
    Creates independent One-Euro filters for each (keypoint_name, axis) pair.
    
    This ensures each landmark dimension (x, y, z) is filtered independently
    with its own state, giving the best per-joint smoothing behavior.
    
    Parameters:
        min_cutoff: One-Euro min_cutoff (passed to each filter)
        beta: One-Euro beta (passed to each filter)
        use_ema: If True, use simple EMA instead of One-Euro
        ema_alpha: Alpha for EMA mode
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.005,
        use_ema: bool = False,
        ema_alpha: float = 0.4,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self._filters: Dict[Tuple[str, str], OneEuroFilter] = {}

    def _get_filter(self, key: Tuple[str, str]):
        """Get or create filter for a specific (landmark_name, axis) pair."""
        if key not in self._filters:
            if self.use_ema:
                self._filters[key] = EMAFilter(alpha=self.ema_alpha)
            else:
                self._filters[key] = OneEuroFilter(
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                )
        return self._filters[key]

    def smooth(self, keypoints: dict, t: Optional[float] = None) -> dict:
        """
        Apply temporal smoothing to a full keypoint dictionary.
        
        Args:
            keypoints: Dict[str, NormalizedKeypoint] from UnifiedKeypoints
            t: Current timestamp (seconds). If None, uses perf_counter.
        
        Returns:
            New dict with smoothed keypoint values (original objects are not mutated).
        """
        from .types import NormalizedKeypoint

        if t is None:
            t = time.perf_counter()

        smoothed = {}
        for name, kp in keypoints.items():
            fx = self._get_filter((name, "x"))
            fy = self._get_filter((name, "y"))
            fz = self._get_filter((name, "z"))

            smoothed[name] = NormalizedKeypoint(
                x=fx(kp.x, t),
                y=fy(kp.y, t),
                z=fz(kp.z, t),
                score=kp.score,  # Don't filter confidence scores
            )
        return smoothed

    def reset(self):
        """Reset all filter channels (e.g., when switching users or re-calibrating)."""
        self._filters.clear()
