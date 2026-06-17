"""Hand-gesture detection shared by every effect that reads hands.

All thresholds are scale-invariant: distances between fingertips are divided by
a reference "hand size" (wrist to middle-finger knuckle) so a hand near the
camera and one far away pinch at the same numeric threshold.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# MediaPipe Hands landmark indices we care about.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5          # index-finger knuckle
INDEX_TIP = 8
MIDDLE_MCP = 9         # middle-finger knuckle (good hand-size anchor)
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20
_FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def _xy(landmarks: np.ndarray, idx: int, w: int, h: int) -> np.ndarray:
    return np.array([landmarks[idx, 0] * w, landmarks[idx, 1] * h], dtype=np.float64)


def hand_size(landmarks: np.ndarray, w: int, h: int) -> float:
    """Reference length in pixels: wrist to middle-finger knuckle.

    Used to normalise every other distance so thresholds are scale-free.
    """
    return float(np.linalg.norm(_xy(landmarks, WRIST, w, h) - _xy(landmarks, MIDDLE_MCP, w, h)) + 1e-6)


def pinch_point(landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
    """Midpoint between thumb tip and index tip, in pixels."""
    return 0.5 * (_xy(landmarks, THUMB_TIP, w, h) + _xy(landmarks, INDEX_TIP, w, h))


def pinch_strength(landmarks: np.ndarray, w: int, h: int) -> float:
    """0 = wide open, ~1 = thumb and index touching. Normalised by hand size."""
    d = np.linalg.norm(_xy(landmarks, THUMB_TIP, w, h) - _xy(landmarks, INDEX_TIP, w, h))
    ratio = d / hand_size(landmarks, w, h)
    # ratio ~0.1 when touching, ~0.9 when spread → remap to 0..1 (touching = 1).
    return float(np.clip(1.0 - (ratio - 0.15) / 0.6, 0.0, 1.0))


def is_pinch(landmarks: np.ndarray, w: int, h: int, threshold: float = 0.6) -> bool:
    """True when thumb and index are close enough to count as a pinch."""
    return pinch_strength(landmarks, w, h) >= threshold


def is_fist(landmarks: np.ndarray, w: int, h: int, threshold: float = 1.1) -> bool:
    """True when all four fingertips are curled close to the palm.

    Compares mean fingertip→wrist distance against hand size; a closed fist
    pulls the tips in toward the wrist.
    """
    size = hand_size(landmarks, w, h)
    wrist = _xy(landmarks, WRIST, w, h)
    tips = np.array([_xy(landmarks, t, w, h) for t in _FINGER_TIPS])
    mean_dist = float(np.mean(np.linalg.norm(tips - wrist, axis=1)))
    return (mean_dist / size) <= threshold


def point_direction(landmarks: np.ndarray, w: int, h: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Index-finger pointing ray as (origin, unit_direction), or None if degenerate.

    Handy for "fire from the fingertip" style effects.
    """
    mcp = _xy(landmarks, INDEX_MCP, w, h)
    tip = _xy(landmarks, INDEX_TIP, w, h)
    v = tip - mcp
    n = np.linalg.norm(v)
    if n < 1e-3:
        return None
    return tip, v / n
