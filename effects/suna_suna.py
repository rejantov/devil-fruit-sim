"""Suna Suna no Mi — sand fruit (Phase 2).

Desaturated tan tone plus a granular noise texture, masked to the body. The
edges are eroded with a noisy threshold so the silhouette crumbles into grains
rather than ending on a clean line.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from effects._blend import over
from effects.base import BaseEffect
from utils.tracking import Tracking

# Sepia-style colour matrix, then biased toward warm tan.
_SEPIA = np.array([
    [0.27, 0.53, 0.18],   # B out
    [0.30, 0.62, 0.22],   # G out
    [0.37, 0.74, 0.27],   # R out
], np.float32)


class SunaSuna(BaseEffect):
    name = "Suna Suna"
    swatch = (130, 180, 215)  # sandy tan (BGR)
    requires = {"mask"}

    def _sandify(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        # Apply colour matrix (note BGR order: reverse to RGB-style math then back).
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        out = np.empty_like(f)
        out[:, :, 0] = _SEPIA[0, 0] * r + _SEPIA[0, 1] * g + _SEPIA[0, 2] * b
        out[:, :, 1] = _SEPIA[1, 0] * r + _SEPIA[1, 1] * g + _SEPIA[1, 2] * b
        out[:, :, 2] = _SEPIA[2, 0] * r + _SEPIA[2, 1] * g + _SEPIA[2, 2] * b

        # Granular texture: per-pixel grain modulating brightness.
        grain = (np.random.rand(*frame.shape[:2]).astype(np.float32) - 0.5) * 60.0
        out += grain[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        sand = self._sandify(frame)

        if mask is None:
            return sand
        # Crumble the edge: erode the hard mask, then knock holes in the border
        # band with noise so it disperses into grains.
        m = (mask > 0.5).astype(np.float32)
        crumble = np.random.rand(*m.shape).astype(np.float32)
        edge = m - cv2.erode(m, np.ones((9, 9), np.uint8))
        m = np.clip(m - edge * (crumble > 0.4), 0, 1)
        m = cv2.GaussianBlur(m, (7, 7), 0)[:, :, None]
        return over(frame, sand, m)
