"""Moku Moku no Mi — smoke fruit (Phase 2).

The body goes soft and hazy (blurred, with edges fading out) and translucent
smoke drifts upward around it. The smoke is one precomputed fractal-noise field
scrolled upward over time and screen-blended, so it costs almost nothing per
frame.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from effects._blend import over, screen
from effects.base import BaseEffect
from utils.tracking import Tracking

SCROLL_SPEED = 60.0       # pixels/second the smoke drifts upward
SMOKE_TINT = (200, 195, 190)  # cool grey (BGR)


def _fractal_noise(h: int, w: int) -> np.ndarray:
    """Cheap multi-octave value noise in [0,1], softened to look like smoke."""
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    total = 0.0
    for octave in (4, 8, 16, 32):
        small = np.random.rand(octave, octave).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += amp * up
        total += amp
        amp *= 0.55
    acc /= total
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=4)
    acc -= acc.min()
    return acc / (acc.max() + 1e-6)


class MokuMoku(BaseEffect):
    name = "Moku Moku"
    swatch = (150, 150, 150)  # smoke grey (BGR)
    requires = {"mask"}

    def __init__(self):
        self._smoke: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

    def _ensure_smoke(self, w: int, h: int) -> None:
        if self._smoke is None or self._size != (h, w):
            self._smoke = _fractal_noise(h, w)
            self._size = (h, w)

    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        h, w = frame.shape[:2]
        self._ensure_smoke(w, h)

        # 1) Soften the body: blur it and fade the edges so the person looks
        #    like they're dissolving into vapour rather than cut out.
        out = frame
        if mask is not None:
            blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=6)
            a = cv2.GaussianBlur((mask > 0.5).astype(np.float32), (41, 41), 0)
            a = (a * 0.7)[:, :, None]   # never fully opaque → translucent body
            out = over(frame, blurred, a)

        # 2) Drifting smoke: scroll the noise field upward and screen-blend it,
        #    concentrated in a dilated band around the body.
        offset = int(t * SCROLL_SPEED) % h
        drift = np.roll(self._smoke, offset, axis=0)
        density = np.clip((drift - 0.45) / 0.4, 0.0, 1.0)  # threshold into wisps

        if mask is not None:
            gate = cv2.dilate((mask > 0.3).astype(np.uint8), np.ones((45, 45), np.uint8))
            gate = cv2.GaussianBlur(gate.astype(np.float32), (61, 61), 0)
            density *= gate

        smoke = (density[:, :, None] * np.array(SMOKE_TINT, np.float32)).astype(np.uint8)
        return screen(out, smoke)
