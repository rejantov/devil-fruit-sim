"""Mera Mera no Mi — flame fruit (Phase 2).

DOOM/Amiga-style heat-grid fire with three heat sources:

  1. Flickering hot baseline along the body silhouette bottom.
  2. Hot blobs at the hands so flames stream off the palms.
  3. Motion blazing: grayscale frame-difference in the body region seeds extra
     heat, so fast movement leaves a fiery afterimage trail that rises with
     the rest of the flames.

Grid is 1/2 the frame (was 1/3) — 4× more cells → smooth fire not chunky blobs.
Higher COOLING + lower DECAY let heat accumulate → bigger, denser flames.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from effects._blend import screen
from effects.base import BaseEffect
from utils import gesture
from utils.tracking import Tracking

DOWNSCALE     = 2       # heat grid = 1/DOWNSCALE of frame (was 3)
COOLING       = 0.990   # heat retained per frame (was 0.985 — higher = more fire)
DECAY         = 0.025   # random per-cell cooling  (was 0.045 — lower = more fire)
MOTION_THRESH = 16      # pixel diff below this is camera noise, ignored
MOTION_SCALE  = 110.0   # diff value that maps to full heat (fast arm = max blaze)


def _build_palette() -> np.ndarray:
    """256-entry BGR fire LUT: black at 0 → white at 255."""
    stops = [
        (0.00,   0,   0,   0),
        (0.25,   0,   0, 110),
        (0.45,  10,  40, 210),
        (0.62,  20, 130, 255),
        (0.80, 130, 230, 255),
        (1.00, 255, 255, 255),
    ]
    xs  = np.linspace(0.0, 1.0, 256)
    pts = np.array([s[0] for s in stops])
    lut = np.zeros((256, 3), np.uint8)
    for ch in range(3):
        vals = np.array([s[ch + 1] for s in stops])
        lut[:, ch] = np.clip(np.interp(xs, pts, vals), 0, 255).astype(np.uint8)
    return lut


class MeraMera(BaseEffect):
    name = "Mera Mera"
    swatch = (20, 90, 240)  # fiery orange-red (BGR)
    requires = {"mask", "hands"}

    def __init__(self):
        super().__init__()
        self._palette   = _build_palette()
        self._heat: Optional[np.ndarray]      = None
        self._size: Optional[Tuple[int, int]] = None   # (gh, gw)
        self._prev_gray: Optional[np.ndarray] = None   # for motion detection

    def reset(self) -> None:
        self._heat      = None
        self._prev_gray = None

    # ------------------------------------------------------------------
    def _ensure_grid(self, w: int, h: int) -> Tuple[int, int]:
        gh, gw = h // DOWNSCALE, w // DOWNSCALE
        if self._heat is None or self._size != (gh, gw):
            self._heat = np.zeros((gh, gw), np.float32)
            self._size = (gh, gw)
        return gh, gw

    def _motion_heat(self, gray: np.ndarray, gw: int, gh: int,
                     mask_small: Optional[np.ndarray]) -> np.ndarray:
        """Frame-diff motion map downsampled to heat grid (float32, 0-1).

        Fast movement in the body region returns high values; still areas and
        background return 0. This is what seeds the afterimage blaze.
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros((gh, gw), np.float32)

        diff = cv2.absdiff(gray, self._prev_gray).astype(np.float32)
        self._prev_gray = gray

        # Small blur to fuse adjacent motion pixels into a thicker trail.
        diff = cv2.GaussianBlur(diff, (7, 7), 0)

        diff_s = cv2.resize(diff, (gw, gh), interpolation=cv2.INTER_AREA)

        if mask_small is not None:
            diff_s *= mask_small   # ignore background motion

        return np.clip(
            (diff_s - MOTION_THRESH) / (MOTION_SCALE - MOTION_THRESH),
            0.0, 1.0,
        )

    def _seed(self, lm: Tracking, mask_small: Optional[np.ndarray],
              motion: np.ndarray) -> None:
        if self._heat is None or self._size is None:
            return
        gh, gw = self._size
        heat   = self._heat

        # 1. Motion blaze: high-diff body pixels ignite immediately.
        heat[:] = np.maximum(heat, motion * 0.95)

        # 2. Flickering hot baseline along the bottom two rows of the grid.
        flicker = np.random.rand(gw).astype(np.float32)
        heat[gh - 2:gh, :] = np.maximum(
            heat[gh - 2:gh, :], 0.70 + 0.30 * flicker
        )

        # 3. Ignite the lowest occupied mask row per column (body outline).
        if mask_small is not None:
            ys, xs = np.where(mask_small > 0.5)
            if ys.size:
                order  = np.lexsort((-ys, xs))
                xs_o, ys_o = xs[order], ys[order]
                first  = np.concatenate(([True], xs_o[1:] != xs_o[:-1]))
                heat[ys_o[first], xs_o[first]] = np.maximum(
                    heat[ys_o[first], xs_o[first]], 0.90
                )

        # 4. Hot blobs at the hands (larger radius than before).
        for hand in lm.hands:
            p  = gesture.pinch_point(hand.landmarks, lm.w, lm.h) / DOWNSCALE
            cx, cy = int(p[0]), int(p[1])
            r  = max(3, gw // 20)          # was gw//30 — bigger palm blobs
            y0, y1 = max(0, cy - r), min(gh, cy + r + 1)
            x0, x1 = max(0, cx - r), min(gw, cx + r + 1)
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], 1.0)

    def _propagate(self) -> None:
        if self._heat is None:
            return
        heat   = self._heat
        # Pull heat upward one row.
        rising = np.roll(heat, -1, axis=0)
        rising[-1, :] = heat[-1, :]           # bottom row keeps its seed
        rising = cv2.blur(rising, (3, 1))     # horizontal wind spread
        decay  = np.random.rand(*heat.shape).astype(np.float32) * DECAY
        self._heat = np.clip(rising * COOLING - decay, 0.0, 1.0)

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        h, w = frame.shape[:2]
        gh, gw = self._ensure_grid(w, h)

        # Convert raw frame to gray for motion detection (before any overlay).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_small = None
        if mask is not None:
            mask_small = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_LINEAR)

        motion = self._motion_heat(gray, gw, gh, mask_small)
        self._seed(landmarks, mask_small, motion)
        self._propagate()

        # Heat grid → colour via LUT, upscale with cubic for smooth gradients.
        assert self._heat is not None
        idx        = np.clip(self._heat * 255.0, 0, 255).astype(np.uint8)
        fire_small = self._palette[idx]
        fire       = cv2.resize(fire_small, (w, h), interpolation=cv2.INTER_CUBIC)

        # Gate to a dilated body region so flames hug and lick outward.
        if mask is not None:
            gate = cv2.dilate(
                (mask > 0.3).astype(np.uint8), np.ones((35, 35), np.uint8)
            )
            gate = cv2.GaussianBlur(gate.astype(np.float32), (41, 41), 0)[:, :, None]
            fire = (fire.astype(np.float32) * np.clip(gate, 0, 1)).astype(np.uint8)

        return screen(frame, fire)
