"""Suna Suna no Mi — sand fruit (Phase 2).

Warm sandy colour grade on the body (luminance-mapped palette, organic grain)
plus a particle system: sand grains continuously fall from the body under
gravity with turbulent sideways drift. Raising an arm seeds extra grains at
the wrist so they stream off the fingertips.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from effects._blend import as_alpha, over
from effects.base import BaseEffect
from utils.tracking import Tracking

# Sand palette (BGR) — dark warm-brown → golden tan → light cream
_SHADOW  = np.array([ 38,  72, 110], np.float32)
_MIDTONE = np.array([ 88, 158, 210], np.float32)
_HILIGHT = np.array([178, 212, 238], np.float32)

# A few sandy/dusty colour variants so particles don't all look identical (BGR).
_COLORS = np.array([
    [100, 170, 215],   # warm gold
    [110, 185, 225],   # bright sand
    [ 78, 140, 192],   # deep amber
    [130, 195, 228],   # pale dust
    [ 95, 160, 205],   # mid tan
], dtype=np.float32)

_MAX_PART    = 900     # hard cap on simultaneous live particles
_GRAVITY     = 280.0   # px s⁻²  downward
_DRIFT_STD   = 28.0    # px s⁻¹  horizontal turbulence σ per frame
_LIFETIME    = 1.8     # seconds until a particle fades out
_SPAWN_BODY  = 20      # new grains from body each frame
_SPAWN_WRIST = 10      # extra grains per raised wrist each frame

# Pose landmark indices (matching tracking.py)
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_WRIST,    _R_WRIST    = 15, 16


class SunaSuna(BaseEffect):
    name = "Suna Suna"
    swatch = (130, 180, 215)  # sandy tan (BGR)
    requires = {"mask", "pose"}

    def __init__(self):
        super().__init__()
        n = _MAX_PART  # pre-allocated to the hard cap
        self._px    = np.zeros(n, np.float32)   # x positions
        self._py    = np.zeros(n, np.float32)   # y positions
        self._vx    = np.zeros(n, np.float32)   # x velocity  (px/s)
        self._vy    = np.zeros(n, np.float32)   # y velocity  (px/s)
        self._life  = np.zeros(n, np.float32)   # 1=fresh, 0=dead
        self._size  = np.ones(n, np.int32)      # radius in pixels
        self._cidx  = np.zeros(n, np.int32)     # index into _COLORS
        self._t_prev = -1.0

    def reset(self) -> None:
        self._life[:] = 0.0
        self._t_prev  = -1.0

    # ------------------------------------------------------------------
    # Sand colour grade
    # ------------------------------------------------------------------

    def _sandify(self, frame: np.ndarray) -> np.ndarray:
        """Map luminance to a warm sand palette and add organic grain."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lum  = gray[:, :, np.newaxis]

        pivot = 0.42
        t1 = np.clip(lum / pivot, 0.0, 1.0)
        t2 = np.clip((lum - pivot) / (1.0 - pivot), 0.0, 1.0)

        sand = np.where(
            lum < pivot,
            _SHADOW  * (1.0 - t1) + _MIDTONE * t1,
            _MIDTONE * (1.0 - t2) + _HILIGHT * t2,
        )

        # Keep 20 % of the original so the person stays recognisable.
        sand = 0.80 * sand + 0.20 * frame.astype(np.float32)

        # Two-scale grain: fine pixel noise + slightly blurred clusters.
        noise = np.random.rand(*frame.shape[:2]).astype(np.float32)
        coarse = cv2.GaussianBlur(noise, (7, 7), 0)
        grain  = (noise * 0.45 + coarse * 0.55 - 0.5) * 48.0
        sand  += grain[:, :, np.newaxis]

        return np.clip(sand, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Particle system
    # ------------------------------------------------------------------

    def _spawn(self, xy: np.ndarray, count: int) -> None:
        """Spawn *count* particles at positions sampled from the Nx2 array *xy*."""
        if len(xy) == 0 or count <= 0:
            return
        dead = np.where(self._life <= 0.0)[0]
        if len(dead) == 0:
            return
        count = min(count, len(dead))
        slots = dead[:count]
        pick  = np.random.randint(0, len(xy), count)

        self._px[slots]   = xy[pick, 0]
        self._py[slots]   = xy[pick, 1]
        self._vx[slots]   = np.random.uniform(-18, 18, count)
        self._vy[slots]   = np.random.uniform(-8,  12, count)  # slight upward burst
        self._life[slots] = 1.0
        self._size[slots] = np.random.choice([1, 1, 1, 2, 2, 3], count)
        self._cidx[slots] = np.random.randint(0, len(_COLORS), count)

    def _update(self, dt: float, h: int, w: int) -> None:
        alive = self._life > 0.0
        if not alive.any():
            return
        n = int(alive.sum())

        self._px[alive] += self._vx[alive] * dt
        self._py[alive] += self._vy[alive] * dt

        # Gravity pulls grains downward.
        self._vy[alive] += _GRAVITY * dt

        # Turbulent horizontal drift.
        self._vx[alive] += (np.random.randn(n).astype(np.float32)
                            * _DRIFT_STD * dt)

        # Age out.
        self._life[alive] -= dt / _LIFETIME
        self._life[self._life < 0.0] = 0.0

        # Kill anything that has left the frame.
        oob = (self._px < 0) | (self._px >= w) | (self._py >= h)
        self._life[oob] = 0.0

    def _draw(self, frame: np.ndarray) -> None:
        """Draw all live particles onto *frame* in-place."""
        h, w  = frame.shape[:2]
        alive = np.where(self._life > 0.0)[0]
        for i in alive:
            x = int(self._px[i])
            y = int(self._py[i])
            if not (0 <= x < w and 0 <= y < h):
                continue
            a = float(self._life[i])
            c = _COLORS[self._cidx[i]]
            color = (int(c[0] * a), int(c[1] * a), int(c[2] * a))
            cv2.circle(frame, (x, y), int(self._size[i]), color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        landmarks: Tracking,
        mask: Optional[np.ndarray],
        t: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        dt = (t - self._t_prev) if self._t_prev >= 0.0 else 1.0 / 30.0
        self._t_prev = t
        dt = float(np.clip(dt, 0.005, 0.10))

        # 1. Sand colour grade composited over the body mask.
        sand  = self._sandify(frame)
        alpha = as_alpha(mask, frame.shape)
        out   = over(frame, sand, alpha)

        # 2. Spawn body grains — bias toward the upper body so they fall
        #    through the frame rather than disappearing instantly at the feet.
        if mask is not None:
            ys, xs = np.where(mask > 0.45)
            if len(ys) > 0:
                upper = ys < int(h * 0.72)      # top ~70 % of the frame
                ys_u, xs_u = ys[upper], xs[upper]
                if len(ys_u) > 0:
                    body_pts = np.stack([xs_u, ys_u], axis=1).astype(np.float32)
                    self._spawn(body_pts, _SPAWN_BODY)

        # 3. Raised-wrist extra spawning.
        if landmarks.pose is not None:
            for wrist_idx, shoulder_idx in [(_L_WRIST, _L_SHOULDER),
                                             (_R_WRIST, _R_SHOULDER)]:
                if (landmarks.pose_visible(wrist_idx) and
                        landmarks.pose_visible(shoulder_idx)):
                    # In image coords y grows downward; wrist < shoulder means raised.
                    if landmarks.pose[wrist_idx, 1] < landmarks.pose[shoulder_idx, 1]:
                        wr = landmarks.pose_px(wrist_idx)
                        if wr is None:
                            continue
                        offsets = np.random.uniform(-22, 22, (_SPAWN_WRIST, 2))
                        wrist_pts = (wr[np.newaxis, :] + offsets).astype(np.float32)
                        self._spawn(wrist_pts, _SPAWN_WRIST)

        # 4. Advance physics.
        self._update(dt, h, w)

        # 5. Render particles on top.
        self._draw(out)

        return out
