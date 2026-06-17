"""Moku Moku no Mi — smoke fruit (Phase 2).

The person becomes a ghost: their body is replaced by a dim, desaturated,
blue-grey spectral form at ~30 % of original brightness so the background
reads clearly through them. Animated smoke (two fractal-noise layers morphing
over time at different scroll speeds) concentrates in a ring at the body
outline — so it looks like it's rising FROM the ghost — then trails off
outward as wispy haze. A sigmoid density curve gives soft, organic wisp edges
instead of the old hard-threshold cutoff.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from effects._blend import screen
from effects.base import BaseEffect
from utils.tracking import Tracking

# Ghost appearance
_GHOST_ALPHA  = 0.88    # how strongly the dim ghost replaces the body (0-1)
# Ghost colour channels as fractions of luminance (BGR) — B>G>R gives cool spectral tint.
_GHOST_B = 0.46
_GHOST_G = 0.38
_GHOST_R = 0.30

# Smoke animation
_SCROLL_A     = 52.0    # px/s — first noise layer
_SCROLL_B     = 33.0    # px/s — second noise layer (different speed = more natural)
_MORPH_PERIOD = 9.0     # seconds for one full blend cycle between the two layers

# Smoke gate geometry
_DILATION_PX  = 50      # how far smoke extends beyond body
_ERODE_PX     = 14      # inset before ring starts (body border width)

# Smoke colour (BGR) — cool grey-blue
_TINT_A = np.array([215, 205, 195], np.float32)
_TINT_B = np.array([228, 218, 208], np.float32)


# ------------------------------------------------------------------
# Noise generation (runs once on first frame / on resize)
# ------------------------------------------------------------------

def _fractal_noise(h: int, w: int) -> np.ndarray:
    """Multi-octave value noise in [0, 1], softened for smoke texture."""
    acc   = np.zeros((h, w), np.float32)
    amp   = 1.0
    total = 0.0
    for size in (4, 8, 16, 32, 64):
        tile = np.random.rand(size, size).astype(np.float32)
        up   = cv2.resize(tile, (w, h), interpolation=cv2.INTER_CUBIC)
        acc  += amp * up
        total += amp
        amp  *= 0.55
    acc /= total
    acc  = cv2.GaussianBlur(acc, (0, 0), sigmaX=6)
    acc -= acc.min()
    mx   = acc.max()
    return acc / (mx + 1e-6) if mx > 1e-6 else acc


def _sigmoid(x: np.ndarray, center: float = 0.50, k: float = 10.0) -> np.ndarray:
    """Smooth density curve — soft wisp edges instead of a hard threshold."""
    return np.clip(1.0 / (1.0 + np.exp(-k * (x - center))), 0.0, 1.0)


# ------------------------------------------------------------------

class MokuMoku(BaseEffect):
    name = "Moku Moku"
    swatch = (150, 150, 150)
    requires = {"mask"}

    def __init__(self):
        super().__init__()
        self._noise_a: Optional[np.ndarray] = None
        self._noise_b: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self._noise_a = None
        self._noise_b = None

    def _ensure_noise(self, w: int, h: int) -> None:
        if self._noise_a is None or self._size != (h, w):
            self._noise_a = _fractal_noise(h, w)
            self._noise_b = _fractal_noise(h, w)
            self._size    = (h, w)

    # ------------------------------------------------------------------
    # Ghost step: dim, desaturated, cool-tinted body
    # ------------------------------------------------------------------

    def _apply_ghost(self, frame: np.ndarray,
                     mask: Optional[np.ndarray]) -> np.ndarray:
        """Replace body pixels with a dim spectral form (~30 % brightness)."""
        if mask is None:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Scale each channel independently: B gets the most, R the least →
        # cool blue-grey that reads as supernatural, not just greyscale.
        ghost = np.stack([
            np.clip(gray * _GHOST_B + 8, 0, 255),   # B
            np.clip(gray * _GHOST_G + 5, 0, 255),   # G
            np.clip(gray * _GHOST_R + 3, 0, 255),   # R
        ], axis=2).astype(np.uint8)

        # Feather mask edges so the ghost has no hard border.
        soft  = cv2.GaussianBlur(mask, (0, 0), sigmaX=9)
        alpha = np.clip(soft, 0.0, 1.0)[:, :, None] * _GHOST_ALPHA

        # over(frame, ghost, alpha) = frame*(1-alpha) + ghost*alpha
        base_f  = frame.astype(np.float32)
        ghost_f = ghost.astype(np.float32)
        out_f   = base_f * (1.0 - alpha) + ghost_f * alpha
        return np.clip(out_f, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Smoke step: animated dual-layer noise
    # ------------------------------------------------------------------

    def _apply_smoke(self, src: np.ndarray,
                     mask: Optional[np.ndarray], t: float) -> np.ndarray:
        if self._noise_a is None or self._noise_b is None:
            return src
        h, w = src.shape[:2]

        # Blend fraction oscillates slowly so the noise pattern evolves.
        blend = (np.sin(t * (2.0 * np.pi / _MORPH_PERIOD)) + 1.0) * 0.5

        # Each layer scrolls upward at its own speed (different offsets).
        off_a = int(t * _SCROLL_A) % h
        off_b = int(t * _SCROLL_B) % h
        noise = (np.roll(self._noise_a, off_a, axis=0) * blend +
                 np.roll(self._noise_b, off_b, axis=0) * (1.0 - blend))

        # Sigmoid density: gradual fade-in/out → organic wisps.
        density = _sigmoid(noise, center=0.50, k=10.0)

        # Smoke gate — three zones:
        #   ring  : dense smoke at the body outline (where ghost meets air)
        #   inner : light haze inside the body (ghost is still somewhat visible)
        #   outer : fading wisps beyond the body
        if mask is not None:
            body     = (mask > 0.35).astype(np.uint8)
            dilated  = cv2.dilate(body, np.ones((_DILATION_PX, _DILATION_PX),
                                                 np.uint8))
            eroded   = cv2.erode(body,  np.ones((_ERODE_PX,   _ERODE_PX),
                                                 np.uint8))

            ring     = np.clip(
                dilated.astype(np.float32) - eroded.astype(np.float32), 0, 1
            )
            ring     = cv2.GaussianBlur(ring, (33, 33), 0)

            outer    = cv2.GaussianBlur(dilated.astype(np.float32), (71, 71), 0)

            inner    = cv2.GaussianBlur(eroded.astype(np.float32),  (25, 25), 0)

            # Ring is fullest; outer tapers; inner is subtle so ghost stays visible.
            gate = np.clip(ring * 1.0 + outer * 0.40 + inner * 0.25, 0.0, 1.0)
            density *= gate

        # Interpolate colour between the two tint shades as the layers morph.
        tint  = _TINT_A * blend + _TINT_B * (1.0 - blend)
        smoke = np.clip(density[:, :, None] * tint, 0, 255).astype(np.uint8)

        return screen(src, smoke)

    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        h, w = frame.shape[:2]
        self._ensure_noise(w, h)

        out = self._apply_ghost(frame, mask)   # dim spectral ghost body
        out = self._apply_smoke(out, mask, t)  # animated smoke from the outline

        return out
