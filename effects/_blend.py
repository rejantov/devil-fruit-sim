"""Tiny compositing helpers shared by the Logia (texture) effects.

The Phase-2 fruits are all the same shape: render a stylised version of the
frame, then paint it back only where the body mask says "person". These two
helpers keep that from being copy-pasted four times.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def as_alpha(mask: Optional[np.ndarray], shape) -> np.ndarray:
    """Return a float32 HxWx1 alpha in [0,1] for ``shape=(h, w)``.

    A missing mask means "apply everywhere" (alpha = 1), so effects still do
    *something* before segmentation has locked on.
    """
    h, w = shape[:2]
    if mask is None:
        return np.ones((h, w, 1), dtype=np.float32)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)[:, :, None]


def over(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Standard ``top`` over ``base`` using a float alpha (HxWx1 or scalar)."""
    base_f = base.astype(np.float32)
    top_f = top.astype(np.float32)
    out = base_f * (1.0 - alpha) + top_f * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def screen(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Photoshop 'screen' blend — good for light-emitting things (fire, smoke)."""
    b = base.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0
    return np.clip((1.0 - (1.0 - b) * (1.0 - t)) * 255.0, 0, 255).astype(np.uint8)
