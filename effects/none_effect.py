"""The 'no fruit' passthrough — also the default on launch."""

from __future__ import annotations

from typing import Optional

import numpy as np

from effects.base import BaseEffect
from utils.tracking import Tracking


class NoEffect(BaseEffect):
    name = "Off"
    swatch = (90, 90, 90)
    requires = set()  # no trackers → cheapest possible frame

    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        return frame
