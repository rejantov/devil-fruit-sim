"""The one contract every fruit implements.

Adding a fruit = write one new ``BaseEffect`` subclass and register it. Nothing
in ``main.py`` changes. Per the plan, the whole point of this abstraction is
that each fruit is a self-contained module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple

import numpy as np

from utils.tracking import Tracking


class BaseEffect(ABC):
    """Abstract fruit power.

    Subclasses set three class attributes and implement
    :meth:`process_frame`:

    * ``name`` — label shown on the button bar.
    * ``swatch`` — BGR colour for the button (OpenCV is BGR, not RGB).
    * ``requires`` — which trackers this effect reads, a subset of
      ``{"hands", "pose", "face", "mask"}``. The tracker runs only these, which
      is what keeps the frame rate up. Default: nothing (a free passthrough).
    """

    name: str = "Effect"
    swatch: Tuple[int, int, int] = (160, 160, 160)
    requires: Set[str] = set()

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        landmarks: Tracking,
        mask: Optional[np.ndarray],
        t: float,
    ) -> np.ndarray:
        """Return the rendered BGR frame.

        ``frame``      the current BGR webcam frame (already mirrored).
        ``landmarks``  a :class:`~utils.tracking.Tracking` snapshot; only the
                       fields named in ``requires`` are populated.
        ``mask``       feathered body mask (float32 HxW in [0,1]) or ``None``.
        ``t``          monotonic timestamp in seconds, for animation/smoothing.

        Implementations may modify ``frame`` in place or return a new array.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-activation state (called when the effect is switched
        away from). Stateful effects — fire grids, smoothers — override this."""
        pass
