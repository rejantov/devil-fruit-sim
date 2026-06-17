"""Landmark jitter filters.

Raw MediaPipe landmarks wobble a few pixels every frame even when you hold
still. Every effect from Phase 1 onward looks bad without smoothing, so this is
the shared foundation the plan calls out building first.

Two tools live here:

* ``moving_average`` — the dead-simple thing the plan says is "enough to start".
* ``OneEuroFilter`` / ``LandmarkSmoother`` — a small upgrade that adapts: heavy
  smoothing when a point is still (kills jitter), light smoothing when it moves
  fast (kills lag). This is what the effects actually use.

Nothing here imports OpenCV or MediaPipe so it stays trivially testable.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np


def moving_average(history: Deque[np.ndarray], value: np.ndarray, window: int) -> np.ndarray:
    """Append ``value`` to ``history`` and return the mean of the last ``window``.

    ``history`` is owned by the caller (e.g. a ``deque``) so the running window
    survives across frames. This is the simplest possible smoother — start here
    if One Euro ever feels like overkill.
    """
    history.append(np.asarray(value, dtype=np.float64))
    while len(history) > window:
        history.popleft()
    return np.mean(history, axis=0)


class OneEuroFilter:
    """1€ filter (Casiez et al., 2012) for a single scalar signal.

    The trick: the cutoff frequency rises with the signal's speed. Slow/still =
    low cutoff = strong smoothing. Fast = high cutoff = responsive. Tune with:

    * ``min_cutoff`` — lower means more smoothing when still (more lag).
    * ``beta`` — higher means it lets fast motion through more readily.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class LandmarkSmoother:
    """Smooth a set of named 2D/3D points, one 1€ filter per coordinate.

    Effects call ``smoother.smooth("wrist", point, t)`` each frame with the raw
    point in pixels and the current timestamp. Filters are created lazily, so
    you can smooth whatever keys you like without declaring them up front.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._filters: Dict[str, list[OneEuroFilter]] = {}

    def smooth(self, key: str, point: np.ndarray, t: float) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        filters = self._filters.get(key)
        if filters is None or len(filters) != point.shape[0]:
            filters = [OneEuroFilter(self._min_cutoff, self._beta) for _ in range(point.shape[0])]
            self._filters[key] = filters
        return np.array([f(float(v), t) for f, v in zip(filters, point)])

    def reset(self, key: Optional[str] = None) -> None:
        """Drop filter state so the next sample is taken as-is (snap, no lag)."""
        if key is None:
            self._filters.clear()
        else:
            self._filters.pop(key, None)


def make_history(maxlen: int = 8) -> Deque[np.ndarray]:
    """Convenience: a fixed-size deque for use with :func:`moving_average`."""
    return deque(maxlen=maxlen)
