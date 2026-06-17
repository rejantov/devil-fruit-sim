"""Fruit-selector button bar.

A row of coloured-swatch buttons along the bottom of the frame: each shows the
fruit's swatch colour and name, and the active one gets a bright border. The bar
owns its own hit-testing so ``main.py`` just forwards mouse clicks to
:meth:`hit` and forwards the result to the effect dispatcher.

Swatch colours come straight from each effect's ``swatch`` attribute, so adding
a fruit needs zero work here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from effects.base import BaseEffect

BAR_HEIGHT = 54
PAD = 4


class ButtonBar:
    def __init__(self, effects: List[BaseEffect]):
        self._effects = effects
        self._rects: List[Tuple[int, int, int, int]] = []  # (x0, y0, x1, y1) per button

    def _layout(self, w: int, h: int) -> None:
        n = len(self._effects)
        bw = (w - PAD * (n + 1)) // n
        y0 = h - BAR_HEIGHT + PAD
        y1 = h - PAD
        self._rects = []
        x = PAD
        for _ in range(n):
            self._rects.append((x, y0, x + bw, y1))
            x += bw + PAD

    def hit(self, x: int, y: int) -> Optional[int]:
        """Return the index of the button under (x, y), or None."""
        for i, (x0, y0, x1, y1) in enumerate(self._rects):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    def draw(self, frame: np.ndarray, active: int) -> None:
        h, w = frame.shape[:2]
        self._layout(w, h)

        # Dim strip behind the buttons for legibility over busy video.
        strip = frame[h - BAR_HEIGHT:h, :].copy()
        cv2.rectangle(strip, (0, 0), (w, BAR_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.45, frame[h - BAR_HEIGHT:h, :], 0.55, 0,
                        dst=frame[h - BAR_HEIGHT:h, :])

        for i, (effect, (x0, y0, x1, y1)) in enumerate(zip(self._effects, self._rects)):
            cv2.rectangle(frame, (x0, y0), (x1, y1), effect.swatch, -1)
            is_active = (i == active)
            border = (255, 255, 255) if is_active else (40, 40, 40)
            cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2 if is_active else 1)

            label = f"{i}:{effect.name}"
            # Pick black or white text depending on swatch brightness.
            b, g, r = effect.swatch
            text_col = (20, 20, 20) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (245, 245, 245)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = x0 + max(4, ((x1 - x0) - tw) // 2)
            ty = y0 + (BAR_HEIGHT - 2 * PAD + th) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        text_col, 1, cv2.LINE_AA)
