"""Hie Hie no Mi — ice fruit (Phase 2).

Turns the body to faceted ice: desaturated blue tint, extra contrast, a frosty
speckle, and — when a face is visible — pale crystalline lines drawn along the
Face Mesh tessellation so the geometry reads as cut crystal rather than smooth
skin. (The plan's Voronoi alternative is the other route if you'd rather have
random cracks not tied to the face topology.)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from effects._blend import as_alpha, over
from effects.base import BaseEffect
from utils.tracking import Tracking

ICE_TINT = np.array([255, 200, 150], np.float32)  # light blue (BGR)


class HieHie(BaseEffect):
    name = "Hie Hie"
    swatch = (240, 200, 130)  # icy cyan-blue (BGR)
    requires = {"mask", "face"}

    def _iced(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.35                                 # desaturate
        hsv[:, :, 0] = 0.7 * hsv[:, :, 0] + 0.3 * 110.0      # nudge hue toward blue
        iced = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

        iced = iced.astype(np.float32)
        iced = (iced - 128.0) * 1.25 + 128.0                 # contrast bump
        iced = 0.78 * iced + 0.22 * ICE_TINT                 # cool tint
        iced = np.clip(iced, 0, 255).astype(np.uint8)

        # Frost speckle: sparse bright noise added on top.
        noise = np.random.rand(*frame.shape[:2]).astype(np.float32)
        frost = ((noise > 0.985) * 200).astype(np.uint8)
        return cv2.add(iced, cv2.cvtColor(frost, cv2.COLOR_GRAY2BGR))

    def _draw_facets(self, frame: np.ndarray, lm: Tracking) -> None:
        """Overlay faint crystalline lines: a Delaunay triangulation of the
        face landmarks (cv2.Subdiv2D), so facets follow the actual face shape.

        This replaces the old face-mesh tessellation constant, which the
        modern MediaPipe Tasks API no longer ships — and triangulating the real
        points looks just as crystalline with no extra dependency.
        """
        if lm.face is None:
            return
        h, w = frame.shape[:2]
        pts = lm.face[:, :2] * np.array([w, h])
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[inside]
        if len(pts) < 3:
            return

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for x, y in pts:
            subdiv.insert((float(x), float(y)))

        overlay = frame.copy()
        for t in subdiv.getTriangleList():
            tri = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            # Drop triangles touching Subdiv2D's outer bounding vertices.
            if any(not (0 <= x <= w and 0 <= y <= h) for x, y in tri):
                continue
            cv2.polylines(overlay, [np.array(tri, np.int32)], True,
                          (255, 240, 220), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        iced = self._iced(frame)
        alpha = as_alpha(mask, frame.shape)
        out = over(frame, iced, alpha)
        self._draw_facets(out, landmarks)  # facets only land on the (visible) face
        return out
