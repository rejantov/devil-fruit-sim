"""Hie Hie no Mi — ice fruit (Phase 2).

Body becomes solid faceted crystal: luminance is mapped to a three-stop ice
colour palette (dark-navy → icy-cyan → near-white), Voronoi-style crack lines
cover the whole body, and the face gets denser crystal edges along the actual
Face Mesh topology. Frost shimmer is very sparse — a glint here and there,
not a blizzard of white dots.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from effects._blend import as_alpha, over
from effects.base import BaseEffect
from utils.tracking import Tracking

_CRACK = (235, 248, 255)  # pale blue-white for body facet lines (BGR)
_FACET = (250, 252, 255)  # near-white for face mesh lines (BGR)


class HieHie(BaseEffect):
    name = "Hie Hie"
    swatch = (240, 200, 130)
    requires = {"mask", "face"}

    def __init__(self):
        super().__init__()
        self._body_tris: list | None = None
        self._body_shape: tuple | None = None

    # ------------------------------------------------------------------
    # Colour grade
    # ------------------------------------------------------------------

    def _crystallize(self, frame: np.ndarray) -> np.ndarray:
        """Map luminance to a three-stop ice palette (shadow→cyan→white).

        This replaces the old HSV-desaturation approach that made people look
        like ghosts being dispelled. The palette-remap produces solid ice/crystal
        rather than a washed-out filter, while keeping 15 % of the original so
        the person stays recognisable.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lum = gray[:, :, np.newaxis]  # HxWx1

        # BGR colour stops
        shadow  = np.array([ 90,  40,  15], np.float32)  # dark navy
        midtone = np.array([210, 170, 100], np.float32)   # icy blue-cyan
        hilight = np.array([255, 252, 245], np.float32)   # near-white

        pivot = 0.42
        t1 = np.clip(lum / pivot, 0.0, 1.0)
        t2 = np.clip((lum - pivot) / (1.0 - pivot), 0.0, 1.0)

        ice = np.where(
            lum < pivot,
            shadow  * (1.0 - t1) + midtone * t1,
            midtone * (1.0 - t2) + hilight * t2,
        )

        # 15 % original keeps facial features readable through the crystal.
        ice = 0.85 * ice + 0.15 * frame.astype(np.float32)
        return np.clip(ice, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Body facets
    # ------------------------------------------------------------------

    def _ensure_body_tris(self, h: int, w: int) -> list:
        """Stable jittered-grid Delaunay triangulation (built once per resolution).

        Fixed RNG seed → facets don't swim around between frames.
        ~55 px grid gives roughly 130 seed points on a 640×480 frame.
        """
        if self._body_tris is not None and self._body_shape == (h, w):
            return self._body_tris

        rng  = np.random.default_rng(7)
        cell = 55
        pts: list[tuple[float, float]] = []
        for y in range(0, h + cell, cell):
            for x in range(0, w + cell, cell):
                jx = int(np.clip(x + rng.integers(-cell // 3, cell // 3 + 1), 0, w - 1))
                jy = int(np.clip(y + rng.integers(-cell // 3, cell // 3 + 1), 0, h - 1))
                pts.append((float(jx), float(jy)))

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for p in pts:
            subdiv.insert(p)

        tris = []
        for tri_data in subdiv.getTriangleList():
            xs = [tri_data[0], tri_data[2], tri_data[4]]
            ys = [tri_data[1], tri_data[3], tri_data[5]]
            if min(xs) >= 0 and max(xs) <= w and min(ys) >= 0 and max(ys) <= h:
                tris.append(
                    np.array([(xs[0], ys[0]), (xs[1], ys[1]), (xs[2], ys[2])], np.int32)
                )

        self._body_tris = tris
        self._body_shape = (h, w)
        return tris

    def _draw_body_facets(self, iced: np.ndarray) -> np.ndarray:
        """Pale crystal-crack lines drawn on the ice layer at 35 % opacity.

        The body mask is applied later in the composite step, so lines only
        appear on the body without any per-triangle mask check here.
        """
        h, w = iced.shape[:2]
        overlay = iced.copy()
        for tri in self._ensure_body_tris(h, w):
            cv2.polylines(overlay, [tri], True, _CRACK, 1, cv2.LINE_AA)
        return cv2.addWeighted(iced, 0.65, overlay, 0.35, 0)

    # ------------------------------------------------------------------
    # Face facets
    # ------------------------------------------------------------------

    def _draw_face_facets(self, frame: np.ndarray, lm: Tracking) -> None:
        """Crystal lines along the Face Mesh triangulation, in-place on frame."""
        if lm.face is None:
            return
        h, w = frame.shape[:2]
        pts   = lm.face[:, :2] * np.array([w, h])
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts   = pts[valid]
        if len(pts) < 3:
            return

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for x, y in pts:
            subdiv.insert((float(x), float(y)))

        overlay = frame.copy()
        for tri_data in subdiv.getTriangleList():
            tri = [(tri_data[0], tri_data[1]), (tri_data[2], tri_data[3]), (tri_data[4], tri_data[5])]
            if any(not (0 <= x <= w and 0 <= y <= h) for x, y in tri):
                continue
            cv2.polylines(overlay, [np.array(tri, np.int32)], True, _FACET, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, dst=frame)

    # ------------------------------------------------------------------
    # Frost shimmer
    # ------------------------------------------------------------------

    def _frost_shimmer(self, frame: np.ndarray, alpha: np.ndarray, t: float) -> np.ndarray:
        """Very sparse animated glints — ~0.2 % of body pixels.

        Old threshold was 0.985 (~1.5 % of pixels = blizzard of white dots).
        0.998 drops that to ~0.2 %, giving occasional glints instead.
        """
        h, w = frame.shape[:2]
        rng   = np.random.default_rng(int(t * 15) % 50000)
        noise = rng.random((h, w)).astype(np.float32)
        hits  = (noise > 0.998).astype(np.float32) * alpha[:, :, 0]
        return np.clip(
            frame.astype(np.float32) + hits[:, :, np.newaxis] * 210,
            0, 255,
        ).astype(np.uint8)

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
        alpha = as_alpha(mask, frame.shape)      # float32 HxWx1 body mask

        # 1. Map whole frame to ice palette.
        iced = self._crystallize(frame)

        # 2. Stamp body-wide crystal facet lines onto the ice layer.
        iced = self._draw_body_facets(iced)

        # 3. Composite: show the ice layer only where the body mask is active.
        out = over(frame, iced, alpha)

        # 4. Face mesh crystal lines — denser, anatomically accurate.
        self._draw_face_facets(out, landmarks)

        # 5. Sparse body-masked shimmer glints.
        out = self._frost_shimmer(out, alpha, t)

        return out
