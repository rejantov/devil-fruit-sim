"""Gum-Gum no Mi — the rubber fruit (Phase 1).

Two interactions, both geometric warps via ``cv2.remap``:

1. **Arm stretch ("Gum-Gum Pistol").** Track an arm's elbow and wrist from
   Pose, then redraw the forearm region elongated along the elbow→wrist axis.
   The stretch amount grows as you straighten/extend the arm, so there's always
   something happening without needing a finicky trigger gesture.

2. **Cheek grab.** Pinch (thumb+index) near your face and the nearest cheek gets
   dragged toward the pinching hand, snapping back on release. The plan suggests
   a full per-triangle face-mesh affine warp here; this is the pragmatic first
   cut — a radial "liquify smudge" anchored to the face-mesh cheek point. It
   reads correctly and is a fraction of the code; the triangulated version is a
   clean later upgrade.

The hard part, as the plan warns, is stability rather than warp math: raw
landmarks jitter, so elbow/wrist/anchor points are all run through the shared
1€ smoother before they touch a warp.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from effects.base import BaseEffect
from utils import gesture
from utils.smoothing import LandmarkSmoother
from utils.tracking import (Tracking, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
                            FACE_CHEEK_LEFT, FACE_CHEEK_RIGHT)


class GumGum(BaseEffect):
    name = "Gum-Gum"
    swatch = (40, 200, 240)  # warm rubbery yellow (BGR)
    requires = {"pose", "hands", "face"}

    # Tunables.
    MAX_STRETCH = 2.4          # forearm length multiplier at full extension
    FOREARM_HALF_WIDTH = 0.30  # half-width of warped band, as fraction of forearm length
    CHEEK_RADIUS = 0.22        # grab radius as fraction of face width
    CHEEK_PULL = 0.9           # how much of the hand displacement to apply

    def __init__(self):
        self._smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.015)
        self._grid_cache: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None

    def reset(self) -> None:
        self._smoother.reset()

    # -- grid helper --------------------------------------------------------
    def _grid(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cached pixel-coordinate grids (xs, ys) for building remap fields."""
        if self._grid_cache is None or self._grid_cache[0] != w or self._grid_cache[1] != h:
            ys, xs = np.mgrid[0:h, 0:w]
            self._grid_cache = (w, h, xs.astype(np.float32), ys.astype(np.float32))
        return self._grid_cache[2], self._grid_cache[3]

    # -- arm stretch --------------------------------------------------------
    def _pick_arm(self, lm: Tracking) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (elbow_px, wrist_px) for the most-extended visible arm."""
        best = None
        best_len = 0.0
        for elbow_i, wrist_i in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
            if not (lm.pose_visible(elbow_i) and lm.pose_visible(wrist_i)):
                continue
            elbow = lm.pose_px(elbow_i)
            wrist = lm.pose_px(wrist_i)
            length = float(np.linalg.norm(wrist - elbow))
            if length > best_len:
                best_len, best = length, (elbow, wrist)
        return best

    def _stretch_arm(self, frame: np.ndarray, lm: Tracking, t: float) -> np.ndarray:
        arm = self._pick_arm(lm)
        if arm is None:
            return frame
        elbow, wrist = arm
        elbow = self._smoother.smooth("elbow", elbow, t)
        wrist = self._smoother.smooth("wrist", wrist, t)

        axis = wrist - elbow
        length = float(np.linalg.norm(axis))
        if length < 25.0:  # arm folded / barely visible → nothing to stretch
            return frame
        d = axis / length

        # Stretch grows with how extended the forearm is, giving a punchy
        # "reach" when you straighten the arm out toward the camera.
        stretch = 1.0 + (self.MAX_STRETCH - 1.0) * np.clip(length / (0.45 * lm.h), 0.0, 1.0)
        half_w = self.FOREARM_HALF_WIDTH * length

        xs, ys = self._grid(lm.w, lm.h)
        px = xs - elbow[0]
        py = ys - elbow[1]
        along = px * d[0] + py * d[1]               # distance along the limb axis
        perp = -px * d[1] + py * d[0]               # signed perpendicular distance

        # Output band runs from elbow out to the stretched wrist position. We
        # sample the source by compressing 'along' back by the stretch factor,
        # so the original forearm texture is spread over a longer span.
        band = (along >= 0.0) & (along <= length * stretch) & (np.abs(perp) <= half_w)
        src_along = along / stretch
        src_x = elbow[0] + d[0] * src_along - d[1] * perp
        src_y = elbow[1] + d[1] * src_along + d[0] * perp

        map_x = np.where(band, src_x, xs).astype(np.float32)
        map_y = np.where(band, src_y, ys).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # -- cheek grab ---------------------------------------------------------
    def _cheek_grab(self, frame: np.ndarray, lm: Tracking, t: float) -> np.ndarray:
        if lm.face is None or not lm.hands:
            return frame

        left_cheek = lm.face_px(FACE_CHEEK_LEFT)
        right_cheek = lm.face_px(FACE_CHEEK_RIGHT)
        face_w = float(np.linalg.norm(right_cheek - left_cheek)) + 1e-6
        radius = self.CHEEK_RADIUS * face_w

        # Find a pinching hand whose pinch point is actually on a cheek.
        for i, hand in enumerate(lm.hands):
            if not gesture.is_pinch(hand.landmarks, lm.w, lm.h):
                continue
            grab = gesture.pinch_point(hand.landmarks, lm.w, lm.h)
            anchor = (left_cheek if np.linalg.norm(grab - left_cheek)
                      < np.linalg.norm(grab - right_cheek) else right_cheek)
            if np.linalg.norm(grab - anchor) > radius * 1.6:
                continue  # pinch isn't near the face

            anchor = self._smoother.smooth(f"cheek{i}", anchor, t)
            pull = (grab - anchor) * self.CHEEK_PULL
            return self._radial_smudge(frame, lm, anchor, pull, radius)
        return frame

    def _radial_smudge(self, frame: np.ndarray, lm: Tracking, center: np.ndarray,
                       displacement: np.ndarray, radius: float) -> np.ndarray:
        """Pull a soft circular region by ``displacement`` (a liquify smudge)."""
        xs, ys = self._grid(lm.w, lm.h)
        dx = xs - center[0]
        dy = ys - center[1]
        dist = np.sqrt(dx * dx + dy * dy)
        # Smooth falloff: full pull at the centre, zero at the radius edge.
        falloff = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 2
        map_x = (xs - displacement[0] * falloff).astype(np.float32)
        map_y = (ys - displacement[1] * falloff).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # -- entry point --------------------------------------------------------
    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        frame = self._stretch_arm(frame, landmarks, t)
        frame = self._cheek_grab(frame, landmarks, t)
        return frame
