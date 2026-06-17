"""Gum-Gum no Mi — the rubber fruit (Phase 1).

Two interactions, both geometric warps via ``cv2.remap``:

1. **Arm + hand + finger stretch.**
   Phase A (one remap): the forearm is stretched from elbow to the new wrist
   position, and the whole hand is translated to follow the stretched wrist so
   it is never covered.  Zone A = forearm (compressed source along arm axis);
   Zone B = hand (identity shifted by the arm-stretch displacement).

   Phase B (one more remap, applied to Phase A result): five per-finger bands,
   each centred on its own MCP→TIP skeleton axis (from MediaPipe hand
   landmarks), so every finger stretches independently along its own direction
   like a rubber finger — not one big block.

2. **Cheek grab.**  Pinch (thumb+index) near the face → nearest cheek is
   dragged toward the hand as a radial liquify smudge.
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

# (MCP landmark index, TIP landmark index) for each finger
_FINGER_PAIRS = [
    (2, 4),    # thumb: MCP → tip
    (5, 8),    # index
    (9, 12),   # middle
    (13, 16),  # ring
    (17, 20),  # pinky
]


class GumGum(BaseEffect):
    name = "Gum-Gum"
    swatch = (40, 200, 240)  # rubbery yellow-cyan (BGR)
    requires = {"pose", "hands", "face"}

    MAX_STRETCH          = 2.4   # forearm-length multiplier at full extension
    FOREARM_HALF_WIDTH   = 0.25  # forearm band half-width as fraction of forearm len
    HAND_HALF_W_FACTOR   = 1.8   # hand zone half-width relative to forearm half-width
    HAND_ZONE_LEN_FACTOR = 0.85  # how far past stretched wrist the hand zone extends
    FINGER_HALF_W_FACTOR = 0.35  # per-finger band half-width as fraction of finger len
    CHEEK_RADIUS         = 0.22  # grab radius as fraction of face width
    CHEEK_PULL           = 0.9   # fraction of hand displacement applied to cheek

    def __init__(self):
        super().__init__()
        self._smoother   = LandmarkSmoother(min_cutoff=1.0, beta=0.015)
        self._grid_cache: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None

    def reset(self) -> None:
        self._smoother.reset()

    # ------------------------------------------------------------------
    # Grid helper
    # ------------------------------------------------------------------

    def _grid(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cached pixel-coordinate grids (xs, ys) for building remap fields."""
        if (self._grid_cache is None
                or self._grid_cache[0] != w
                or self._grid_cache[1] != h):
            ys, xs = np.mgrid[0:h, 0:w]
            self._grid_cache = (w, h, xs.astype(np.float32), ys.astype(np.float32))
        return self._grid_cache[2], self._grid_cache[3]

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def _pick_arm(self, lm: Tracking) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (elbow_px, wrist_px) for the most-extended visible arm."""
        best: Optional[Tuple[np.ndarray, np.ndarray]] = None
        best_len = 0.0
        for elbow_i, wrist_i in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
            if not (lm.pose_visible(elbow_i) and lm.pose_visible(wrist_i)):
                continue
            elbow = lm.pose_px(elbow_i)
            wrist = lm.pose_px(wrist_i)
            if elbow is None or wrist is None:
                continue
            length = float(np.linalg.norm(wrist - elbow))
            if length > best_len:
                best_len, best = length, (elbow, wrist)
        return best

    # ------------------------------------------------------------------
    # Arm stretch — Phase A: forearm stretch + hand translation
    # ------------------------------------------------------------------

    def _phase_a(self, frame: np.ndarray, elbow: np.ndarray, wrist: np.ndarray,
                 fore_len: float, d: np.ndarray, stretch: float,
                 xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Stretch the forearm and translate the hand to the new wrist position.

        Returns (remapped_frame, shift_x, shift_y) where shift_{x,y} is how
        far the hand moved (needed by Phase B to find the new finger positions).
        """
        arm_half_w  = self.FOREARM_HALF_WIDTH * fore_len
        hand_half_w = arm_half_w * self.HAND_HALF_W_FACTOR
        arm_shift   = (stretch - 1.0) * fore_len  # distance wrist moved along d
        shift_x     = float(arm_shift * d[0])
        shift_y     = float(arm_shift * d[1])

        # Pixel coords relative to elbow, in arm's local frame.
        ex, ey = float(elbow[0]), float(elbow[1])
        px    = xs - ex
        py    = ys - ey
        along = px * d[0] + py * d[1]   # component along arm axis
        perp  = -px * d[1] + py * d[0]  # perpendicular component

        # Zone A — forearm: pixels in [0, fore_len*stretch] along the arm axis
        # sample from compressed source [0, fore_len].
        forearm_band = ((along >= 0.0)
                        & (along <= fore_len * stretch)
                        & (np.abs(perp) <= arm_half_w))
        src_along = along / stretch
        src_x_fa  = ex + d[0] * src_along - d[1] * perp
        src_y_fa  = ey + d[1] * src_along + d[0] * perp

        # Zone B — hand: pixels just past the stretched wrist are shifted back
        # to sample from the original hand position in the source.
        hand_zone = ((along > fore_len * stretch)
                     & (along <= fore_len * stretch + fore_len * self.HAND_ZONE_LEN_FACTOR)
                     & (np.abs(perp) <= hand_half_w))
        src_x_h = xs - shift_x
        src_y_h = ys - shift_y

        map_x = np.where(forearm_band, src_x_fa,
                         np.where(hand_zone, src_x_h, xs)).astype(np.float32)
        map_y = np.where(forearm_band, src_y_fa,
                         np.where(hand_zone, src_y_h, ys)).astype(np.float32)

        result = cv2.remap(frame, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return result, shift_x, shift_y

    # ------------------------------------------------------------------
    # Arm stretch — Phase B: per-finger stretches on the Phase A result
    # ------------------------------------------------------------------

    def _phase_b(self, step1: np.ndarray, matched_hand,
                 wrist: np.ndarray, lm: Tracking,
                 shift_x: float, shift_y: float,
                 stretch: float, t: float,
                 xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Apply one rectangular stretch band per finger using hand skeleton.

        Each band is aligned to the MCP→TIP axis of that finger, identical
        in principle to the forearm band but in each finger's own direction.
        The hand is already translated by (shift_x, shift_y) in step1, so
        the new MCP/TIP positions are simply the original ones shifted.
        """
        shift_vec = np.array([shift_x, shift_y], np.float64)

        # Start with identity: every pixel maps to itself.
        map_x2 = xs.copy()
        map_y2 = ys.copy()

        for base_i, tip_i in _FINGER_PAIRS:
            orig_base = np.array([
                matched_hand.landmarks[base_i, 0] * lm.w,
                matched_hand.landmarks[base_i, 1] * lm.h,
            ], np.float64)
            orig_tip = np.array([
                matched_hand.landmarks[tip_i, 0] * lm.w,
                matched_hand.landmarks[tip_i, 1] * lm.h,
            ], np.float64)
            orig_base = self._smoother.smooth(f"fb{base_i}", orig_base, t)
            orig_tip  = self._smoother.smooth(f"ft{tip_i}",  orig_tip,  t)

            # In step1 the hand is at shifted positions.
            new_base = orig_base + shift_vec
            new_tip  = orig_tip  + shift_vec

            fi_vec = new_tip - new_base
            fi_len = float(np.linalg.norm(fi_vec))
            if fi_len < 5.0:
                continue

            # Finger-local axis unit vectors.
            d_fi      = fi_vec / fi_len
            half_w_fi = fi_len * self.FINGER_HALF_W_FACTOR

            # Pixel coords relative to this finger's base in step1.
            bx, by  = float(new_base[0]), float(new_base[1])
            px_fi   = xs - bx
            py_fi   = ys - by
            along_fi = px_fi * d_fi[0] + py_fi * d_fi[1]
            perp_fi  = -px_fi * d_fi[1] + py_fi * d_fi[0]

            # Output band stretches to fi_len * stretch; source is clipped to fi_len.
            finger_band = ((along_fi >= 0.0)
                           & (along_fi <= fi_len * stretch)
                           & (np.abs(perp_fi) <= half_w_fi))
            src_along_fi = np.clip(along_fi / stretch, 0.0, fi_len)
            src_x_fi = bx + d_fi[0] * src_along_fi - d_fi[1] * perp_fi
            src_y_fi = by + d_fi[1] * src_along_fi + d_fi[0] * perp_fi

            # Later fingers in the list overwrite earlier ones where bands overlap
            # (overlap is minor, near the palm, so ordering doesn't matter much).
            map_x2 = np.where(finger_band, src_x_fi, map_x2)
            map_y2 = np.where(finger_band, src_y_fi, map_y2)

        return cv2.remap(step1,
                         map_x2.astype(np.float32),
                         map_y2.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------

    def _stretch_arm(self, frame: np.ndarray, lm: Tracking, t: float) -> np.ndarray:
        arm = self._pick_arm(lm)
        if arm is None:
            return frame
        elbow, wrist = arm
        elbow = self._smoother.smooth("elbow", elbow, t)
        wrist = self._smoother.smooth("wrist", wrist, t)

        axis     = wrist - elbow
        fore_len = float(np.linalg.norm(axis))
        if fore_len < 20.0:
            return frame

        d = axis / fore_len  # unit vector elbow → wrist

        stretch = (1.0 + (self.MAX_STRETCH - 1.0)
                   * float(np.clip(fore_len / (0.45 * lm.h), 0.0, 1.0)))

        h, w   = frame.shape[:2]
        xs, ys = self._grid(w, h)

        # Phase A: forearm stretch + hand translation in one remap.
        step1, shift_x, shift_y = self._phase_a(
            frame, elbow, wrist, fore_len, d, stretch, xs, ys)

        # Find the hand whose wrist landmark is closest to the pose wrist.
        matched_hand = None
        for hand in lm.hands:
            hw = np.array([hand.landmarks[0, 0] * lm.w,
                           hand.landmarks[0, 1] * lm.h], np.float64)
            if float(np.linalg.norm(hw - wrist)) < fore_len * 0.55:
                matched_hand = hand
                break

        if matched_hand is None:
            return step1

        # Phase B: per-finger stretches on the Phase A output.
        return self._phase_b(step1, matched_hand, wrist, lm,
                             shift_x, shift_y, stretch, t, xs, ys)

    # ------------------------------------------------------------------
    # Cheek grab
    # ------------------------------------------------------------------

    def _cheek_grab(self, frame: np.ndarray, lm: Tracking, t: float) -> np.ndarray:
        if lm.face is None or not lm.hands:
            return frame

        left_cheek  = lm.face_px(FACE_CHEEK_LEFT)
        right_cheek = lm.face_px(FACE_CHEEK_RIGHT)
        if left_cheek is None or right_cheek is None:
            return frame

        face_w = float(np.linalg.norm(right_cheek - left_cheek)) + 1e-6
        radius = self.CHEEK_RADIUS * face_w

        for i, hand in enumerate(lm.hands):
            if not gesture.is_pinch(hand.landmarks, lm.w, lm.h):
                continue
            grab   = gesture.pinch_point(hand.landmarks, lm.w, lm.h)
            anchor = (left_cheek
                      if np.linalg.norm(grab - left_cheek) < np.linalg.norm(grab - right_cheek)
                      else right_cheek)
            if float(np.linalg.norm(grab - anchor)) > radius * 1.6:
                continue

            anchor = self._smoother.smooth(f"cheek{i}", anchor, t)
            pull   = (grab - anchor) * self.CHEEK_PULL
            return self._radial_smudge(frame, lm, anchor, pull, radius)
        return frame

    def _radial_smudge(self, frame: np.ndarray, lm: Tracking,
                       center: np.ndarray, displacement: np.ndarray,
                       radius: float) -> np.ndarray:
        """Pull a soft circular region by *displacement* (liquify smudge)."""
        xs, ys  = self._grid(lm.w, lm.h)
        dx      = xs - float(center[0])
        dy      = ys - float(center[1])
        dist    = np.sqrt(dx * dx + dy * dy)
        falloff = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 2
        map_x   = (xs - float(displacement[0]) * falloff).astype(np.float32)
        map_y   = (ys - float(displacement[1]) * falloff).astype(np.float32)
        return cv2.remap(frame, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, landmarks: Tracking,
                      mask: Optional[np.ndarray], t: float) -> np.ndarray:
        frame = self._stretch_arm(frame, landmarks, t)
        frame = self._cheek_grab(frame, landmarks, t)
        return frame
