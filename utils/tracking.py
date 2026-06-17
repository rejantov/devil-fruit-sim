"""MediaPipe wrapper: one call per frame, only the models the effect needs.

``main.py`` doesn't talk to MediaPipe directly. It asks the active effect what
it needs (``effect.requires``) and hands that set to :meth:`Tracker.process`,
which runs *only* those models and returns a tidy :class:`Tracking` snapshot
plus a feathered body mask.

Why only-what-you-need: running Hands + Pose + Face + Segmentation every frame
is the difference between ~10 fps and ~30 fps. Most effects need one or two of
them, so we lazily build models on first request and skip the rest.

This uses MediaPipe's current **Tasks** API (the legacy ``solutions`` API was
removed in mediapipe 0.10.x). The Tasks API needs ``.task``/``.tflite`` model
files; we auto-download them to ``models/`` on first use, so a fresh clone just
works with no manual steps.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Set

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Pose landmark indices (MediaPipe Pose, 33 points).
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# Rough cheek anchors in the 478-point face mesh (left/right outer cheek).
FACE_CHEEK_LEFT = 234
FACE_CHEEK_RIGHT = 454

VALID_REQUIRES = {"hands", "pose", "face", "mask"}

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODELS = {
    "hands": ("hand_landmarker.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "face": ("face_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "mask": ("selfie_segmenter.tflite",
             "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
}


def _model_path(key: str) -> str:
    """Path to a model file, downloading it on first use."""
    name, url = _MODELS[key]
    path = os.path.join(_MODEL_DIR, name)
    if not os.path.exists(path):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"Downloading {name} (first run only)...")
        urllib.request.urlretrieve(url, path)
    return path


@dataclass
class HandTrack:
    """One detected hand: normalised landmarks plus its left/right label."""

    landmarks: np.ndarray            # (21, 3), x/y in [0,1], z relative
    handedness: str                  # "Left" or "Right"
    score: float


@dataclass
class Tracking:
    """Everything the effects might read for the current frame.

    Coordinates are stored normalised (0..1) exactly as MediaPipe returns them;
    use :meth:`px` (or the ``_px`` helpers) to get pixel coordinates. Any field
    not requested by the active effect stays empty/None.
    """

    w: int
    h: int
    hands: List[HandTrack] = field(default_factory=list)
    pose: Optional[np.ndarray] = None   # (33, 4): x, y, z, visibility
    face: Optional[np.ndarray] = None   # (N, 3)

    def px(self, norm_xy) -> np.ndarray:
        """Normalised (x, y) → pixel (x, y) as float array."""
        return np.array([norm_xy[0] * self.w, norm_xy[1] * self.h], dtype=np.float64)

    def pose_px(self, idx: int) -> Optional[np.ndarray]:
        if self.pose is None:
            return None
        return self.px(self.pose[idx])

    def pose_visible(self, idx: int, threshold: float = 0.5) -> bool:
        return self.pose is not None and self.pose[idx, 3] >= threshold

    def face_px(self, idx: int) -> Optional[np.ndarray]:
        if self.face is None:
            return None
        return self.px(self.face[idx])


def _to_xyz(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)


def _to_xyzv(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float64)


class Tracker:
    """Lazily-built MediaPipe Tasks models with a single per-frame entry point."""

    def __init__(self, mask_feather: int = 15):
        self._hands = None
        self._pose = None
        self._face = None
        self._segmenter = None
        self._mask_feather = mask_feather  # odd kernel size for edge feathering
        self._t0 = time.monotonic()
        self._last_ts = -1                 # strictly-increasing ms for VIDEO mode

    # -- lazy model builders ------------------------------------------------
    def _base(self, key: str):
        return mp_python.BaseOptions(model_asset_path=_model_path(key))

    def _ensure_hands(self):
        if self._hands is None:
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=self._base("hands"),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=0.6, min_tracking_confidence=0.5))
        return self._hands

    def _ensure_pose(self):
        if self._pose is None:
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=self._base("pose"),
                    running_mode=vision.RunningMode.VIDEO, num_poses=1,
                    min_pose_detection_confidence=0.6, min_tracking_confidence=0.5))
        return self._pose

    def _ensure_face(self):
        if self._face is None:
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=self._base("face"),
                    running_mode=vision.RunningMode.VIDEO, num_faces=1))
        return self._face

    def _ensure_segmenter(self):
        if self._segmenter is None:
            self._segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=self._base("mask"),
                    running_mode=vision.RunningMode.VIDEO,
                    output_confidence_masks=True, output_category_mask=False))
        return self._segmenter

    def _next_ts(self) -> int:
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
        self._last_ts = ts
        return ts

    # -- per-frame entry point ---------------------------------------------
    def process(self, frame_bgr: np.ndarray, requires: Set[str]):
        """Run the requested models on a BGR frame.

        Returns ``(Tracking, mask)`` where ``mask`` is a float32 HxW array in
        [0, 1] with feathered edges (or ``None`` if not requested).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._next_ts()

        tracking = Tracking(w=w, h=h)
        mask: Optional[np.ndarray] = None

        if "hands" in requires:
            res = self._ensure_hands().detect_for_video(mp_img, ts)
            for i, lms in enumerate(res.hand_landmarks):
                label, score = "Right", 1.0
                if i < len(res.handedness) and res.handedness[i]:
                    label = res.handedness[i][0].category_name
                    score = res.handedness[i][0].score
                tracking.hands.append(HandTrack(_to_xyz(lms), label, score))

        if "pose" in requires:
            res = self._ensure_pose().detect_for_video(mp_img, ts)
            if res.pose_landmarks:
                tracking.pose = _to_xyzv(res.pose_landmarks[0])

        if "face" in requires:
            res = self._ensure_face().detect_for_video(mp_img, ts)
            if res.face_landmarks:
                tracking.face = _to_xyz(res.face_landmarks[0])

        if "mask" in requires:
            res = self._ensure_segmenter().segment_for_video(mp_img, ts)
            if res.confidence_masks:
                raw = res.confidence_masks[0].numpy_view()  # (H, W, 1) float32, foreground
                raw = np.ascontiguousarray(raw).reshape(h, w).astype(np.float32)
                k = self._mask_feather
                # Feather the edge so composites don't look like a cardboard
                # cutout pasted on the video (plan calls this out explicitly).
                mask = np.clip(cv2.GaussianBlur(raw, (k, k), 0), 0.0, 1.0)

        return tracking, mask

    def close(self) -> None:
        for model in (self._hands, self._pose, self._face, self._segmenter):
            if model is not None:
                model.close()
