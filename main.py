"""Devil Fruit Cam — entry point.

Camera loop → active-effect dispatch → button bar overlay. The loop knows
nothing about individual fruits: it asks the registry for effects, asks the
active effect which trackers it needs, runs only those, and calls the effect.

Controls
--------
* Click a button, or press its number key (0–5), to switch fruit.
* ``h`` toggles the help overlay, ``q`` / ``Esc`` quits.
"""

from __future__ import annotations

import os

# Quiet MediaPipe/TF GL + absl log spam before those libs are imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from typing import Optional

import cv2
import numpy as np

from effects import build_effects
from ui.button_bar import ButtonBar
from utils.tracking import Tracker

WINDOW = "Devil Fruit Cam"
CAM_WIDTH, CAM_HEIGHT = 1280, 720
FADE_DURATION = 0.4  # seconds of crossfade when switching fruit (Phase 3 polish)


class App:
    def __init__(self):
        self.effects = build_effects()
        self.bar = ButtonBar(self.effects)
        self.tracker = Tracker()

        self.active = 0
        self.pending: Optional[int] = None   # set by mouse/keys, applied in-loop
        self.fade_from: Optional[int] = None
        self.fade_start = 0.0
        self.show_help = True
        self._fps = 0.0

    # -- input --------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.bar.hit(x, y)
            if idx is not None:
                self.pending = idx

    def _apply_pending(self, t: float) -> None:
        if self.pending is None or self.pending == self.active:
            self.pending = None
            return
        self.fade_from = self.active
        self.fade_start = t
        self.active = self.pending
        self.effects[self.active].reset()  # fresh state (e.g. fire cold-starts)
        self.pending = None

    # -- rendering ----------------------------------------------------------
    def _required_trackers(self) -> set:
        req = set(self.effects[self.active].requires)
        if self.fade_from is not None:
            req |= self.effects[self.fade_from].requires
        return req

    def _render(self, frame: np.ndarray, tracking, mask, t: float) -> np.ndarray:
        out = self.effects[self.active].process_frame(frame.copy(), tracking, mask, t)

        if self.fade_from is not None:
            elapsed = t - self.fade_start
            if elapsed >= FADE_DURATION:
                self.effects[self.fade_from].reset()
                self.fade_from = None
            else:
                prev = self.effects[self.fade_from].process_frame(frame.copy(), tracking, mask, t)
                alpha = elapsed / FADE_DURATION
                out = cv2.addWeighted(out, alpha, prev, 1.0 - alpha, 0)
        return out

    def _draw_hud(self, frame: np.ndarray) -> None:
        name = self.effects[self.active].name
        cv2.putText(frame, f"{name}   {self._fps:4.1f} fps", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{name}   {self._fps:4.1f} fps", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if self.show_help:
            cv2.putText(frame, "click a fruit or press 0-5  |  h: help  q: quit",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "click a fruit or press 0-5  |  h: help  q: quit",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        cap = _open_camera()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        start = time.monotonic()
        last = start
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed; stopping.")
                    break

                frame = cv2.flip(frame, 1)  # mirror, so it behaves like a mirror
                t = time.monotonic() - start

                self._apply_pending(t)
                tracking, mask = self.tracker.process(frame, self._required_trackers())
                out = self._render(frame, tracking, mask, t)

                self.bar.draw(out, self.active)
                self._draw_hud(out)

                now = time.monotonic()
                inst = 1.0 / max(now - last, 1e-3)
                self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst
                last = now

                cv2.imshow(WINDOW, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):            # q or Esc
                    break
                if key == ord("h"):
                    self.show_help = not self.show_help
                if ord("0") <= key <= ord("9"):
                    idx = key - ord("0")
                    if idx < len(self.effects):
                        self.pending = idx
                # Window closed via the [x] button.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()


def _open_camera() -> cv2.VideoCapture:
    """Open the first working webcam and request a sensible resolution."""
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            return cap
        cap.release()
    raise SystemExit("No webcam found (tried indices 0, 1, 2).")


if __name__ == "__main__":
    App().run()
