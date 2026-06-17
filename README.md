# Devil Fruit Sim

A local Python app that uses your webcam to simulate One Piece devil-fruit
powers in real time. MediaPipe tracks your hands/pose/face and segments your
body; OpenCV does the camera I/O, the on-screen UI, and the pixel warping;
NumPy does the array math. Pick a fruit from the on-screen button bar — no
keyboard needed (though number keys work too). Runs entirely on your machine.

## Fruits in v1

| Button | Fruit | What it does | Technique |
|--------|-------|--------------|-----------|
| 0 | **Off** | Raw webcam feed | — |
| 1 | **Gum-Gum** | Rubber: forearm stretches as you extend it; pinch your cheek to drag it | geometric warp (`cv2.remap`) |
| 2 | **Mera Mera** | Fire streaming off your body and hands | DOOM-style heat-grid fire sim |
| 3 | **Hie Hie** | Ice: faceted, frosty, blue, crystalline face lines | masked colour grade + face-mesh facets |
| 4 | **Suna Suna** | Sand: tan, granular, crumbling edges | masked sepia + grain |
| 5 | **Moku Moku** | Smoke: hazy body with drifting vapour | masked blur + scrolling fractal noise |

Fruits 2–5 use the body segmentation mask (masked colour/texture grading);
Gum-Gum uses landmark-driven geometric warping. The two categories are
deliberately different problems — see `effects/` for each.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requires a webcam. Tested on Python 3.12 / mediapipe 0.10.35. The MediaPipe
Tasks model files (~17 MB total) download automatically to `models/` the first
time you run it — no manual fetching.

## Controls

- **Click a button** on the bottom bar, or press its **number key** (`0`–`5`).
- `h` toggles the help overlay, `q` / `Esc` quits.

## Project layout

```
main.py              camera loop, effect dispatch, crossfade, input
effects/
  base.py            BaseEffect — the one method every fruit implements
  __init__.py        registry (build_effects) — add a fruit in one line
  none_effect.py     passthrough / default
  gum_gum.py         Phase 1: rubber stretch + cheek grab
  mera_mera.py       Phase 2: fire
  hie_hie.py         Phase 2: ice
  suna_suna.py       Phase 2: sand
  moku_moku.py       Phase 2: smoke
  _blend.py          shared mask-compositing helpers
ui/button_bar.py     fruit selector + mouse hit-testing
utils/
  tracking.py        MediaPipe wrapper — runs only the models the effect needs
  smoothing.py       landmark jitter filters (1€ filter, moving average)
  gesture.py         pinch / fist / pointing detection
assets/              (empty — v1 textures are all procedural)
```

## Adding a new fruit

1. Create `effects/your_fruit.py` with a `BaseEffect` subclass: set `name`,
   `swatch` (BGR colour for the button), and `requires` (any of
   `{"hands", "pose", "face", "mask"}`), then implement `process_frame`.
2. Add one line to `build_effects()` in `effects/__init__.py`.

That's it — `main.py` and the button bar pick it up automatically.

## Architecture notes

- **`requires` drives performance.** The tracker only runs the MediaPipe models
  the active effect asks for, which is the difference between ~10 fps (all four
  models) and ~30 fps (one or two). "Off" runs none.
- **Smoothing matters more than warp math.** Raw landmarks jitter; every
  geometric effect routes its key points through `utils/smoothing.py` first.
- **Masks are feathered** (`utils/tracking.py`) so composites don't look like a
  cardboard cutout pasted on the video.

## Roadmap

- **Phase 0** — scaffolding, tracking, button bar, registry ✅
- **Phase 1** — Gum-Gum rubber stretch + cheek grab ✅
- **Phase 2** — Logia set: Mera / Hie / Suna / Moku ✅
- **Phase 3** — crossfade on switch ✅; record demo clips; write the blog post

Stretch fruits if there's energy left: Magma-Magma, Goro Goro (lightning),
Dark-Dark. The cheek grab is currently a radial liquify smudge; the planned
upgrade is a per-triangle face-mesh affine warp.
