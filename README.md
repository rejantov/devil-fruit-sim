# Devil Fruit Sim

A local Python app that uses your webcam to simulate One Piece devil-fruit
powers in real time. MediaPipe tracks your hands/pose/face and segments your
body; OpenCV does the camera I/O, the on-screen UI, and the pixel warping;
NumPy does the array math. Pick a fruit from the on-screen button bar — no
keyboard needed (though number keys work too). Runs entirely on your machine,
nothing is sent to any server.

## Fruits

| Button | Fruit | What it does | Technique |
|--------|-------|--------------|-----------|
| 0 | **Off** | Raw webcam feed | — |
| 1 | **Gum-Gum** | Rubber: forearm stretches as you extend it; each finger has its own stretch band; pinch a cheek to drag it | geometric warp (`cv2.remap`), hand skeleton landmarks |
| 2 | **Mera Mera** | Fire streaming off your body and hands; fast movement leaves a blazing afterimage | DOOM-style heat-grid fire sim + frame-diff motion detection |
| 3 | **Hie Hie** | Ice: solid faceted crystal skin, Voronoi crack lines over the body, denser face-mesh lines on the face | luminance palette remap + Delaunay facets |
| 4 | **Suna Suna** | Sand: warm granular skin, sand grains fall from your body under gravity, raise your arm to stream sand from your wrist | masked colour grade + particle system |
| 5 | **Moku Moku** | Smoke: ghost-dim spectral form with animated smoke ring rising from the body outline | dual-layer fractal noise + sigmoid gate + ghost composite |

---

## Setup — Linux / macOS

```bash
git clone <repo-url>
cd devil-fruit-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The MediaPipe Tasks model files (~17 MB total) download automatically to
`models/` the first time you run it — you do not need to fetch them by hand.

---

## Setup — Windows

### 1. Install Python

Download **Python 3.11** or **3.12** from [python.org/downloads](https://www.python.org/downloads/).

During installation, **check the box "Add Python to PATH"** — if you miss this
the `python` command will not be found in the terminal.

### 2. Open a terminal

- Press **Win + S**, type **PowerShell**, and open it.
  Or use **Command Prompt** (`cmd`) — both work.

### 3. Clone / download the repo

If you have Git installed:
```powershell
git clone <repo-url>
cd devil-fruit-sim
```

Otherwise download the ZIP from GitHub, extract it, and `cd` into the folder:
```powershell
cd C:\Users\YourName\Downloads\devil-fruit-sim
```

### 4. Create a virtual environment

```powershell
python -m venv venv
```

### 5. Activate the virtual environment

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

If PowerShell shows a red error about scripts being disabled, run this **once**
and then try again:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Command Prompt (cmd):**
```cmd
venv\Scripts\activate.bat
```

When the venv is active you will see `(venv)` at the start of the prompt.

### 6. Install dependencies

```powershell
pip install -r requirements.txt
```

### 7. Run

```powershell
python main.py
```

A window titled **Devil Fruit Sim** will open showing your webcam feed with
the button bar at the bottom. If the window does not appear, see
[Troubleshooting](#troubleshooting) below.

### 8. Quit

Press **q** or **Esc** inside the window, or close it with the X button.

---

## Controls

| Input | Action |
|-------|--------|
| Click a button | Switch fruit |
| Number keys `0`–`5` | Switch fruit |
| `h` | Toggle help overlay |
| `q` / `Esc` | Quit |

---

## Troubleshooting

**"python is not recognised" (Windows)**
You did not check "Add Python to PATH" during install. Re-run the Python
installer, choose **Modify**, and enable **Add Python to environment variables**,
or add `C:\Users\YourName\AppData\Local\Programs\Python\Python312` to your
PATH manually.

**"No module named cv2 / mediapipe / numpy"**
Your terminal is using the system Python instead of the venv. Make sure
`(venv)` appears in your prompt. If not, activate the venv again (Step 5).
In VS Code, press `Ctrl+Shift+P` → **Python: Select Interpreter** and pick
`./venv/Scripts/python.exe`.

**Camera does not open / black screen (Windows)**
- Open **Settings → Privacy & security → Camera** and make sure **Camera access**
  and access for **Desktop apps** are both enabled.
- If another app (Teams, Zoom, browser) is using the camera, close it first.
- Try a different camera index: open `main.py` and change `cv2.VideoCapture(0)`
  to `cv2.VideoCapture(1)`.

**Window appears then immediately closes**
Run from the terminal rather than by double-clicking so you can see the error
message printed before the crash.

**Very low frame rate**
Each fruit only runs the MediaPipe models it needs — **Gum-Gum** is the
heaviest (pose + hands + face). Close other apps, especially browser tabs with
video. On a laptop, plug into power so the CPU does not throttle.

**"Script cannot be loaded" error in PowerShell**
Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once
(Step 5 above).

---

## Project layout

```
main.py              camera loop, effect dispatch, crossfade, input
effects/
  base.py            BaseEffect — the one method every fruit implements
  __init__.py        registry (build_effects) — add a fruit in one line
  none_effect.py     passthrough / default
  gum_gum.py         rubber stretch (arm + hand translation + per-finger bands)
  mera_mera.py       fire (heat grid + motion blazing)
  hie_hie.py         ice (faceted crystal + face mesh lines)
  suna_suna.py       sand (colour grade + falling particle system)
  moku_moku.py       smoke (ghost form + dual-layer fractal noise)
  _blend.py          shared mask-compositing helpers (over, screen)
ui/button_bar.py     fruit selector + mouse hit-testing
utils/
  tracking.py        MediaPipe wrapper — runs only the models the effect needs
  smoothing.py       landmark jitter filters (1€ filter)
  gesture.py         pinch / fist / pointing detection
models/              auto-downloaded .task / .tflite files (first run only)
assets/              (empty — all textures are procedural)
```

---

## Adding a new fruit

1. Create `effects/your_fruit.py` with a `BaseEffect` subclass.  Set `name`,
   `swatch` (BGR colour for the button), and `requires` (any subset of
   `{"hands", "pose", "face", "mask"}`), then implement `process_frame`.
2. Add one line to `build_effects()` in `effects/__init__.py`.

That's it — `main.py` and the button bar pick it up automatically.

---

## Architecture notes

- **`requires` drives performance.** The tracker only runs the MediaPipe models
  the active effect asks for — the difference between ~10 fps (all four models)
  and ~30 fps (one or two). "Off" runs none.
- **Smoothing matters more than warp math.** Raw landmarks jitter badly; every
  geometric effect routes its key points through a 1€ filter in
  `utils/smoothing.py` before doing any warping.
- **Masks are feathered** (`utils/tracking.py`) so composites don't look like a
  cardboard cutout pasted on the video.
- **Gum-Gum uses two remap passes.** Pass A stretches the forearm and
  translates the whole hand to the new wrist position in one `cv2.remap`.
  Pass B applies five independent per-finger rectangular bands (using the 21
  MediaPipe hand skeleton landmarks) so each finger stretches along its own
  axis, not along the arm axis.

---

## Roadmap

- **Phase 0** — scaffolding, tracking, button bar, registry ✅
- **Phase 1** — Gum-Gum rubber stretch + cheek grab ✅
- **Phase 2** — Logia set: Mera / Hie / Suna / Moku ✅
- **Phase 3** — crossfade on switch ✅; record demo clips; write the blog post

Stretch fruits if there's energy left: Magma-Magma, Goro Goro (lightning),
Dark-Dark. The cheek grab is currently a radial liquify smudge; the planned
upgrade is a per-triangle face-mesh affine warp.
