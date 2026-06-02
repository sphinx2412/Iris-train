# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Data-prep tooling + a training runbook for a **YOLO11 detector** that finds **iris** and
**pupil** circles `(cx, cy, r)` in eye photos — meant to replace a slow SAM-based pipeline with
millisecond inference. **No training has been run yet**; this repo is scaffolding + scripts.
`README.md` is the English runbook.

## Pipeline (the whole project is these six ordered steps)

Numbered `*.bat` files wrap each step (they `cd` to repo root and activate `.venv`); the
underlying Python is below. Steps 1–3 are CPU-only data prep; 4–6 need a GPU + `ultralytics`.

```powershell
python tools/json_to_yolo.py            # 1. jsonOutputs/*.json  -> labels/<stem>.txt
python tools/visualize_labels.py --n 5  # 2. sanity-check: draw labels back onto photos -> viz/
python tools/make_split.py              # 3. contiguous-by-id split -> dataset/ + data.yaml
yolo detect train model=yolo11m.pt data=dataset/data.yaml imgsz=1280 epochs=100 batch=8 patience=20 device=0
yolo detect val   model=runs/detect/train/weights/best.pt data=dataset/data.yaml split=test
yolo export       model=runs/detect/train/weights/best.pt format=onnx imgsz=1280
```

`labels/`, `dataset/`, `runs/`, `viz/`, `*.pt`, `*.onnx` are all generated and gitignored.
There is no build, lint, or test suite — verification is the visual/metric checks below.

## Setup

`0_setup_install.bat` creates `.venv` (if missing) and installs everything. To do it by
hand, install CUDA torch **first**, then the rest:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Data-prep only (no GPU): `pip install numpy opencv-python pyyaml`.

## Architecture / things that aren't obvious from a single file

- **Circle ↔ bbox is the central trick.** Manual annotations are circles; YOLO needs boxes.
  `circle_to_yolo_line` (`tools/json_to_yolo.py`) maps a circle to `cx, cy, w=h=2r`; at
  inference you invert it as `r = (w+h)/4`. iris=class 0, pupil=class 1; they overlap
  concentrically and YOLO handles that fine.
- **Coordinates are normalized against `source.width/height` from the JSON, never the image
  file's pixels.** `json_to_yolo.py` never decodes the image — it only checks the file exists.
  Keep this invariant; reading dimensions from the image would silently break alignment.
- **Origin is the top-left corner; `y` grows downward** (image convention). `center_y/H` is a
  plain divide, never `H - center_y` — same as YOLO and OpenCV, so annotation → YOLO label →
  viz overlay all share one coordinate frame and no axis flip exists anywhere. This holds only
  because the JSON's `center_x/center_y` live in the same frame as `source.width/height`; if an
  annotation ever measured `y` from the bottom it would fail silently (circles drift vertically,
  no error) — the step-2 visual check is the only guard.
- **"No eye" frames (`detection.success == false`) become empty `.txt` files** — YOLO reads
  these as background/negative examples. This is intentional; don't filter them out. `make_split.py`
  even checks the test split contains some (`is_no_eye`).
- **The train/val/test split is contiguous-by-id with a dropped buffer — NOT random, NOT
  by date** (`tools/make_split.py`, `split_indices`). Photos were shot over ~4 days, so the same
  person recurs only among neighboring numeric ids; the split sorts by id (`numeric_id` = last
  digit run in the stem), cuts contiguous blocks, and drops `--buffer` ids (default 4) on each
  side of every junction so one person can't leak across splits. **The `test` block is frozen** —
  do not re-run the split once training has started. If a person might span more than 4 ids at a
  seam, raise `--buffer`; do **not** delete photos (that wastes data and doesn't fix leakage —
  it's a detector, not a recognizer, so same-person repeats are useful training variety).
- **Counts are always computed from the files at runtime — never hardcode a total** (e.g. the
  oft-cited ~1600 is only a target; annotation is ongoing).
- **All image I/O is unicode-safe** via `np.fromfile` + `cv2.imdecode` / `cv2.imencode` +
  `.tofile` (see `imread_unicode` in `tools/visualize_labels.py`). Use this pattern, not
  `cv2.imread`/`imwrite`, for Windows paths with non-ASCII characters.

## Annotation JSON schema (input in `jsonOutputs/`)

Each photo has `<stem>.<ext>` (image), `<stem>.json` (label), and an ignored
`_debug_<stem>.jpg`. Schema (`schema_version: 1`):

```json
{ "source": {"filename": "x.jpg", "width": 5760, "height": 3240},
  "detection": {"success": true, "tier": "manual"},
  "iris":  {"center_x": 1234, "center_y": 988, "radius": 210},
  "pupil": {"center_x": 1240, "center_y": 990, "radius": 32} }
```
On `success:false`, `iris`/`pupil` are `null`.

## Hardware caveat

The `imgsz=1280` default targets the **training box (RTX 3060 12 GB)**, where `batch=8` fits —
`batch=12` OOMs it (yolo11m@1280 needs ~15–18 GB). The local dev machine is an RTX 2070 8 GB —
if training here, lower `batch` further, use `imgsz=1024`, or `batch=-1` (auto); alternative
config is `yolo11s.pt` at `imgsz=1536`. The known hard case is the tiny pupil
(r≈32 on a 5760-wide image → ~14 px box at imgsz=1280): if pupil recall is poor, raise `imgsz` to
1536, or train iris-only and detect pupil on the iris crop (fallback, not the default).

## Verification (no automated tests)

1. After step 1: draw a few labels back on photos (step 2) and compare to `_debug_*.jpg` — circle
   centers/radii should coincide.
2. After step 3: confirm split id-ranges don't overlap, buffer ids were dropped at junctions, and
   the test split's no-eye fraction is > 0.
3. After training: `yolo detect val ... split=test` — target iris mAP50 > 0.9; judge pupil by
   applied center/radius error in px rather than mAP.
