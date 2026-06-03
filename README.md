# Iris-train

Train a fast YOLO11 detector for **iris** and **pupil** circles `(cx, cy, r)` from eye
photos, to replace the slow SAM-based pipeline. This repo holds the data-prep tooling and
the training runbook.

> Scope of the current init: scaffolding + scripts only. **No training has been run.**

## Layout

```
jsonOutputs/        # source: <stem>.<ext> + <stem>.json (+ ignored _debug_<stem>.jpg)
tools/
  json_to_yolo.py   # json annotations -> YOLO labels/<stem>.txt
  make_split.py     # contiguous block split by id (+ buffer) -> dataset/ + data.yaml
  visualize_labels.py  # draw labels back onto photos for a sanity check
labels/             # generated YOLO .txt  (gitignored)
dataset/            # generated YOLO dataset + data.yaml  (gitignored)
runs/               # training outputs  (gitignored)
```

## Setup

Run `0_setup_install.bat` to create `.venv` (if missing) and install everything. To do it by
hand on the training box (RTX 3060 12 GB), install a CUDA torch build first:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For data-prep only (no GPU), `pip install numpy opencv-python pyyaml` is enough.

## Pipeline

```powershell
# 1. Annotations -> YOLO labels (empty .txt for "no eye" frames)
python tools/json_to_yolo.py

# 2. Sanity check: draw a few labels back onto the photos, compare to _debug_*.jpg
python tools/visualize_labels.py --n 5

# 3. Block split by id (+ buffer) -> dataset/ + data.yaml. test = last contiguous block.
python tools/make_split.py

# 4. Train  (on the RTX 3060 12 GB box)
yolo detect train model=yolo11m.pt data=dataset/data.yaml imgsz=1280 epochs=100 batch=8 patience=20 device=0

# 4b. Resume an interrupted run (instead of step 4; do NOT re-run the split first)
yolo detect train resume model=runs/detect/train/weights/last.pt

# 5. Honest eval on the frozen test split (per-class precision/recall, mAP)
yolo detect val model=runs/detect/train/weights/best.pt data=dataset/data.yaml split=test

# 6. Export
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=1280

# 7. Inference — consume best.onnx, draw predicted circles / write JSON annotations
python tools/infer_onnx.py jsonOutputs --model runs/detect/train/weights/best.onnx --viz viz_infer --out infer_out
```

Step 7 is the *consumer* side: a lightweight `onnxruntime` script (no torch/ultralytics)
that loads `best.onnx`, detects iris/pupil, runs NMS (export has none), and inverts each
bbox back to a circle `(cx, cy, r)`. `--out` writes `<stem>.json` in the **same schema as
the input annotations** (drop-in for the old SAM output); `--viz` draws overlays to compare
against `_debug_*.jpg`. Install the runtime first: `pip install numpy opencv-python
onnxruntime` (GPU optional: `onnxruntime-gpu`). `7_infer_onnx.bat` wraps it — the model path
is an editable `MODEL=` variable at the top (defaults to `runs\detect\train\weights\best.onnx`)
and can also be passed as the first argument.

## Notes

- **Photo count is never hardcoded** — every script counts the actual files at runtime and
  logs the real numbers (annotation is ongoing; ~1600 is only a target).
- **GPU caveat:** the `imgsz=1280` default targets the **RTX 3060 12 GB** training box,
  where `batch=8` fits (batch=12 OOMs — yolo11m@1280 needs ~15–18 GB). This dev machine is
  an RTX 2070 8 GB; if you train here, lower `batch` further, use `imgsz=1024`, or `batch=-1`
  (auto). Alternative: `yolo11s.pt` at `imgsz=1536`.
- **Split is contiguous-by-id, not random** — photos span only ~4 days and same-person
  repeats occur only among neighbouring ids; a buffer is dropped at each junction to prevent
  leakage. The `test` block is frozen — don't re-run the split against it once training starts.
- **Training is resumable.** A `last.pt` checkpoint is written at the end of every epoch under
  `runs/detect/train/weights/`. If the console is closed / the run crashes, continue with step 4b
  (`4b_resume_train.bat`) instead of step 4 — re-running step 4 starts from scratch. `resume=True`
  reuses `args.yaml` and the same frozen split, so don't re-run the split in between. You lose only
  the partial current epoch, not completed ones.
- `jsonOutputs/` originals are large (≈5760×3240). Decide deliberately whether to commit them.
