"""Run the exported YOLO11 detector (best.onnx) and turn detections back into circles.

This is the *consumer* side of the pipeline: it loads the ONNX model with onnxruntime
(no torch / ultralytics needed), detects iris (class 0) and pupil (class 1), and inverts
each bbox back to the circle the annotations actually describe:

    cx, cy = box center      r = (w + h) / 4

The model was exported WITHOUT built-in NMS (see 6_export_onnx.bat), so confidence
thresholding and NMS are done here.

For each image we print the detections, and optionally:
  * --out DIR : write <stem>.json in the SAME schema as the input annotations
                (source / detection / iris / pupil), so it's a drop-in replacement for
                the old SAM pipeline output. No iris found -> detection.success = false.
  * --viz DIR : draw bbox + circle overlays to <stem>_pred.jpg for a visual check.

Usage:
    python tools/infer_onnx.py IMG [IMG ...] [--model best.onnx] [--out infer_out] [--viz viz_infer]
    python tools/infer_onnx.py jsonOutputs --model runs/detect/train/weights/best.onnx --viz viz_infer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # BGR: iris green, pupil red
NAMES = {0: "iris", 1: "pupil"}
PAD_COLOR = 114  # ultralytics letterbox pad value


# ---------------------------------------------------------------- unicode-safe I/O

def imread_unicode(path: Path):
    """Read an image via numpy so non-ASCII Windows paths work (cv2.imread does not)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img) -> bool:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


# ---------------------------------------------------------------- model I/O

def letterbox(img, new: int):
    """Resize keeping aspect ratio and pad to new x new. Returns (padded, ratio, dw, dh)."""
    h, w = img.shape[:2]
    ratio = min(new / h, new / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2, (new - nh) / 2  # split padding both sides
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_COLOR, PAD_COLOR, PAD_COLOR),
    )
    return padded, ratio, left, top


def preprocess(img, imgsz: int):
    padded, ratio, dw, dh = letterbox(img, imgsz)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # HWC -> CHW
    blob = np.ascontiguousarray(blob[None], dtype=np.float32) / 255.0  # add batch, scale
    return blob, ratio, dw, dh


def decode(out, imgsz: int, conf_thr: float):
    """Raw YOLO11 output -> (boxes_xywh, scores, classes) in letterbox-pixel space.

    Output is [1, 4+nc, N]; squeeze and transpose to [N, 4+nc]. First 4 = cx,cy,w,h,
    rest = per-class scores (no objectness in YOLO11). Robust to either axis order.
    """
    arr = np.squeeze(out, axis=0) if out.ndim == 3 else out
    if arr.shape[0] < arr.shape[1]:  # [channels, anchors] -> [anchors, channels]
        arr = arr.T
    boxes = arr[:, :4].astype(np.float32)
    cls_scores = arr[:, 4:]
    classes = cls_scores.argmax(axis=1)
    scores = cls_scores.max(axis=1)

    # Heuristic: ultralytics exports pixel coords; if they look normalized, scale up.
    if boxes[:, :4].max(initial=0.0) <= 1.5:
        boxes *= imgsz

    keep = scores >= conf_thr
    return boxes[keep], scores[keep], classes[keep]


def nms_per_class(boxes_xywh, scores, classes, iou_thr: float, conf_thr: float):
    """Class-wise NMS via cv2.dnn.NMSBoxes. boxes_xywh are center-form; convert to TL."""
    kept = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        rects = []
        for i in idx:
            cx, cy, w, h = boxes_xywh[i]
            rects.append([float(cx - w / 2), float(cy - h / 2), float(w), float(h)])
        picks = cv2.dnn.NMSBoxes(rects, scores[idx].tolist(), conf_thr, iou_thr)
        for p in np.array(picks).reshape(-1):
            kept.append(idx[int(p)])
    return kept


def detect(session, input_name, img, imgsz, conf_thr, iou_thr):
    """Return a list of dicts: {cls, name, cx, cy, r, conf} in original-image pixels."""
    blob, ratio, dw, dh = preprocess(img, imgsz)
    out = session.run(None, {input_name: blob})[0]
    boxes, scores, classes = decode(out, imgsz, conf_thr)
    if len(boxes) == 0:
        return []

    dets = []
    for i in nms_per_class(boxes, scores, classes, iou_thr, conf_thr):
        cx, cy, w, h = boxes[i]
        # undo letterbox: remove pad, then unscale to original pixels
        cx = (cx - dw) / ratio
        cy = (cy - dh) / ratio
        w, h = w / ratio, h / ratio
        cls = int(classes[i])
        dets.append({
            "cls": cls, "name": NAMES.get(cls, str(cls)),
            "cx": float(cx), "cy": float(cy), "r": float((w + h) / 4.0),
            "w": float(w), "h": float(h), "conf": float(scores[i]),
        })
    return dets


# ---------------------------------------------------------------- outputs

def best_per_class(dets, cls):
    cand = [d for d in dets if d["cls"] == cls]
    return max(cand, key=lambda d: d["conf"]) if cand else None


def to_annotation_json(img_path: Path, W: int, H: int, dets) -> dict:
    """Build a record in the input annotation schema (schema_version 1)."""
    iris, pupil = best_per_class(dets, 0), best_per_class(dets, 1)

    def circle(d):
        return None if d is None else {
            "center_x": round(d["cx"], 2),
            "center_y": round(d["cy"], 2),
            "radius": round(d["r"], 2),
        }

    return {
        "schema_version": 1,
        "source": {"filename": img_path.name, "width": W, "height": H},
        "detection": {"success": iris is not None, "tier": "yolo-onnx"},
        "iris": circle(iris),
        "pupil": circle(pupil),
    }


def draw_overlay(img, dets):
    vis = img.copy()
    if not dets:
        cv2.putText(vis, "no detection", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    for d in dets:
        color = COLORS.get(d["cls"], (255, 255, 255))
        cx, cy, w, h, r = d["cx"], d["cy"], d["w"], d["h"], d["r"]
        p1 = (int(cx - w / 2), int(cy - h / 2))
        p2 = (int(cx + w / 2), int(cy + h / 2))
        cv2.rectangle(vis, p1, p2, color, 3)
        cv2.circle(vis, (int(cx), int(cy)), int(r), color, 3)
        cv2.putText(vis, f"{d['name']} {d['conf']:.2f}", (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return vis


# ---------------------------------------------------------------- driver

def gather_images(paths) -> list[Path]:
    imgs: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            imgs += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
        elif p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
        else:
            print(f"  SKIP {p}: not an image or directory")
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image file(s) or a folder of images")
    ap.add_argument("--model", type=Path, default=Path("best.onnx"),
                    help="path to the exported ONNX model (default: best.onnx)")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size (match training)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    ap.add_argument("--out", type=Path, default=None, help="folder for <stem>.json output")
    ap.add_argument("--viz", type=Path, default=None, help="folder for <stem>_pred.jpg overlays")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                    help="onnxruntime execution providers (e.g. CUDAExecutionProvider)")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    session = ort.InferenceSession(str(args.model), providers=args.providers)
    input_name = session.get_inputs()[0].name
    print(f"Loaded {args.model}  (providers: {session.get_providers()})")

    images = gather_images(args.inputs)
    print(f"Running inference on {len(images)} image(s)  imgsz={args.imgsz} "
          f"conf={args.conf} iou={args.iou}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    if args.viz:
        args.viz.mkdir(parents=True, exist_ok=True)

    for ip in images:
        img = imread_unicode(ip)
        if img is None:
            print(f"  SKIP {ip.name}: cannot decode image")
            continue
        H, W = img.shape[:2]
        dets = detect(session, input_name, img, args.imgsz, args.conf, args.iou)

        if dets:
            summary = ", ".join(
                f"{d['name']}(cx={d['cx']:.0f},cy={d['cy']:.0f},r={d['r']:.1f},"
                f"conf={d['conf']:.2f})" for d in dets)
        else:
            summary = "no detection"
        print(f"  {ip.name}: {summary}")

        if args.out:
            rec = to_annotation_json(ip, W, H, dets)
            (args.out / f"{ip.stem}.json").write_text(
                json.dumps(rec, indent=2), encoding="utf-8")
        if args.viz:
            dst = args.viz / f"{ip.stem}_pred.jpg"
            if not imwrite_unicode(dst, draw_overlay(img, dets)):
                print(f"  SKIP viz {ip.name}: cannot encode overlay")

    print("Done.")


if __name__ == "__main__":
    main()
