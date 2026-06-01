"""Draw YOLO labels back onto the source photo, to eyeball the converter output.

For each chosen stem, reads jsonOutputs/<stem>.<ext> (unicode-safe) and labels/<stem>.txt,
denormalizes the bboxes, and draws both the bbox and its equivalent circle
(iris = green, pupil = red). Saves overlays to viz/<stem>_label.jpg so you can compare
against the existing _debug_<stem>.jpg.

Usage:
    python tools/visualize_labels.py [stem ...]        # specific stems
    python tools/visualize_labels.py --n 5             # 5 random labelled stems
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # BGR: iris green, pupil red
NAMES = {0: "iris", 1: "pupil"}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_source_image(src: Path, stem: str) -> Path | None:
    for p in src.iterdir():
        if p.stem == stem and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def draw(stem: str, src: Path, labels: Path, out: Path) -> bool:
    img_path = find_source_image(src, stem)
    lbl_path = labels / f"{stem}.txt"
    if img_path is None or not lbl_path.exists():
        print(f"  SKIP {stem}: missing image or label")
        return False

    img = imread_unicode(img_path)
    if img is None:
        print(f"  SKIP {stem}: cannot decode image")
        return False
    H, W = img.shape[:2]

    lines = [ln for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        cv2.putText(img, "no-eye (empty label)", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    for ln in lines:
        cls, nx, ny, nw, nh = ln.split()
        cls = int(cls)
        cx, cy = float(nx) * W, float(ny) * H
        bw, bh = float(nw) * W, float(nh) * H
        r = (bw + bh) / 4.0
        color = COLORS.get(cls, (255, 255, 255))
        p1 = (int(cx - bw / 2), int(cy - bh / 2))
        p2 = (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(img, p1, p2, color, 3)
        cv2.circle(img, (int(cx), int(cy)), int(r), color, 3)
        cv2.putText(img, NAMES.get(cls, str(cls)), (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{stem}_label.jpg"
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(str(dst))  # unicode-safe write
        print(f"  wrote {dst}")
        return True
    print(f"  SKIP {stem}: cannot encode output")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems", nargs="*", help="specific stems to render")
    ap.add_argument("--src", type=Path, default=Path("jsonOutputs"))
    ap.add_argument("--labels", type=Path, default=Path("labels"))
    ap.add_argument("--out", type=Path, default=Path("viz"))
    ap.add_argument("--n", type=int, default=0, help="render N random labelled stems")
    args = ap.parse_args()

    stems = list(args.stems)
    if args.n and not stems:
        all_lbls = [p.stem for p in args.labels.glob("*.txt")]
        # random.sample needs no global seed; fine for a visual spot-check
        stems = random.sample(all_lbls, min(args.n, len(all_lbls)))
    if not stems:
        print("No stems given. Pass stems or --n N.")
        return

    for s in stems:
        draw(s, args.src, args.labels, args.out)


if __name__ == "__main__":
    main()
