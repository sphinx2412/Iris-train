"""Convert manual iris/pupil annotations (jsonOutputs/*.json) into YOLO labels.

For each ``<stem>.json`` that has a sibling source image, writes ``labels/<stem>.txt``:

  * ``detection.success == true``  -> two lines (class cx cy w h, all normalized 0..1):
        0  cx/W  cy/H  (2r)/W  (2r)/H      # iris
        1  cx/W  cy/H  (2r)/W  (2r)/H      # pupil
  * ``detection.success == false`` -> an empty .txt  (YOLO treats it as a background frame).

W,H are taken from ``source.width/height`` (the coordinate system the labels live in),
not from the image file. The image only needs to exist; its pixels are never read here.

Usage:
    python tools/json_to_yolo.py [--src jsonOutputs] [--out labels]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_source_image(json_path: Path) -> Path | None:
    """Return the sibling source image for a label json, or None if missing.

    Matches ``<stem>.<ext>`` case-insensitively (e.g. .JPG on disk vs .jpg here).
    """
    stem = json_path.stem
    for p in json_path.parent.iterdir():
        if p.stem == stem and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def circle_to_yolo_line(cls: int, cx: float, cy: float, r: float, W: int, H: int) -> str:
    """A circle (center, radius) -> a YOLO bbox line, normalized and clipped to [0,1]."""
    def clip(v: float) -> float:
        return min(1.0, max(0.0, v))

    nx = clip(cx / W)
    ny = clip(cy / H)
    nw = clip((2.0 * r) / W)
    nh = clip((2.0 * r) / H)
    return f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def convert(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Count actual files at runtime — never assume a fixed total.
    jsons = sorted(p for p in src.glob("*.json") if not p.name.startswith("_debug_"))
    print(f"Found {len(jsons)} json label file(s) in {src}")

    n_written = n_empty = n_skipped = 0
    for jp in jsons:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {jp.name}: cannot read json ({e})")
            n_skipped += 1
            continue

        img = find_source_image(jp)
        if img is None:
            print(f"  SKIP {jp.name}: no sibling source image")
            n_skipped += 1
            continue

        txt_path = out / f"{jp.stem}.txt"
        success = bool(data.get("detection", {}).get("success"))

        if not success:
            txt_path.write_text("", encoding="utf-8")
            n_empty += 1
            continue

        source = data.get("source", {})
        W, H = source.get("width"), source.get("height")
        if not W or not H:
            print(f"  SKIP {jp.name}: missing source.width/height")
            n_skipped += 1
            continue

        lines = []
        for cls, key in ((0, "iris"), (1, "pupil")):
            obj = data.get(key)
            if not obj:
                continue
            lines.append(
                circle_to_yolo_line(
                    cls, obj["center_x"], obj["center_y"], obj["radius"], W, H
                )
            )
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_written += 1

    print(
        f"\nDone. labels with objects: {n_written}  empty(no-eye): {n_empty}  "
        f"skipped: {n_skipped}  -> {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("jsonOutputs"),
                    help="folder with <stem>.json + source images (default: jsonOutputs)")
    ap.add_argument("--out", type=Path, default=Path("labels"),
                    help="output folder for YOLO .txt labels (default: labels)")
    args = ap.parse_args()
    convert(args.src, args.out)


if __name__ == "__main__":
    main()
