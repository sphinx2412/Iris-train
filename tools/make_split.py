"""Block split by id with a buffer, into a YOLO dataset/ layout + data.yaml.

Why blocks, not random: the photos were shot over only ~4 days, and repeats of the
same person appear ONLY among neighbouring ids. So we split into CONTIGUOUS ranges of
id (train | val | test) and drop a small buffer of ids on each side of every junction,
so one person can't land in two sets. test is the last contiguous block — freeze it.

Layout produced:
    dataset/
      train/{images,labels}/
      val/{images,labels}/
      test/{images,labels}/
    data.yaml   (nc: 2, names: [iris, pupil])

Counts are computed from the actual files at runtime (never a hardcoded total).

Usage:
    python tools/make_split.py [--src jsonOutputs] [--labels labels] [--out dataset]
                               [--buffer 4] [--ratios 0.75 0.15 0.10]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_ID_RE = re.compile(r"(\d+)(?!.*\d)")  # last run of digits in the stem


def numeric_id(stem: str) -> int | None:
    m = _ID_RE.search(stem)
    return int(m.group(1)) if m else None


def find_source_image(src: Path, stem: str) -> Path | None:
    for p in src.iterdir():
        if p.stem == stem and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def is_no_eye(label_path: Path) -> bool:
    """An empty label file (no objects) == a 'no eye' background frame."""
    return label_path.read_text(encoding="utf-8").strip() == ""


def collect(src: Path, labels: Path):
    """Return sorted [(id, stem, image_path, label_path)] for stems that have all parts."""
    items = []
    for jp in sorted(src.glob("*.json")):
        if jp.name.startswith("_debug_"):
            continue
        stem = jp.stem
        nid = numeric_id(stem)
        img = find_source_image(src, stem)
        lbl = labels / f"{stem}.txt"
        if nid is None:
            print(f"  WARN {stem}: no numeric id, skipping")
            continue
        if img is None:
            print(f"  WARN {stem}: no source image, skipping")
            continue
        if not lbl.exists():
            print(f"  WARN {stem}: no label {lbl.name} (run json_to_yolo.py first), skipping")
            continue
        items.append((nid, stem, img, lbl))
    items.sort(key=lambda t: t[0])
    return items


def split_indices(n: int, ratios: tuple[float, float, float], buffer: int):
    """Indices for train/val/test contiguous blocks, with `buffer` ids dropped per junction."""
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # train: [0, n_train), val: [n_train, n_train+n_val), test: [..., n)
    train = list(range(0, n_train))
    val = list(range(n_train, n_train + n_val))
    test = list(range(n_train + n_val, n))

    dropped = set()
    for boundary in (n_train, n_train + n_val):
        for d in range(buffer):
            if boundary - 1 - d >= 0:
                dropped.add(boundary - 1 - d)
            if boundary + d < n:
                dropped.add(boundary + d)

    keep = lambda block: [i for i in block if i not in dropped]
    return keep(train), keep(val), keep(test), dropped


def lay_out(items, idxs, out: Path, split: str) -> tuple[int, int]:
    img_dir = out / split / "images"
    lbl_dir = out / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_no_eye = 0
    for i in idxs:
        _, stem, img, lbl = items[i]
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)
        n_total += 1
        if is_no_eye(lbl):
            n_no_eye += 1
    return n_total, n_no_eye


def report(name, items, idxs, n_total, n_no_eye):
    if idxs:
        lo, hi = items[idxs[0]][0], items[idxs[-1]][0]
        frac = (n_no_eye / n_total) if n_total else 0.0
        print(f"  {name:5s}: {n_total:4d} photos  id[{lo}..{hi}]  no-eye={n_no_eye} ({frac:.0%})")
    else:
        print(f"  {name:5s}: 0 photos")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("jsonOutputs"))
    ap.add_argument("--labels", type=Path, default=Path("labels"))
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--buffer", type=int, default=4,
                    help="ids dropped on EACH side of EACH junction (default 4)")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.75, 0.15, 0.10),
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    print(f"Params: buffer={args.buffer} ratios={tuple(args.ratios)} "
          f"src={args.src} labels={args.labels} out={args.out}")

    items = collect(args.src, args.labels)
    n = len(items)
    print(f"\nUsable stems (json + image + label): {n}")
    if n == 0:
        print("Nothing to split. Run json_to_yolo.py first.")
        return
    if n < 30:
        print(f"WARNING: only {n} samples — block split is NOT meaningful yet; "
              f"this is a smoke run. Re-run after annotation grows.")

    tr, va, te, dropped = split_indices(n, tuple(args.ratios), args.buffer)
    dropped_ids = sorted(items[i][0] for i in dropped)
    print(f"\nDropped {len(dropped_ids)} buffer id(s) at junctions: {dropped_ids}")

    if args.out.exists():
        shutil.rmtree(args.out)
    print("\nSplit:")
    stats = {}
    for name, idxs in (("train", tr), ("val", va), ("test", te)):
        stats[name] = lay_out(items, idxs, args.out, name)
        report(name, items, idxs, *stats[name])

    if stats["test"][0] > 0 and stats["test"][1] == 0:
        print("  NOTE: test has 0 no-eye frames — consider adjusting blocks once data grows.")

    data_yaml = args.out / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(args.out.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 2,
                "names": ["iris", "pupil"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {data_yaml}")


if __name__ == "__main__":
    main()
