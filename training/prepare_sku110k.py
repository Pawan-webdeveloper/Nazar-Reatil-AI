"""Convert SKU-110K Hugging Face parquet shards into YOLO detection format.

Source dataset: https://huggingface.co/datasets/benjamintli/sku110k
(mirror of Goldman et al., "Precise Detection in Densely Packed Scenes", CVPR 2019).
Boxes in the parquet are COCO style [x, y, w, h] in original pixel units.

Images are re-encoded with the longest side capped (default 1280 px) to keep the
dataset small; labels are normalised so they are resolution independent.

Usage:
    python training/prepare_sku110k.py --src <parquet_dir> --dst <yolo_dir>
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def convert_row(img_bytes: bytes, bboxes, max_side: int):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    lines = []
    for b in bboxes:
        x, y, bw, bh = (float(v) for v in b)
        # clip to image bounds
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # degenerate / fully outside
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        nw, nh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return img, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="folder with <split>_<n>.parquet files")
    ap.add_argument("--dst", required=True, help="output YOLO dataset root")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit-per-split", type=int, default=0, help="0 = no limit")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    stats = {}
    for pfile in sorted(src.glob("*.parquet")):
        split_raw = pfile.stem.rsplit("_", 1)[0]
        split = SPLIT_MAP.get(split_raw)
        if split is None:
            print(f"skip {pfile.name}: unknown split")
            continue
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(pfile)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["image", "objects"])
            images = table.column("image").to_pylist()
            objects = table.column("objects").to_pylist()
            for i, (im, ob) in enumerate(zip(images, objects)):
                n = stats.get(split, 0)
                if args.limit_per_split and n >= args.limit_per_split:
                    break
                stem = f"{pfile.stem}_{rg:02d}_{i:04d}"
                img_path = dst / "images" / split / f"{stem}.jpg"
                lbl_path = dst / "labels" / split / f"{stem}.txt"
                if img_path.exists() and lbl_path.exists():
                    stats[split] = n + 1
                    continue
                try:
                    img, lines = convert_row(im["bytes"], ob["bbox"], args.max_side)
                except Exception as exc:  # corrupt image -> skip, never crash the run
                    print(f"  corrupt sample {stem}: {exc}")
                    continue
                if not lines:
                    continue
                img.save(img_path, quality=92)
                lbl_path.write_text("\n".join(lines) + "\n")
                stats[split] = n + 1
            del table, images, objects
        print(f"{pfile.name}: done  running totals {stats}", flush=True)

    yaml_path = dst / "sku110k.yaml"
    splits = {s: f"images/{s}" for s in ("train", "val", "test") if (dst / "images" / s).exists()}
    yaml_path.write_text(
        f"path: {dst.as_posix()}\n"
        + "".join(f"{k}: {v}\n" for k, v in splits.items())
        + "names:\n  0: product\n"
    )
    print("final counts:", stats, "->", yaml_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
