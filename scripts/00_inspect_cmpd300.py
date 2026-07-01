"""00_inspect_cmpd300.py — CMPD300 inspection + split generation (Phase 5).

CMPD300 already comes split into folders (train/val/test → one subfolder per ID), so
it is NOT re-split (01_make_splits is not needed for this dataset). This script:

  1. walks <CMPD300_DIR>/{train,val,test}/<ID>/*.JPG,
  2. reports number of classes, images per split, min/max/mean per class, IDs missing
     from val/test, and (optionally) corrupt images,
  3. builds the label_map (folder→integer, from PRESENT folders, not assuming 1..N),
  4. writes outputs/splits_cmpd300/{train,val,test}.json + label_map.json,
     in the SAME {"path","label"} format read by src/dataset.py (paths RELATIVE to
     config.CMPD300_DIR → the same split works locally and on Kaggle).

Usage:
    python scripts/00_inspect_cmpd300.py                  # report + generate splits
    python scripts/00_inspect_cmpd300.py --check-corrupt  # also verify unreadable images
    python scripts/00_inspect_cmpd300.py --no-write        # report only, do not write
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# allow running from any cwd (project root on sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils import get_logger, save_json

SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # case-insensitive (CMPD300 uses .JPG)


def list_images(d: Path) -> list[Path]:
    """Images (by extension, case-insensitive) inside a folder."""
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def list_class_dirs(split_dir: Path) -> list[str]:
    """Names of subfolders (IDs) present in a split."""
    if not split_dir.is_dir():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def scan(cmpd_dir: Path) -> dict:
    """Walk all three splits and return, per split, {ID: [relative paths]}."""
    per_split: dict[str, dict[str, list[str]]] = {}
    for split in SPLITS:
        split_dir = cmpd_dir / split
        by_class: dict[str, list[str]] = {}
        for cls in list_class_dirs(split_dir):
            imgs = list_images(split_dir / cls)
            by_class[cls] = [p.relative_to(cmpd_dir).as_posix() for p in imgs]
        per_split[split] = by_class
    return per_split


def stats(by_class: dict[str, list[str]]) -> dict:
    counts = [len(v) for v in by_class.values()]
    n_imgs = sum(counts)
    return {
        "n_classes": len(by_class),
        "n_images": n_imgs,
        "min_per_class": min(counts) if counts else 0,
        "max_per_class": max(counts) if counts else 0,
        "mean_per_class": round(n_imgs / len(by_class), 2) if by_class else 0.0,
    }


def check_corrupt(cmpd_dir: Path, per_split: dict, log) -> list[str]:
    """Try to open each image; return unreadable paths."""
    from PIL import Image
    bad: list[str] = []
    total = 0
    for split in SPLITS:
        for rels in per_split[split].values():
            for rel in rels:
                total += 1
                try:
                    with Image.open(cmpd_dir / rel) as im:
                        im.convert("RGB").load()
                except Exception as e:  # noqa: BLE001
                    bad.append(rel)
                    log.warning(f"corrupt: {rel} ({e})")
    log.info(f"checked {total} images | corrupt: {len(bad)}")
    return bad


def build_label_map(per_split: dict, log) -> dict[str, int]:
    """folder→integer 0..N-1, for classes PRESENT in train (canonical)."""
    train_ids = set(per_split["train"].keys())
    all_ids = set().union(*(set(per_split[s].keys()) for s in SPLITS))

    # every class in val/test should be in train (otherwise cannot be learned)
    not_in_train = sorted(all_ids - train_ids)
    if not_in_train:
        log.warning(f"⚠ {len(not_in_train)} classes in val/test are NOT in train "
                    f"(cannot be learned): {not_in_train}")

    classes = sorted(train_ids)  # numbered by folders present in train
    return {cls: i for i, cls in enumerate(classes)}


def to_entries(by_class: dict[str, list[str]], label_map: dict[str, int]) -> list[dict]:
    """{ID: [paths]} → [{'path','label'}], skipping classes without a label (not in train)."""
    entries: list[dict] = []
    for cls, rels in sorted(by_class.items()):
        if cls not in label_map:
            continue
        lbl = label_map[cls]
        entries += [{"path": rel, "label": lbl} for rel in rels]
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="CMPD300 inspection + split generation.")
    ap.add_argument("--check-corrupt", action="store_true",
                    help="verify that each image can be opened (slower).")
    ap.add_argument("--no-write", action="store_true",
                    help="report only, do not write JSON files.")
    args = ap.parse_args()

    log = get_logger("inspect.cmpd300")
    cmpd_dir = config.CMPD300_DIR
    log.info(f"CMPD300_DIR: {cmpd_dir}  (exists: {cmpd_dir.is_dir()})")
    if not (cmpd_dir / "train").is_dir():
        log.error("Cannot find <CMPD300_DIR>/train. Check config.CMPD300_DIR "
                  "or set CMPD300_DATA_DIR.")
        sys.exit(1)

    per_split = scan(cmpd_dir)

    # ---- Per-split report ----
    report: dict = {"cmpd300_dir": str(cmpd_dir), "splits": {}}
    for split in SPLITS:
        s = stats(per_split[split])
        report["splits"][split] = s
        log.info(f"[{split:5s}] classes={s['n_classes']:4d} imgs={s['n_images']:5d} "
                 f"min={s['min_per_class']} max={s['max_per_class']} "
                 f"mean={s['mean_per_class']}")

    # ---- Missing classes across splits ----
    train_ids = set(per_split["train"].keys())
    for split in ("val", "test"):
        missing = sorted(train_ids - set(per_split[split].keys()))
        report["splits"][split]["missing_vs_train"] = missing
        if missing:
            log.warning(f"[{split}] missing {len(missing)} IDs that are in train: "
                        f"{missing}")

    # ---- label_map + num_classes ----
    label_map = build_label_map(per_split, log)
    num_classes = len(label_map)
    report["num_classes"] = num_classes
    total_imgs = sum(report["splits"][s]["n_images"] for s in SPLITS)
    report["total_images"] = total_imgs
    log.info(f"=> num_classes (folders in train) = {num_classes} | "
             f"total images = {total_imgs}")

    # ---- Corrupt check (optional) ----
    if args.check_corrupt:
        report["corrupt"] = check_corrupt(cmpd_dir, per_split, log)

    # ---- Write splits + label_map + report ----
    if not args.no_write:
        config.CMPD300_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            entries = to_entries(per_split[split], label_map)
            save_json(entries, config.CMPD300_SPLITS_DIR / f"{split}.json")
            log.info(f"wrote {split}.json ({len(entries)} entries)")
        save_json(label_map, config.CMPD300_SPLITS_DIR / "label_map.json")
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(report, config.RESULTS_DIR / "00_inspect_cmpd300.json")
        log.info(f"label_map.json + report written to {config.CMPD300_SPLITS_DIR} "
                 f"and {config.RESULTS_DIR}")
    else:
        log.info("--no-write: nothing written.")

    log.info("Done.")


if __name__ == "__main__":
    main()
