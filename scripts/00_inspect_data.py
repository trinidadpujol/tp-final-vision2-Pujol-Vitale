"""00_inspect_data.py — Phase 0: dataset inspection and validation.

ALWAYS run first. Confirms the real dataset structure before writing any training code.
Do not advance to Phase 1 until the report is correct.

Reports: number of classes, total images, min/max/mean per class, histogram,
extensions present, corrupt/unreadable images, and the smallest classes.
Compares against sanity checks in config.py (268 classes, 4923 images, 4–70 per class).

Usage:
    python scripts/00_inspect_data.py
    python scripts/00_inspect_data.py --check-corrupt   # opens each image (slow)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow importing config.py and src/ when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_class_dirs(data_dir: Path) -> list[Path]:
    """Subdirectories representing a class (one animal)."""
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def count_images(class_dirs: list[Path]) -> tuple[dict[str, int], Counter]:
    """Return {class: image_count} and a Counter of extensions."""
    per_class: dict[str, int] = {}
    ext_counter: Counter = Counter()
    for d in class_dirs:
        n = 0
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                n += 1
                ext_counter[f.suffix.lower()] += 1
        per_class[d.name] = n
    return per_class, ext_counter


def find_corrupt(class_dirs: list[Path]) -> list[str]:
    """Try to open and verify each image; return unreadable paths."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed: skipping corrupt check)")
        return []
    corrupt: list[str] = []
    for d in class_dirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception:  # noqa: BLE001
                    corrupt.append(str(f))
    return corrupt


def histogram(per_class: dict[str, int], bins: int = 10) -> list[tuple[str, int]]:
    """Simple ASCII histogram of images per class."""
    counts = list(per_class.values())
    lo, hi = min(counts), max(counts)
    width = max(1, (hi - lo) // bins + 1)
    edges = list(range(lo, hi + width, width))
    hist: list[tuple[str, int]] = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        c = sum(1 for v in counts if a <= v < b)
        hist.append((f"[{a:>3}-{b:>3})", c))
    # include the upper edge
    hist.append((f"[{edges[-1]:>3}+   )", sum(1 for v in counts if v >= edges[-1])))
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-corrupt", action="store_true",
                    help="Open and verify each image (slow).")
    args = ap.parse_args()

    log = get_logger("inspect")
    data_dir = config.DATA_DIR
    log.info(f"DATA_DIR = {data_dir}")

    if not data_dir.is_dir():
        log.error(f"DATA_DIR does not exist. Set CATTLE_DATA_DIR or extract the dataset. "
                  f"Expected: {data_dir}")
        return 1

    class_dirs = list_class_dirs(data_dir)
    per_class, ext_counter = count_images(class_dirs)
    total = sum(per_class.values())
    counts = list(per_class.values())

    n_classes = len(class_dirs)
    n_min, n_max = min(counts), max(counts)
    n_mean = total / n_classes if n_classes else 0.0

    print("\n" + "=" * 60)
    print("INSPECTION REPORT — Phase 0")
    print("=" * 60)
    print(f"Number of classes (folders) : {n_classes}")
    print(f"Total images                : {total}")
    print(f"Images per class  min       : {n_min}")
    print(f"                  max       : {n_max}")
    print(f"                  mean      : {n_mean:.1f}")
    print(f"Extensions present          : {dict(ext_counter)}")

    print("\nHistogram (images per class):")
    for label, c in histogram(per_class):
        print(f"  {label} | {'#' * c} ({c})")

    print("\n5 classes with FEWEST images:")
    for name, c in sorted(per_class.items(), key=lambda x: x[1])[:5]:
        print(f"  {name}: {c}")
    print("5 classes with MOST images:")
    for name, c in sorted(per_class.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {c}")

    # ---- Sanity checks against config.py ----
    print("\n" + "-" * 60)
    print("SANITY CHECKS (against config.py)")
    print("-" * 60)
    checks = [
        ("num classes", n_classes, config.NUM_CLASSES),
        ("total images", total, config.EXPECTED_IMAGES),
        ("min imgs/class", n_min, config.MIN_IMAGES_PER_CLASS),
        ("max imgs/class", n_max, config.MAX_IMAGES_PER_CLASS),
    ]
    all_ok = True
    for label, got, exp in checks:
        ok = got == exp
        all_ok &= ok
        print(f"  [{'OK ' if ok else 'MISMATCH'}] {label}: got={got} expected={exp}")

    corrupt: list[str] = []
    if args.check_corrupt:
        print("\nChecking for corrupt images (this takes a while)...")
        corrupt = find_corrupt(class_dirs)
        print(f"  Corrupt/unreadable images: {len(corrupt)}")
        for c in corrupt[:10]:
            print(f"    {c}")

    # ---- Save report ----
    config.ensure_output_dirs()
    report = {
        "data_dir": str(data_dir),
        "n_classes": n_classes,
        "total_images": total,
        "min_per_class": n_min,
        "max_per_class": n_max,
        "mean_per_class": round(n_mean, 2),
        "extensions": dict(ext_counter),
        "per_class_counts": per_class,
        "sanity_checks_pass": all_ok,
        "n_corrupt": len(corrupt),
        "corrupt_files": corrupt,
    }
    out = config.RESULTS_DIR / "00_inspect_report.json"
    save_json(report, out)
    print(f"\nReport saved to: {out}")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: sanity checks OK. Ready for Phase 1.")
    else:
        print("RESULT: MISMATCHES found. Review config.py / DEVIATIONS.md "
              "before proceeding.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
