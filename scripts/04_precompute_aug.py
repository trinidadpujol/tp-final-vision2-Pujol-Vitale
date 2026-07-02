"""04_precompute_aug.py — Precompute synthetic (augmented) images to disk.

PAPER: data augmentation is a PRE-PROCESSING STEP that "creates synthetic images
and enlarges the dataset". Here we generate them ONCE and save them as files,
instead of re-augmenting online every epoch. Advantages: the synthetic set is FIXED
(faithful to the paper), reproducible and inspectable (you can open the .jpg files).

Per class we add `max(0, min(AUG_TARGET_CAP, AUG_FACTOR*N_i) - N_i)` copies,
sampling with replacement from that class's images (fixed seed) and applying
flip / brightness (0.2–0.5) / rotation (±15°) / blur. Same criterion as the online
path (`dataset.build_augmented_entries`), only here it is materialized on disk.

Output (in OUTPUT/aug_cache/):
  <class_name>/<orig_stem>__aug<k>.jpg   # synthetic images (300x300)
  aug_manifest.json                      # [{"path", "label"}] relative to aug_cache/

Training uses this cache automatically when it exists (see dataset.make_train_loader).

NOTE: does NOT speed up training (the bottleneck is the GPU forward pass, which
processes the same number of images). Serves to fix and reproduce the synthetic set.

Usage:
    python scripts/04_precompute_aug.py            # generate if cache does not exist
    python scripts/04_precompute_aug.py --force    # regenerate
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import config  # noqa: E402
from src.dataset import build_augmented_entries, load_split  # noqa: E402
from src.transforms import build_pil_aug_transform  # noqa: E402
from src.utils import get_logger, load_json, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED,
                    help="Seed for sampling + random ops (reproducibility).")
    ap.add_argument("--force", action="store_true", help="Regenerate even if cache exists.")
    args = ap.parse_args()

    log = get_logger("04_precompute_aug")
    config.ensure_output_dirs()
    cache = config.AUG_CACHE_DIR
    manifest_path = cache / "aug_manifest.json"

    if manifest_path.is_file() and not args.force:
        m = load_json(manifest_path)
        log.info(f"Cache already exists at {manifest_path} ({len(m)} images). "
                 f"Use --force to regenerate.")
        return 0

    cache.mkdir(parents=True, exist_ok=True)
    label_map = load_json(config.SPLITS_DIR / "label_map.json")  # name -> idx
    idx_to_name = {v: k for k, v in label_map.items()}

    train = load_split("train")
    extra = build_augmented_entries(train, seed=args.seed)
    log.info(f"Generating {len(extra)} synthetic images in {cache} ...")

    aug_tf = build_pil_aug_transform()  # PIL ops (resize + aug), without ToTensor
    # Determinism for torchvision random ops (they use torch's RNG) + sampling.
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    manifest: list[dict] = []
    counters: dict[tuple, int] = defaultdict(int)
    for e in extra:
        label = e["label"]
        cls = idx_to_name.get(label, f"label_{label}")
        stem = Path(e["path"]).stem
        img = Image.open(config.DATA_DIR / e["path"]).convert("RGB")
        out_img = aug_tf(img)  # PIL 300x300 augmented

        k = counters[(label, stem)]
        counters[(label, stem)] += 1
        rel = f"{cls}/{stem}__aug{k}.jpg"
        (cache / cls).mkdir(parents=True, exist_ok=True)
        out_img.save(cache / rel, quality=95)
        manifest.append({"path": rel, "label": label})

    save_json(manifest, manifest_path)
    log.info(f"Done: {len(manifest)} synthetic images + manifest at {manifest_path}")
    log.info("Training will use them automatically (dataset.make_train_loader).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
