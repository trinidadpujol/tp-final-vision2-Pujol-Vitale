"""dataset.py — PyTorch Dataset + DataLoader reading from split JSON files.

Splits are generated once by `scripts/01_make_splits.py` and saved in
`outputs/splits/`. This module only reads them (no re-splitting per run → reproducibility).

Paths in the JSONs are RELATIVE to `config.DATA_DIR`, so the same split works
on Kaggle and locally without rewriting paths.

Stage 2: `make_dataloader` and `make_train_loader` accept a `data_dir` argument
(default `config.DATA_DIR`) to point at a different dataset — e.g. `config.CMPD300_DIR` —
reusing the same Dataset class and split JSON format.

Designed for the future embedding phase: `MuzzleDataset` returns (image, label) and
optionally the path, sufficient for gallery/probe downstream.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import config
from src.utils import load_json


class MuzzleDataset(Dataset):
    """Muzzle image dataset. Reads {"path", "label"} entries from a split file."""

    def __init__(self, entries: list[dict], transform=None,
                 data_dir: Path = config.DATA_DIR, return_path: bool = False):
        self.entries = entries
        self.transform = transform
        self.data_dir = Path(data_dir)
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        path = self.data_dir / e["path"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = e["label"]
        if self.return_path:
            return img, label, str(e["path"])
        return img, label


def load_split(split_name: str, splits_dir: Path = config.SPLITS_DIR) -> list[dict]:
    """Load a split ('train' | 'val' | 'test') from its JSON file."""
    path = Path(splits_dir) / f"{split_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Generate splits first: "
            f"python scripts/01_make_splits.py"
        )
    return load_json(path)


def _make_loader(ds, *, shuffle: bool, batch_size: int, num_workers: int) -> DataLoader:
    """DataLoader over an already-built Dataset (shared helper)."""
    # persistent_workers: with 50 epochs avoids recreating workers each epoch
    # (only applies when num_workers > 0). Does not change data or results, only speed.
    extra = {}
    if num_workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = 4
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        **extra,
    )


def make_dataloader(entries: list[dict], transform, *, shuffle: bool,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    data_dir: Path = config.DATA_DIR,
                    return_path: bool = False) -> DataLoader:
    """Build a DataLoader over an already-loaded split (single transform for all entries).

    `data_dir` (Stage 2): root from which entry paths are resolved.
    Default = config.DATA_DIR (paper dataset); pass config.CMPD300_DIR for CMPD300.
    """
    ds = MuzzleDataset(entries, transform=transform, data_dir=data_dir,
                       return_path=return_path)
    return _make_loader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)


def build_augmented_entries(entries: list[dict],
                            cap: int = config.AUG_TARGET_CAP,
                            factor: int = config.AUG_FACTOR,
                            seed: int = 0) -> list[dict]:
    """EXTRA entries to synthesize per class to enlarge the dataset.

    PAPER: augmentation "creates synthetic images and enlarges the dataset", especially for
    classes with few images, KEEPING the originals. Per class, expand up to
    `min(cap, factor * N_i)`; returns only the EXTRA copies (originals are added separately
    with the clean transform). Sampling is deterministic by `seed` (reproducibility).
    """
    by_label: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_label[e["label"]].append(e)
    rng = random.Random(seed)
    extra: list[dict] = []
    for label, items in sorted(by_label.items()):
        n = len(items)
        n_add = max(0, min(cap, factor * n) - n)
        for _ in range(n_add):
            extra.append(rng.choice(items))
    return extra


def load_augmented_manifest(cache_dir: Path = config.AUG_CACHE_DIR) -> list[dict] | None:
    """Return the precomputed synthetic image manifest, or None if it does not exist."""
    mp = Path(cache_dir) / "aug_manifest.json"
    return load_json(mp) if mp.is_file() else None


def make_train_loader(entries: list[dict], *, use_aug: bool, clean_tf, aug_tf,
                      seed: int = 0,
                      batch_size: int = config.BATCH_SIZE,
                      num_workers: int = config.NUM_WORKERS,
                      cap: int = config.AUG_TARGET_CAP,
                      factor: int = config.AUG_FACTOR,
                      use_precomputed: bool = True,
                      aug_cache_dir: Path = config.AUG_CACHE_DIR,
                      data_dir: Path = config.DATA_DIR) -> DataLoader:
    """Training DataLoader.

    - `use_aug=False`: originals with clean transform (ce / wce variants).
    - `use_aug=True` (PAPER): clean originals + augmented synthetic copies → the dataset is
      ENLARGED, not replaced. This way the model still sees images at real brightness
      (avoids train/test mismatch) and gains variety in classes with few images.

    For synthetic copies, if `use_precomputed` is True and the manifest from
    `scripts/04_precompute_aug.py` exists, the FIXED disk images are used (clean transform,
    already augmented). Otherwise, they are generated online with `aug_tf` (fallback; e.g.
    in smoke-tests or for datasets without an aug cache like CMPD300 → pass use_precomputed=False).

    `data_dir` (Stage 2): root for the original entries and online copies.
    (The precomputed cache is always resolved against `aug_cache_dir`.)
    """
    clean_ds = MuzzleDataset(entries, transform=clean_tf, data_dir=data_dir)
    if not use_aug:
        return _make_loader(clean_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    manifest = load_augmented_manifest(aug_cache_dir) if use_precomputed else None
    if manifest is not None:
        # Precomputed: images already augmented on disk → only resize + ToTensor.
        aug_ds = MuzzleDataset(manifest, transform=clean_tf, data_dir=aug_cache_dir)
    else:
        # Online fallback: augment on the fly (same expansion criterion).
        extra = build_augmented_entries(entries, cap=cap, factor=factor, seed=seed)
        aug_ds = MuzzleDataset(extra, transform=aug_tf, data_dir=data_dir)
    train_ds = ConcatDataset([clean_ds, aug_ds])
    return _make_loader(train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
