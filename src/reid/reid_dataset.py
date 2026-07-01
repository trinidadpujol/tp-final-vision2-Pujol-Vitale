"""reid_dataset.py — Armado de entries y split gallery/probe por identidad (Fase 6).

Lee un dataset organizado como <root>/<individuo>/*.jpg (CMPD300/train, caras de Ahmed,
etc.), arma entries {path,label} y los parte en gallery/probe POR IDENTIDAD.

Multi-shot: por cada individuo con >= min_images imágenes, la mitad va a gallery y la
otra a probe. Los individuos con menos de min_images se descartan (no se pueden evaluar)
y se reportan.
"""
from __future__ import annotations

import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def entries_from_folders(root: Path, max_per_id: int | None = None) -> tuple[list[dict], dict]:
    """<root>/<id>/*.img → (entries [{path,label}], id_map {carpeta: entero}).

    `path` es relativa a `root` (para usar como data_dir en el extractor).
    `max_per_id`: si se pasa, corta a las primeras N imágenes por individuo (velocidad).
    """
    root = Path(root)
    id_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    id_map = {name: i for i, name in enumerate(id_names)}
    entries: list[dict] = []
    for name in id_names:
        imgs = sorted(f for f in (root / name).iterdir()
                      if f.is_file() and f.suffix.lower() in IMG_EXTS)
        if max_per_id is not None:
            imgs = imgs[:max_per_id]
        for f in imgs:
            entries.append({"path": (Path(name) / f.name).as_posix(), "label": id_map[name]})
    return entries, id_map


def split_gallery_probe(entries: list[dict], seed: int = 0, min_images: int = 2,
                        gallery_frac: float = 0.5) -> tuple[list[dict], list[dict], dict]:
    """Parte entries en (gallery, probe) por identidad. Devuelve también info del split."""
    by_label: dict[int, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)

    rng = random.Random(seed)
    gallery, probe = [], []
    used, dropped = 0, 0
    for lab, items in sorted(by_label.items()):
        if len(items) < min_images:
            dropped += 1
            continue
        items = items[:]
        rng.shuffle(items)
        n_gal = max(1, int(round(len(items) * gallery_frac)))
        n_gal = min(n_gal, len(items) - 1)          # dejar al menos 1 para probe
        gallery += items[:n_gal]
        probe += items[n_gal:]
        used += 1

    info = {
        "n_ids_total": len(by_label),
        "n_ids_used": used,
        "n_ids_dropped_lt_min": dropped,
        "min_images": min_images,
        "n_gallery": len(gallery),
        "n_probe": len(probe),
    }
    return gallery, probe, info
