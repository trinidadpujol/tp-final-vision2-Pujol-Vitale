"""reid_dataset.py — entries + gallery/probe split by identity (Phase 6).

Splits:
- `split_gallery_probe`: random split within each individual.
- `split_gallery_probe_by_session`: groups by SESSION (filename timestamp) and does not
  split a session between gallery and probe (avoids matching twin photos from the same burst).

`gallery_shots`: if passed (e.g. 1 = single-shot), the gallery gets exactly that number
of images (or sessions) per individual and the rest goes to probe. Single-shot reduces
burst-photo leakage: with only one reference per individual it is harder to match by photo
similarity instead of biometrics.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BURST_SUFFIX = re.compile(r"-\d+$")   # "-00", "-01", ... at the end of the stem


def session_id(rel_path: str) -> str:
    """Session ID = file stem without the burst suffix `-NN`."""
    return _BURST_SUFFIX.sub("", Path(rel_path).stem)


def entries_from_folders(root: Path, max_per_id: int | None = None) -> tuple[list[dict], dict]:
    """<root>/<id>/*.img → (entries [{path,label}], id_map {folder: int})."""
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
                        gallery_frac: float = 0.5,
                        gallery_shots: int | None = None) -> tuple[list[dict], list[dict], dict]:
    """Random split per individual. `gallery_shots` fixes how many images go to gallery/id."""
    by_label: dict[int, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    rng = random.Random(seed)
    gallery, probe, used, dropped = [], [], 0, 0
    for lab, items in sorted(by_label.items()):
        if len(items) < min_images:
            dropped += 1
            continue
        items = items[:]; rng.shuffle(items)
        if gallery_shots is not None:
            n_gal = min(gallery_shots, len(items) - 1)   # at least 1 goes to probe
        else:
            n_gal = min(max(1, round(len(items) * gallery_frac)), len(items) - 1)
        gallery += items[:n_gal]; probe += items[n_gal:]; used += 1
    info = {"split": "single_shot" if gallery_shots == 1 else "random",
            "gallery_shots": gallery_shots, "n_ids_total": len(by_label), "n_ids_used": used,
            "n_ids_dropped": dropped, "n_gallery": len(gallery), "n_probe": len(probe)}
    return gallery, probe, info


def split_gallery_probe_by_session(entries: list[dict], seed: int = 0, min_sessions: int = 2,
                                   gallery_frac: float = 0.5,
                                   gallery_shots: int | None = None
                                   ) -> tuple[list[dict], list[dict], dict]:
    """Session-based split. `gallery_shots` = how many SESSIONS go to gallery per individual."""
    by_label: dict[int, dict[str, list[dict]]] = {}
    for e in entries:
        by_label.setdefault(e["label"], {}).setdefault(session_id(e["path"]), []).append(e)
    rng = random.Random(seed)
    gallery, probe, used, dropped, tot_sessions = [], [], 0, 0, 0
    for lab, sessions in sorted(by_label.items()):
        sids = list(sessions.keys())
        if len(sids) < min_sessions:
            dropped += 1
            continue
        rng.shuffle(sids)
        if gallery_shots is not None:
            n_gal = min(gallery_shots, len(sids) - 1)
        else:
            n_gal = min(max(1, round(len(sids) * gallery_frac)), len(sids) - 1)
        gal_sids = set(sids[:n_gal])
        for sid in sids:
            (gallery if sid in gal_sids else probe).extend(sessions[sid])
        used += 1; tot_sessions += len(sids)
    info = {"split": "by_session_single" if gallery_shots == 1 else "by_session",
            "gallery_shots": gallery_shots, "n_ids_total": len(by_label), "n_ids_used": used,
            "n_ids_dropped_lt_min_sessions": dropped, "min_sessions": min_sessions,
            "avg_sessions_per_id": round(tot_sessions / max(used, 1), 2),
            "n_gallery": len(gallery), "n_probe": len(probe)}
    return gallery, probe, info
