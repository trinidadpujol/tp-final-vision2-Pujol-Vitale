"""04_precompute_aug.py — Precomputa las imágenes sintéticas (augmentation) a disco.

PAPER: la data augmentation es un PASO DE PREPROCESAMIENTO que "crea imágenes sintéticas
y agranda el dataset". Acá las generamos UNA vez y las guardamos como archivos, en vez de
re-aumentar online en cada época. Ventajas: el conjunto sintético queda FIJO (fiel al
paper), reproducible e inspeccionable (podés abrir las .jpg y verlas).

Por clase se agregan `max(0, min(AUG_TARGET_CAP, AUG_FACTOR*N_i) - N_i)` copias,
muestreando con reemplazo de las imágenes de esa clase (seed fijo) y aplicando
flip / brillo (0.2–0.5) / rotación (±15°) / blur. Mismo criterio que el path online
(`dataset.build_augmented_entries`), solo que acá queda materializado.

Salida (en OUTPUT/aug_cache/):
  <class_name>/<orig_stem>__aug<k>.jpg   # imágenes sintéticas (300x300)
  aug_manifest.json                      # [{"path", "label"}] relativo a aug_cache/

El entrenamiento usa este cache automáticamente si existe (ver dataset.make_train_loader).

NOTA: NO acelera el entrenamiento (el cuello de botella es el forward en GPU, que procesa
la misma cantidad de imágenes). Sirve para fijar y reproducir el set sintético.

Uso:
    python scripts/04_precompute_aug.py            # genera si no existe
    python scripts/04_precompute_aug.py --force    # regenera
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
                    help="Semilla del muestreo + ops aleatorias (reproducibilidad).")
    ap.add_argument("--force", action="store_true", help="Regenerar aunque ya exista.")
    args = ap.parse_args()

    log = get_logger("04_precompute_aug")
    config.ensure_output_dirs()
    cache = config.AUG_CACHE_DIR
    manifest_path = cache / "aug_manifest.json"

    if manifest_path.is_file() and not args.force:
        m = load_json(manifest_path)
        log.info(f"Ya existe {manifest_path} ({len(m)} imágenes). Usar --force para regenerar.")
        return 0

    cache.mkdir(parents=True, exist_ok=True)
    label_map = load_json(config.SPLITS_DIR / "label_map.json")  # name -> idx
    idx_to_name = {v: k for k, v in label_map.items()}

    train = load_split("train")
    extra = build_augmented_entries(train, seed=args.seed)
    log.info(f"Generando {len(extra)} imágenes sintéticas en {cache} ...")

    aug_tf = build_pil_aug_transform()  # PIL ops (resize + aug), sin ToTensor
    # Determinismo de las ops aleatorias de torchvision (usan el RNG de torch) + del muestreo.
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
        out_img = aug_tf(img)  # PIL 300x300 aumentada

        k = counters[(label, stem)]
        counters[(label, stem)] += 1
        rel = f"{cls}/{stem}__aug{k}.jpg"
        (cache / cls).mkdir(parents=True, exist_ok=True)
        out_img.save(cache / rel, quality=95)
        manifest.append({"path": rel, "label": label})

    save_json(manifest, manifest_path)
    log.info(f"Listo: {len(manifest)} imágenes sintéticas + manifest en {manifest_path}")
    log.info("El entrenamiento las usará automáticamente (dataset.make_train_loader).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
