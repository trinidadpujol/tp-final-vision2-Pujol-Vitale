"""00_inspect_data.py — Fase 0: inspección y validación del dataset.

Correr SIEMPRE primero. Confirma la estructura real del dataset antes de escribir
nada de entrenamiento. No avanzar a Fase 1 hasta que el reporte cuadre.

Reporta: nº de clases, total de imágenes, min/max/media por clase, histograma,
extensiones presentes, imágenes corruptas/ilegibles, y las clases más chicas.
Compara contra los sanity checks de config.py (268 clases, 4923 imágenes, 8–140).

Uso:
    python scripts/00_inspect_data.py
    python scripts/00_inspect_data.py --check-corrupt   # abre cada imagen (lento)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Permitir importar config.py y src/ corriendo desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_class_dirs(data_dir: Path) -> list[Path]:
    """Subcarpetas que representan una clase (un animal)."""
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def count_images(class_dirs: list[Path]) -> tuple[dict[str, int], Counter]:
    """Devuelve {clase: nº imágenes} y un Counter de extensiones."""
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
    """Intenta abrir y verificar cada imagen; devuelve rutas ilegibles."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow no instalado: salteo chequeo de corruptas)")
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
    """Histograma simple (ASCII) de imágenes por clase."""
    counts = list(per_class.values())
    lo, hi = min(counts), max(counts)
    width = max(1, (hi - lo) // bins + 1)
    edges = list(range(lo, hi + width, width))
    hist: list[tuple[str, int]] = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        c = sum(1 for v in counts if a <= v < b)
        hist.append((f"[{a:>3}-{b:>3})", c))
    # incluir el borde superior
    hist.append((f"[{edges[-1]:>3}+   )", sum(1 for v in counts if v >= edges[-1])))
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-corrupt", action="store_true",
                    help="Abrir y verificar cada imagen (lento).")
    args = ap.parse_args()

    log = get_logger("inspect")
    data_dir = config.DATA_DIR
    log.info(f"DATA_DIR = {data_dir}")

    if not data_dir.is_dir():
        log.error(f"No existe DATA_DIR. Setear CATTLE_DATA_DIR o extraer el dataset. "
                  f"Esperado: {data_dir}")
        return 1

    class_dirs = list_class_dirs(data_dir)
    per_class, ext_counter = count_images(class_dirs)
    total = sum(per_class.values())
    counts = list(per_class.values())

    n_classes = len(class_dirs)
    n_min, n_max = min(counts), max(counts)
    n_mean = total / n_classes if n_classes else 0.0

    print("\n" + "=" * 60)
    print("REPORTE DE INSPECCIÓN — Fase 0")
    print("=" * 60)
    print(f"Nº de clases (carpetas)   : {n_classes}")
    print(f"Total de imágenes         : {total}")
    print(f"Imágenes por clase  min   : {n_min}")
    print(f"                    max   : {n_max}")
    print(f"                    media : {n_mean:.1f}")
    print(f"Extensiones presentes     : {dict(ext_counter)}")

    print("\nHistograma (imágenes por clase):")
    for label, c in histogram(per_class):
        print(f"  {label} | {'#' * c} ({c})")

    print("\n5 clases con MENOS imágenes:")
    for name, c in sorted(per_class.items(), key=lambda x: x[1])[:5]:
        print(f"  {name}: {c}")
    print("5 clases con MÁS imágenes:")
    for name, c in sorted(per_class.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {c}")

    # ---- Sanity checks contra config.py ----
    print("\n" + "-" * 60)
    print("SANITY CHECKS (contra config.py)")
    print("-" * 60)
    checks = [
        ("nº de clases", n_classes, config.NUM_CLASSES),
        ("total imágenes", total, config.EXPECTED_IMAGES),
        ("min imgs/clase", n_min, config.MIN_IMAGES_PER_CLASS),
        ("max imgs/clase", n_max, config.MAX_IMAGES_PER_CLASS),
    ]
    all_ok = True
    for label, got, exp in checks:
        ok = got == exp
        all_ok &= ok
        print(f"  [{'OK ' if ok else 'MISMATCH'}] {label}: got={got} esperado={exp}")

    corrupt: list[str] = []
    if args.check_corrupt:
        print("\nVerificando imágenes corruptas (esto tarda)...")
        corrupt = find_corrupt(class_dirs)
        print(f"  Imágenes corruptas/ilegibles: {len(corrupt)}")
        for c in corrupt[:10]:
            print(f"    {c}")

    # ---- Guardar reporte ----
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
    print(f"\nReporte guardado en: {out}")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULTADO: sanity checks OK. Listo para Fase 1.")
    else:
        print("RESULTADO: hay MISMATCHES. Revisar config.py / DEVIATIONS.md "
              "antes de avanzar.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
