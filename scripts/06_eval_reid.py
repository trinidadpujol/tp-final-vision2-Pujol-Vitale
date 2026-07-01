"""06_eval_reid.py — Harness de re-ID (Fase 6): sanity intra-CMPD300 + gap hocico→cara.

Dos mediciones con el encoder source (`cmpd300_source.pt`):

1. SANITY intra-CMPD300 (`--source-dir <CMPD300>/train`): gallery/probe sobre las propias
   vacas de CMPD300. Las identidades fueron VISTAS en el entrenamiento → hay leakage, así que
   el número NO es reportable: es solo para validar que el harness da Rank-1 alto en territorio
   conocido (plomería). Si acá da bajo, hay un bug.

2. GAP crudo hocico→cara (`--target-dir <caras_Ahmed>`): gallery/probe sobre las caras de
   Ahmed (identidades nuevas, disjuntas). El encoder de hocico se evalúa sobre caras enteras,
   SIN adaptar. Este SÍ es el número reportable: el punto de partida que la domain adaptation
   tiene que mejorar.

Uso:
    python scripts/06_eval_reid.py --source-dir /content/cmpd300/Baseline/train \\
                                   --target-dir /content/ahmed_subset \\
                                   --max-per-id 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.reid.embeddings import EmbeddingExtractor
from src.reid.eval_reid import rank_metrics
from src.reid.reid_dataset import entries_from_folders, split_gallery_probe
from src.utils import get_logger, save_json


def evaluate_folder(extractor: EmbeddingExtractor, root: Path, *, seed: int,
                    min_images: int, max_per_id: int | None, batch_size: int,
                    log) -> dict:
    """Arma gallery/probe desde <root>/<id>/* , extrae embeddings y calcula métricas."""
    entries, id_map = entries_from_folders(root, max_per_id=max_per_id)
    gal, prb, info = split_gallery_probe(entries, seed=seed, min_images=min_images)
    log.info(f"  {root}: {len(id_map)} individuos, {len(entries)} imgs | "
             f"gallery={info['n_gallery']} probe={info['n_probe']} "
             f"(usados {info['n_ids_used']} ids, descartados {info['n_ids_dropped_lt_min']})")
    if not gal or not prb:
        raise RuntimeError("gallery o probe vacíos: ¿pocas imágenes por individuo?")
    ge, gl = extractor.embed(gal, data_dir=root, batch_size=batch_size)
    pe, pl = extractor.embed(prb, data_dir=root, batch_size=batch_size)
    m = rank_metrics(pe, pl, ge, gl)
    return {**m, **info}


def main() -> None:
    ap = argparse.ArgumentParser(description="Harness de re-ID (Fase 6).")
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "cmpd300_source.pt"))
    ap.add_argument("--source-dir", default=None,
                    help="carpeta <id>/* de CMPD300 (p.ej. .../Baseline/train) para el sanity.")
    ap.add_argument("--target-dir", default=None,
                    help="carpeta <id>/* de las caras de Ahmed para el gap crudo.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-images", type=int, default=2)
    ap.add_argument("--max-per-id", type=int, default=None,
                    help="cortar a N imágenes por individuo (velocidad).")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    log = get_logger("reid.eval")
    config.ensure_output_dirs()
    if not args.source_dir and not args.target_dir:
        log.error("Pasá al menos --source-dir (sanity) o --target-dir (gap).")
        sys.exit(1)

    extractor = EmbeddingExtractor(Path(args.ckpt))
    results: dict = {"ckpt": args.ckpt}

    if args.source_dir:
        log.info("== SANITY intra-CMPD300 (identidades VISTAS → plomería, NO reportable) ==")
        r = evaluate_folder(extractor, Path(args.source_dir), seed=args.seed,
                            min_images=args.min_images, max_per_id=args.max_per_id,
                            batch_size=args.batch_size, log=log)
        r["nota"] = "identidades vistas en train (leakage); solo valida el harness"
        results["sanity_cmpd300"] = r
        log.info(f"  -> Rank-1={r['rank1']:.4f} Rank-5={r['rank5']:.4f} mAP={r['mAP']:.4f} "
                 f"(esperado ALTO)")

    if args.target_dir:
        log.info("== GAP crudo hocico→cara sobre Ahmed (SIN adaptar → REPORTABLE) ==")
        r = evaluate_folder(extractor, Path(args.target_dir), seed=args.seed,
                            min_images=args.min_images, max_per_id=args.max_per_id,
                            batch_size=args.batch_size, log=log)
        r["nota"] = "gap crudo cross-modality, encoder de hocico sobre caras, sin adaptar"
        results["gap_ahmed_wholeface"] = r
        log.info(f"  -> Rank-1={r['rank1']:.4f} Rank-5={r['rank5']:.4f} mAP={r['mAP']:.4f}")

    out = config.RESULTS_DIR / "06_reid_summary.json"
    save_json(results, out)
    log.info(f"resumen guardado en {out}")

    # resumen legible
    print("\n" + "=" * 62)
    print("FASE 6 — RE-ID")
    print("=" * 62)
    if "sanity_cmpd300" in results:
        s = results["sanity_cmpd300"]
        print(f"SANITY CMPD300 (plomería) : Rank-1={s['rank1']:.3f}  mAP={s['mAP']:.3f}")
    if "gap_ahmed_wholeface" in results:
        g = results["gap_ahmed_wholeface"]
        print(f"GAP crudo hocico→cara     : Rank-1={g['rank1']:.3f}  Rank-5={g['rank5']:.3f}  mAP={g['mAP']:.3f}")
        print(f"  (sobre {g['n_ids_used']} individuos de Ahmed, {g['n_probe']} probes)")
    print("=" * 62)


if __name__ == "__main__":
    main()
