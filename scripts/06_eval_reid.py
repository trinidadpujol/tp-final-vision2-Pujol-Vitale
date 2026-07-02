"""06_eval_reid.py — Re-ID harness (Phase 6): sanity + gap + ImageNet baseline.

- `--source-dir`: intra-CMPD300 sanity check (validates the harness plumbing).
- `--target-dir`: gap on the target domain (Ahmed faces or Zenodo muzzles).
- `--by-session`: honest session-based split (for datasets with burst photos and
  timestamps, like Ahmed). Avoids matching twin photos from the same burst.
- `--single-shot`: 1 image (or session) per individual in gallery. Reduces leakage
  from look-alike photos — a single reference per individual makes it harder to match
  by photo similarity instead of biometrics.
- `--compare-imagenet`: also run plain ImageNet ResNet-50 on THE SAME split. If it
  matches your encoder, the number is not measuring muzzle recognition.

Usage:
    python scripts/06_eval_reid.py --source-dir .../train --target-dir .../zenodo \\
                                   --single-shot --compare-imagenet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.reid.embeddings import EmbeddingExtractor
from src.reid.eval_reid import rank_metrics
from src.reid.reid_dataset import (entries_from_folders, split_gallery_probe,
                                   split_gallery_probe_by_session)
from src.utils import get_logger, save_json


def score(extractor, gal, prb, root, batch_size):
    ge, gl = extractor.embed(gal, data_dir=root, batch_size=batch_size)
    pe, pl = extractor.embed(prb, data_dir=root, batch_size=batch_size)
    return rank_metrics(pe, pl, ge, gl)


def build_split(entries, args, by_session):
    shots = 1 if args.single_shot else None
    if by_session:
        return split_gallery_probe_by_session(entries, seed=args.seed,
                                              min_sessions=args.min_sessions, gallery_shots=shots)
    return split_gallery_probe(entries, seed=args.seed, min_images=args.min_images,
                               gallery_shots=shots)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-ID harness (Phase 6).")
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_DIR / "cmpd300_source.pt"))
    ap.add_argument("--source-dir", default=None)
    ap.add_argument("--target-dir", default=None)
    ap.add_argument("--by-session", action="store_true")
    ap.add_argument("--single-shot", action="store_true",
                    help="1 image/session per individual in gallery (reduces twin-photo leakage).")
    ap.add_argument("--compare-imagenet", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-images", type=int, default=2)
    ap.add_argument("--min-sessions", type=int, default=2)
    ap.add_argument("--max-per-id", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    log = get_logger("reid.eval")
    config.ensure_output_dirs()
    if not args.source_dir and not args.target_dir:
        log.error("Pass --source-dir and/or --target-dir."); sys.exit(1)

    source = EmbeddingExtractor.from_checkpoint(Path(args.ckpt))
    results = {"ckpt": args.ckpt, "single_shot": args.single_shot}

    # ---- SANITY intra-CMPD300 (random split) ----
    if args.source_dir:
        log.info("== SANITY intra-CMPD300 (SEEN identities → plumbing check) ==")
        entries, _ = entries_from_folders(Path(args.source_dir), max_per_id=args.max_per_id)
        gal, prb, info = split_gallery_probe(entries, seed=args.seed, min_images=args.min_images)
        m = score(source, gal, prb, Path(args.source_dir), args.batch_size)
        results["sanity_cmpd300"] = {**m, **info, "note": "leakage; only validates plumbing"}
        log.info(f"  source -> Rank-1={m['rank1']:.4f} mAP={m['mAP']:.4f} (expected HIGH)")

    # ---- GAP on the target (same split for all encoders) ----
    if args.target_dir:
        tag = "single-shot" if args.single_shot else "multi-shot"
        tag += " by session" if args.by_session else ""
        log.info(f"== RAW GAP on target — split {tag} ==")
        entries, _ = entries_from_folders(Path(args.target_dir), max_per_id=args.max_per_id)
        gal, prb, info = build_split(entries, args, args.by_session)
        log.info(f"  {info['n_ids_used']} individuals | gallery={info['n_gallery']} "
                 f"probe={info['n_probe']} | {info}")
        if not gal or not prb:
            log.error("gallery or probe is empty."); sys.exit(1)

        m_src = score(source, gal, prb, Path(args.target_dir), args.batch_size)
        results["gap_source"] = {**m_src, **info, "encoder": source.name}
        log.info(f"  source(muzzle) -> Rank-1={m_src['rank1']:.4f} mAP={m_src['mAP']:.4f}")

        if args.compare_imagenet:
            imagenet = EmbeddingExtractor.from_imagenet()
            m_in = score(imagenet, gal, prb, Path(args.target_dir), args.batch_size)
            results["gap_imagenet"] = {**m_in, "encoder": "imagenet_resnet50"}
            log.info(f"  imagenet(plain) -> Rank-1={m_in['rank1']:.4f} mAP={m_in['mAP']:.4f}")

    out = config.RESULTS_DIR / "06_reid_summary.json"
    save_json(results, out)
    log.info(f"summary saved to {out}")

    # ---- Human-readable summary ----
    print("\n" + "=" * 66)
    print("PHASE 6 — RE-ID" + ("  (SINGLE-SHOT)" if args.single_shot else "  (multi-shot)"))
    print("=" * 66)
    if "sanity_cmpd300" in results:
        s = results["sanity_cmpd300"]
        print(f"SANITY CMPD300 (plumbing)    : Rank-1={s['rank1']:.3f}  mAP={s['mAP']:.3f}")
    if "gap_source" in results:
        g = results["gap_source"]
        print(f"Target — muzzle encoder      : Rank-1={g['rank1']:.3f}  mAP={g['mAP']:.3f}"
              f"  ({g['n_ids_used']} ids, {g['n_probe']} probes)")
    if "gap_imagenet" in results:
        i = results["gap_imagenet"]
        print(f"Target — PLAIN ImageNet      : Rank-1={i['rank1']:.3f}  mAP={i['mAP']:.3f}")
        d = results["gap_source"]["rank1"] - i["rank1"]
        print("-" * 66)
        print(f"Muzzle encoder advantage over ImageNet: {d:+.3f} in Rank-1")
        if d < 0.05:
            print("WARNING: muzzle encoder does NOT improve over ImageNet, even with single-shot.")
            print("  The metric does not measure muzzle biometrics → data limitation.")
        else:
            print("OK: muzzle encoder improves over ImageNet: real muzzle signal present.")
    print("=" * 66)


if __name__ == "__main__":
    main()
