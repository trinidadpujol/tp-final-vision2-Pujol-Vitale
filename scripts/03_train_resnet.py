"""03_train_resnet.py — Phase 4: own ResNet-50 backbone.

PAPER: same recipe as the VGG replication (300x300, raw [0,1], SGD mom=0.9,
lr=0.001, StepLR(7, 0.1), 50 epochs). ResNet-50 is NOT the model that replicates
the 98.7% (that is VGG16_BN); we train it as our own backbone for reuse in the
future domain adaptation stage.

Runs TWO modes (see plan.md §Phase 4):
  - freeze   : freeze_backbone=True  → FC only, as in the paper.
  - finetune : freeze_backbone=False → full fine-tuning.

For each (mode × seed): train, evaluate on test (global + balanced + per-class) and
aggregate mean ± std. The BEST run (by val accuracy, without looking at test → no
leakage) is copied to a canonical checkpoint in outputs/checkpoints/ — that is the
backbone reused by domain adaptation.

OUR CHOICE (plan leaves this open): default loss=ce, no augmentation and 1 seed
(this is our backbone, not the paper's replication number). All overrideable by flags
if a stronger backbone is desired (e.g. --aug).

Usage:
    python scripts/03_train_resnet.py                      # freeze + finetune, seed 0, 50 epochs
    python scripts/03_train_resnet.py --modes freeze       # paper mode only
    python scripts/03_train_resnet.py --aug --loss wce     # stronger backbone
    python scripts/03_train_resnet.py --seeds 0 1 2        # multiple seeds → mean±std
    python scripts/03_train_resnet.py --smoke              # quick pipeline on CPU (subset, 2 epochs)
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.evaluate import evaluate_checkpoint  # noqa: E402
from src.train import RunConfig, train_one_run  # noqa: E402
from src.utils import get_device, get_logger, load_json, save_json  # noqa: E402

MODEL = "resnet50"

# mode -> freeze_backbone.
MODES = {
    "freeze":   True,    # PAPER: FC only.
    "finetune": False,   # OUR CHOICE: full fine-tune (better backbone for DA).
}

# IDs of the 8 classes with 4 images (see DEVIATIONS.md). Focused report: do not
# hide the worst case behind the average (integrity principle from CLAUDE.md).
SMALL_CLASSES = ["cattle_2100", "cattle_3420", "cattle_4549", "cattle_5208",
                 "cattle_5355", "cattle_5630", "cattle_5925", "cattle_8050"]


def _small_class_accs(ev: dict) -> dict[str, float | None]:
    """Test accuracy for the 8 classes with 4 images (None if no samples)."""
    try:
        label_map = load_json(config.SPLITS_DIR / "label_map.json")  # name -> idx
    except FileNotFoundError:
        return {}
    out: dict[str, float | None] = {}
    for name in SMALL_CLASSES:
        idx = label_map.get(name)
        if idx is None:
            continue
        n = ev["per_class_total"][idx]
        a = ev["per_class_acc"][idx]
        out[name] = None if (n == 0 or a != a) else round(a, 4)  # a!=a → NaN
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()),
                    choices=list(MODES.keys()),
                    help="freeze (paper) and/or finetune (full fine-tune).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Seeds. Default: 1 only (this is the backbone, not the replication).")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--loss", default="ce", choices=["ce", "wce"],
                    help="Base loss. Default ce (paper's base recipe).")
    ap.add_argument("--aug", action="store_true",
                    help="Data augmentation in train (off by default).")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick pipeline: 1 seed, 2 epochs, small subset, no ImageNet weights.")
    args = ap.parse_args()

    log = get_logger("03_train_resnet")
    device = get_device()
    config.ensure_output_dirs()

    # Smoke: validate the end-to-end pipeline, not the science. Smaller subset than
    # VGG because full fine-tuning of ResNet-50 on CPU is heavy.
    smoke = args.smoke
    seeds = [0] if smoke else args.seeds
    epochs = 2 if smoke else args.epochs
    max_train = 16 if smoke else None
    max_val = 8 if smoke else None
    max_test = 8 if smoke else None
    pretrained = not smoke  # avoid downloading 98 MB of ImageNet weights in smoke mode

    log.info(f"device={device} | model={MODEL} | modes={args.modes} | "
             f"loss={args.loss} | aug={args.aug} | seeds={seeds} | epochs={epochs} | smoke={smoke}")

    results: dict[str, list[dict]] = {m: [] for m in args.modes}
    # Best overall run by VAL accuracy (selection without test leakage) → canonical backbone.
    best_overall: dict | None = None

    for mode in args.modes:
        freeze = MODES[mode]
        for seed in seeds:
            tag = (f"{MODEL}_{mode}_{args.loss}_{'aug' if args.aug else 'noaug'}_s{seed}"
                   + ("_smoke" if smoke else ""))
            rc = RunConfig(
                model_name=MODEL, loss_kind=args.loss, use_aug=args.aug,
                seed=seed, freeze_backbone=freeze, epochs=epochs, pretrained=pretrained,
                max_train=max_train, max_val=max_val,
                num_workers=0 if smoke else config.NUM_WORKERS,
                tag=tag,
            )
            run = train_one_run(rc, device=device)
            csv_path = config.RESULTS_DIR / f"perclass_{rc.tag}.csv"
            ev = evaluate_checkpoint(run["checkpoint"], device=device,
                                     max_test=max_test, save_csv=csv_path)
            small = _small_class_accs(ev)
            log.info(f"[{mode} s{seed}] val={run['best_val_acc']:.4f} "
                     f"test_global={ev['global_acc']:.4f} test_balanced={ev['balanced_acc']:.4f} "
                     f"| {ev['ms_per_image']} ms/img")
            if small:
                log.info(f"[{mode} s{seed}] small-class acc (4 imgs): {small}")

            entry = {
                "mode": mode,
                "seed": seed,
                "freeze_backbone": freeze,
                "best_val_acc": run["best_val_acc"],
                "test_global_acc": ev["global_acc"],
                "test_balanced_acc": ev["balanced_acc"],
                "ms_per_image": ev["ms_per_image"],
                "small_class_acc": small,
                "checkpoint": run["checkpoint"],
                "perclass_csv": str(csv_path),
            }
            results[mode].append(entry)
            # Select by val acc (NOT by test → no leakage).
            if best_overall is None or run["best_val_acc"] > best_overall["best_val_acc"]:
                best_overall = entry

    # ---- Mean ± std summary per mode ----
    summary = {}
    for mode, runs in results.items():
        gaccs = [r["test_global_acc"] for r in runs]
        baccs = [r["test_balanced_acc"] for r in runs]
        summary[mode] = {
            "n_runs": len(runs),
            "test_global_mean": round(statistics.mean(gaccs), 4),
            "test_global_std": round(statistics.pstdev(gaccs), 4) if len(gaccs) > 1 else 0.0,
            "test_balanced_mean": round(statistics.mean(baccs), 4),
            "test_balanced_std": round(statistics.pstdev(baccs), 4) if len(baccs) > 1 else 0.0,
            "runs": runs,
        }

    # ---- Canonical backbone: copy the best checkpoint (by val acc) ----
    backbone_name = "resnet50_backbone_smoke.pt" if smoke else "resnet50_backbone.pt"
    backbone_path = config.CHECKPOINTS_DIR / backbone_name
    shutil.copyfile(best_overall["checkpoint"], backbone_path)
    log.info(f"Canonical backbone (best by val acc: {best_overall['mode']} "
             f"s{best_overall['seed']}, val={best_overall['best_val_acc']:.4f}) → {backbone_path}")

    out = config.RESULTS_DIR / ("03_resnet_summary_smoke.json" if smoke else "03_resnet_summary.json")
    save_json({
        "model": MODEL, "epochs": epochs, "seeds": seeds, "loss": args.loss,
        "aug": args.aug, "device": device, "smoke": smoke,
        "backbone_checkpoint": str(backbone_path),
        "best_run": {k: best_overall[k] for k in
                     ("mode", "seed", "best_val_acc", "test_global_acc", "test_balanced_acc")},
        "summary": summary,
    }, out)

    # ---- Report ----
    print("\n" + "=" * 70)
    print(f"OWN BACKBONE SUMMARY — {MODEL}" + (" [SMOKE]" if smoke else ""))
    print("=" * 70)
    print(f"{'mode':10} | {'test global (mean±std)':24} | {'balanced (mean±std)':22}")
    print("-" * 70)
    for m in args.modes:
        s = summary[m]
        print(f"{m:10} | {s['test_global_mean']:.4f} ± {s['test_global_std']:.4f}        "
              f"| {s['test_balanced_mean']:.4f} ± {s['test_balanced_std']:.4f}")
    print("=" * 70)
    print(f"Best run (by val acc): {best_overall['mode']} s{best_overall['seed']} "
          f"| val={best_overall['best_val_acc']:.4f} test={best_overall['test_global_acc']:.4f}")
    print(f"Domain adaptation backbone: {backbone_path}")
    print(f"Summary: {out}")
    if not smoke:
        print("\nNote: ResNet-50 does NOT replicate the paper's 98.7% (that is VGG16_BN). Here "
              "we care about backbone quality for reuse in domain adaptation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
