"""08_train_dann.py — Domain-Adversarial NN (DANN, Ganin et al. 2016) para re-ID de hocico.

Adaptado de MNIST->MNIST-M al caso cattle re-ID con IDENTIDADES DISJUNTAS:
  - SOURCE (CMPD300, etiquetado): el `task_classifier` sobre las identidades del source
    es un AUXILIAR que mantiene las features discriminativas. NO se usa en test
    (las vacas del target no son las del source).
  - TARGET (Zenodo / morros-de-cara, SIN etiquetas): solo se usa para la alineacion
    adversarial de dominios (domain_classifier + GRL).
  - Encoder: ResNet-50 (ImageNet) -> GAP 2048 -> Linear(2048, feat_dim) -> BN.
    El feature `feat_dim`-d (L2-norm) es el embedding usado en el re-ID.
  - EVAL: gallery/probe sobre el target (cross-sesion, single-shot) con `get_features`,
    comparado contra ImageNet puro en el MISMO split (como en 06_eval_reid).

Como el target tiene identidades disjuntas NO hay "target task acc" que monitorear
(a diferencia de MNIST-M con clases compartidas). Por eso se evalua el RE-ID cada
`--eval-every` epocas y se reporta la MEJOR epoca ademas de la final: si la mejor es
temprana, hay sobreajuste-al-source por epoca (oracle model selection: aclarar en el
informe que usa labels del target para elegir epoca).

Ablation: --mode baseline entrena lo mismo con lam=0 (sin adaptacion).

Uso:
    python scripts/08_train_dann.py --target-dir ~/data/cows_face_muzzle --mode dann \\
        --warmup-epochs 5 --lam-max 0.5 --feat-dim 512 --dropout 0.1 --gamma 5 --eval-every 5
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import config
from src.dataset import MuzzleDataset
from src.models import build_model
from src.reid.embeddings import EmbeddingExtractor
from src.reid.eval_reid import rank_metrics
from src.reid.reid_dataset import (entries_from_folders, split_gallery_probe,
                                   split_gallery_probe_by_session)
from src.transforms import build_transforms
from src.utils import get_device, get_logger, save_json, set_seed


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer (Ganin et al. 2016)
# --------------------------------------------------------------------------- #
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam):
    return GradientReversalFunction.apply(x, lam)


# --------------------------------------------------------------------------- #
# Arquitectura
# --------------------------------------------------------------------------- #
class ResNetEncoder(nn.Module):
    """ResNet-50 (ImageNet) -> GAP 2048 -> Linear(2048, feat_dim) -> BN. Feature de re-ID."""

    def __init__(self, feat_dim: int = 256, pretrained: bool = True, init_from: str | None = None):
        super().__init__()
        base = build_model("resnet50", num_classes=2, freeze_backbone=False,
                           pretrained=pretrained, init_from=init_from)
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(2048, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)

    def forward(self, x):
        return self.bn(self.proj(self.backbone(x)))


class DANN(nn.Module):
    """Encoder compartido + task predictor (source) + domain predictor (GRL)."""

    def __init__(self, encoder: nn.Module, n_classes: int, feat_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.task_classifier = nn.Sequential(
            nn.Linear(feat_dim, 100), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(100, n_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 100), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(100, 2),
        )

    def forward(self, x, lam: float = 1.0):
        features = self.encoder(x)
        class_logits = self.task_classifier(features)
        domain_logits = self.domain_classifier(grad_reverse(features, lam))
        return class_logits, domain_logits

    @torch.no_grad()
    def get_features(self, x):
        return F.normalize(self.encoder(x), dim=1)


def train_transform(image_size: int):
    return T.Compose([T.Resize((image_size, image_size)), T.RandomHorizontalFlip(),
                      T.ColorJitter(brightness=0.2, contrast=0.2), T.ToTensor(),
                      T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)])


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def embed(model: DANN, entries, data_dir, tf, device, bs, nw):
    ds = MuzzleDataset(entries, transform=tf, data_dir=Path(data_dir))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    model.eval()
    embs, labs = [], []
    for imgs, labels in loader:
        embs.append(model.get_features(imgs.to(device)).cpu().numpy())
        labs.append(np.asarray(labels))
    return np.concatenate(embs), np.concatenate(labs)


def build_target_split(target_dir, args):
    entries, _ = entries_from_folders(Path(target_dir))
    shots = args.gallery_shots
    if args.split == "by_session":
        return split_gallery_probe_by_session(entries, seed=args.seed,
                                              min_sessions=args.min_sessions, gallery_shots=shots)
    return split_gallery_probe(entries, seed=args.seed, min_images=args.min_images, gallery_shots=shots)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-dir", default="~/data/cmpd300/MuzzleSplit/train")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--mode", choices=["dann", "baseline"], default="dann")
    ap.add_argument("--feat-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=10.0, help="Ganin: pendiente del schedule de lam.")
    ap.add_argument("--lam-max", type=float, default=1.0, help="Techo de lam (bajalo si el task no aprende).")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Epocas iniciales con lam=0 (el encoder aprende la tarea antes de lo adversarial).")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="Evaluar el re-ID en el target cada N epocas (detecta overfit-al-source).")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE_S2)
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["by_session", "random"], default="by_session")
    ap.add_argument("--gallery-shots", type=int, default=1)
    ap.add_argument("--min-sessions", type=int, default=2)
    ap.add_argument("--min-images", type=int, default=2)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    config.ensure_output_dirs()
    log = get_logger("train.dann")
    device = get_device()
    source_dir, target_dir = Path(args.source_dir).expanduser(), Path(args.target_dir).expanduser()

    # ---- datos ----
    src_entries, id_map = entries_from_folders(source_dir)
    tgt_entries, _ = entries_from_folders(target_dir)
    if args.smoke:
        src_entries, tgt_entries = src_entries[:256], tgt_entries[:256]
    n_classes = len(id_map)
    epochs = 2 if args.smoke else args.epochs
    tf = train_transform(args.image_size)
    src_loader = DataLoader(MuzzleDataset(src_entries, transform=tf, data_dir=source_dir),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(), drop_last=True)
    tgt_loader = DataLoader(MuzzleDataset(tgt_entries, transform=tf, data_dir=target_dir),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(), drop_last=True)
    tgt_iter = infinite(tgt_loader)
    log.info(f"mode={args.mode} | device={device} | n_classes={n_classes} | "
             f"src_imgs={len(src_entries)} tgt_imgs={len(tgt_entries)} | epochs={epochs}")

    # ---- modelo ----
    encoder = ResNetEncoder(feat_dim=args.feat_dim, pretrained=not args.smoke, init_from=args.init_from)
    model = DANN(encoder, n_classes=n_classes, feat_dim=args.feat_dim, dropout=args.dropout).to(device)
    ce = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # ---- split del target + baseline ImageNet (constantes, para la eval periodica) ----
    gal, prb, info = build_target_split(target_dir, args)
    if not gal or not prb:
        log.error("gallery o probe vacio; proba --split random o bajar --min-sessions.")
        return 1
    eval_tf = build_transforms(train=False, image_size=args.image_size, use_imagenet_norm=True)
    imagenet = EmbeddingExtractor.from_imagenet(device=device)
    ige, igl = imagenet.embed(gal, data_dir=target_dir, batch_size=args.batch_size)
    ipe, ipl = imagenet.embed(prb, data_dir=target_dir, batch_size=args.batch_size)
    m_in = rank_metrics(ipe, ipl, ige, igl)
    log.info(f"target: {info['n_ids_used']} ids | gallery={info['n_gallery']} probe={info['n_probe']} "
             f"| ImageNet Rank-1={m_in['rank1']:.3f} mAP={m_in['mAP']:.3f}")

    def eval_reid():
        ge, gl = embed(model, gal, target_dir, eval_tf, device, args.batch_size, args.num_workers)
        pe, pl = embed(model, prb, target_dir, eval_tf, device, args.batch_size, args.num_workers)
        return rank_metrics(pe, pl, ge, gl)

    steps = epochs * len(src_loader)
    step = 0
    best = {"rank1": -1.0, "epoch": -1}
    m_last = None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        task_ok = dom_ok = n = 0
        for src_imgs, src_labels in src_loader:
            p = step / max(1, steps)
            if args.mode == "baseline" or epoch <= args.warmup_epochs:
                lam = 0.0
            else:
                wsteps = args.warmup_epochs * len(src_loader)          # progreso adversarial post-warmup
                pa = (step - wsteps) / max(1, steps - wsteps)
                lam = args.lam_max * (2.0 / (1.0 + math.exp(-args.gamma * pa)) - 1.0)
            for g in optim.param_groups:                               # Ganin: lr / (1 + 10 p)^0.75
                g["lr"] = args.lr / (1.0 + 10.0 * p) ** 0.75

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs, _ = next(tgt_iter)
            tgt_imgs = tgt_imgs.to(device)

            src_task, src_dom = model(src_imgs, lam)
            _, tgt_dom = model(tgt_imgs, lam)
            task_loss = ce(src_task, src_labels)
            dom_labels = torch.cat([torch.zeros(src_imgs.size(0), dtype=torch.long),
                                    torch.ones(tgt_imgs.size(0), dtype=torch.long)]).to(device)
            dom_loss = ce(torch.cat([src_dom, tgt_dom]), dom_labels)
            loss = task_loss + dom_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            task_ok += (src_task.argmax(1) == src_labels).sum().item()
            dom_ok += (torch.cat([src_dom, tgt_dom]).argmax(1) == dom_labels).sum().item()
            n += src_imgs.size(0)
            step += 1
        log.info(f"ep {epoch:02d}/{epochs} | lam {lam:.3f} | task_acc {task_ok / max(1, n):.4f} | "
                 f"dom_acc {dom_ok / max(1, 2 * n):.4f} (0.5=alineado) | lr {optim.param_groups[0]['lr']:.2e}")

        # ---- eval periodica del re-ID en el target (detecta overfit-al-source por epoca) ----
        if epoch % args.eval_every == 0 or epoch == epochs:
            m = eval_reid()
            m_last = {**m, "epoch": epoch}
            mark = "  <== MEJOR" if m["rank1"] > best["rank1"] else ""
            log.info(f"   [eval ep{epoch:02d}] DANN Rank-1={m['rank1']:.3f} mAP={m['mAP']:.3f} | "
                     f"vs ImageNet {m_in['rank1']:.3f} ({m['rank1'] - m_in['rank1']:+.3f}){mark}")
            if m["rank1"] > best["rank1"]:
                best = {**m, "epoch": epoch}

    # ---- resultados: MEJOR epoca vs FINAL vs ImageNet ----
    summary = {"mode": args.mode, "target": str(target_dir), "n_classes": n_classes, "epochs": epochs,
               "feat_dim": args.feat_dim, "lam_max": args.lam_max, "warmup_epochs": args.warmup_epochs,
               **info, "imagenet": m_in, "dann_best": best, "dann_final": m_last,
               "best_beats_final": best["epoch"] != epochs, "elapsed_sec": round(time.time() - t0, 1)}
    save_json(summary, config.RESULTS_DIR / f"08_dann_{args.mode}_summary.json")

    print("\n" + "=" * 68)
    print(f"PHASE — DANN re-ID   (mode={args.mode.upper()}, target={target_dir.name})")
    print("=" * 68)
    print(f"PLAIN ImageNet                : Rank-1={m_in['rank1']:.3f}  mAP={m_in['mAP']:.3f}")
    print(f"DANN — MEJOR epoca (ep{best['epoch']:02d})    : Rank-1={best['rank1']:.3f}  mAP={best['mAP']:.3f}"
          f"   ({best['rank1'] - m_in['rank1']:+.3f} vs ImageNet)")
    print(f"DANN — epoca FINAL (ep{m_last['epoch']:02d})    : Rank-1={m_last['rank1']:.3f}  mAP={m_last['mAP']:.3f}"
          f"   ({m_last['rank1'] - m_in['rank1']:+.3f} vs ImageNet)")
    print("-" * 68)
    if best["epoch"] != epochs:
        print(f"OJO: la mejor epoca ({best['epoch']}) NO es la final ({epochs}) -> "
              f"sobreajuste-al-source por epoca (tu hipotesis).")
    if best["rank1"] - m_in["rank1"] > 0.05:
        print("La MEJOR epoca supera a ImageNet -> la adaptacion aporta (en su mejor punto).")
    else:
        print("Ni la mejor epoca supera a ImageNet -> consistente con limite de dominio/data.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())