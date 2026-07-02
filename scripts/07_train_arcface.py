"""07_train_arcface.py — Encoder source con ArcFace (metric learning, Stage 2).

Diferencia clave con `05_train_source.py` (clasificacion softmax): ArcFace no aprende
a separar N clases fijas, sino que moldea el ESPACIO DE EMBEDDINGS con un margen
angular. Resultado: embeddings que generalizan mejor a IDENTIDADES NUEVAS, que es lo
que necesita el re-ID (y donde el encoder de clasificacion empataba con ImageNet).

Entrena ResNet-50 + ArcFace leyendo las CARPETAS de un dataset de hocico
(<root>/<ID>/*.jpg) directo — no necesita los split JSON de config. El feature usado es
el GAP de ResNet-50 (2048-d) L2-normalizado: el MISMO que consume EmbeddingExtractor.

El checkpoint se guarda como un ResNet-50 ESTANDAR (la cabeza ArcFace va en el slot de
`fc`, que el extractor descarta), asi que entra directo en `scripts/06_eval_reid.py`.

Uso:
    python scripts/07_train_arcface.py --train-dir ~/data/cmpd300/MuzzleSplit/train --epochs 40
    python scripts/07_train_arcface.py --train-dir ~/data/cmpd300/MuzzleSplit/train --smoke

Despues, re-correr los MISMOS 3 tests del diagnostico, apuntando al nuevo encoder:
    python scripts/06_eval_reid.py --ckpt outputs/checkpoints/cmpd300_arcface.pt \\
        --target-dir <CMPD300 test | Zenodo | morros-de-cara> --single-shot --compare-imagenet
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import config
from src.dataset import MuzzleDataset
from src.models import build_model
from src.reid.reid_dataset import entries_from_folders
from src.utils import get_device, get_logger, save_json, set_seed


# --------------------------------------------------------------------------- #
# ArcFace
# --------------------------------------------------------------------------- #
class ArcMarginProduct(nn.Module):
    """Cabeza ArcFace: coseno feature-clase con margen angular aditivo m, escala s."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)          # umbral de estabilidad
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(feat), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m                # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)   # estabilidad numerica
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return self.s * (one_hot * phi + (1.0 - one_hot) * cosine)


class ArcFaceModel(nn.Module):
    """ResNet-50 (backbone ImageNet, fine-tune completo) -> feature 2048-d -> ArcFace."""

    def __init__(self, num_classes: int, s: float, m: float, pretrained: bool = True,
                 init_from: str | None = None):
        super().__init__()
        base = build_model("resnet50", num_classes=num_classes, freeze_backbone=False,
                           pretrained=pretrained, init_from=init_from)
        self.feat_dim = base.fc.in_features       # 2048
        base.fc = nn.Identity()                   # backbone -> feature 2048-d
        self.backbone = base
        self.arc = ArcMarginProduct(self.feat_dim, num_classes, s=s, m=m)

    def forward(self, x, labels):
        return self.arc(self.backbone(x), labels)

    def export_state_dict(self, num_classes: int) -> dict:
        """State_dict de ResNet-50 ESTANDAR, compatible con EmbeddingExtractor.from_checkpoint.

        La cabeza ArcFace [num_classes, 2048] entra en fc.weight (el extractor descarta fc;
        se guarda solo para que el load_state_dict sea estricto y consistente en forma).
        """
        export = build_model("resnet50", num_classes=num_classes, freeze_backbone=False,
                             pretrained=False)
        sd = export.state_dict()
        sd.update(self.backbone.state_dict())     # conv/bn/layer entrenados (backbone.fc=Identity)
        sd["fc.weight"] = self.arc.weight.detach().cpu().clone()
        sd["fc.bias"] = torch.zeros(num_classes)
        return sd


def build_train_transform(image_size: int, use_imagenet_norm: bool):
    ops = [T.Resize((image_size, image_size)), T.RandomHorizontalFlip(),
           T.ColorJitter(brightness=0.2, contrast=0.2), T.ToTensor()]
    if use_imagenet_norm:
        ops.append(T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD))
    return T.Compose(ops)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", required=True, help="Carpeta <root>/<ID>/*.jpg (p.ej. CMPD300 train).")
    ap.add_argument("--out", default=str(config.CHECKPOINTS_DIR / "cmpd300_arcface.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--s", type=float, default=30.0, help="Escala de ArcFace.")
    ap.add_argument("--m", type=float, default=0.50, help="Margen angular de ArcFace.")
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE_S2)
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="Warm-start del backbone desde un checkpoint (p.ej. resnet50_backbone.pt).")
    ap.add_argument("--smoke", action="store_true", help="Prueba rapida: subset + 2 epocas + sin ImageNet.")
    args = ap.parse_args()

    set_seed(args.seed)
    config.ensure_output_dirs()
    log = get_logger("train.arcface")
    device = get_device()

    train_dir = Path(args.train_dir).expanduser()
    entries, id_map = entries_from_folders(train_dir)
    num_classes = len(id_map)
    if args.smoke:
        entries = entries[:256]
    epochs = 2 if args.smoke else args.epochs
    use_imagenet_norm = False if args.smoke else config.USE_IMAGENET_NORM_S2
    log.info(f"device={device} | train_dir={train_dir}")
    log.info(f"num_classes={num_classes} | imgs={len(entries)} | epochs={epochs} | "
             f"s={args.s} m={args.m} | image_size={args.image_size} | imagenet_norm={use_imagenet_norm}")

    tf = build_train_transform(args.image_size, use_imagenet_norm)
    ds = MuzzleDataset(entries, transform=tf, data_dir=train_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                        drop_last=True)

    model = ArcFaceModel(num_classes, s=args.s, m=args.m, pretrained=not args.smoke,
                         init_from=args.init_from).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, labels)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)
        scheduler.step()
        log.info(f"ep {epoch:02d}/{epochs} | loss {run_loss / max(1, n):.4f} | "
                 f"train acc {correct / max(1, n):.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

    ckpt = {
        "model_state": model.export_state_dict(num_classes),
        "model_name": "resnet50",
        "num_classes": num_classes,
        "run_config": {"image_size": args.image_size, "use_imagenet_norm": use_imagenet_norm},
        "method": "arcface", "arc_s": args.s, "arc_m": args.m,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)

    summary = {"encoder": "arcface", "num_classes": num_classes, "epochs": epochs,
               "arc_s": args.s, "arc_m": args.m, "image_size": args.image_size,
               "use_imagenet_norm": use_imagenet_norm, "checkpoint": str(out),
               "elapsed_sec": round(time.time() - t0, 1)}
    save_json(summary, config.RESULTS_DIR / "07_arcface_summary.json")
    log.info(f"encoder ArcFace guardado en {out}")
    log.info(f"summary: {summary}")
    log.info("Siguiente: re-correr los 3 tests con "
             f"`06_eval_reid.py --ckpt {out} --target-dir <...> --single-shot --compare-imagenet`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())