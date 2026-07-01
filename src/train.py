"""train.py — Training + validation loop for ONE run (1 model, 1 recipe).

Paper recipe (see plan.md / config.py):
- SGD(momentum=0.9), lr=0.001, StepLR(step_size=7, gamma=0.1), 50 epochs.
- Only trainable parameters are optimized (with freeze_backbone, only the FC layers).
- Val accuracy is tracked per epoch and the BEST checkpoint by val acc is saved.

`train_one_run` is the unit reused by the Phase 3/4/5 scripts. Supports subsetting
(max_train/max_val) and reducing epochs for fast pipeline smoke-tests.

Stage 2 (additive, backward-compatible): RunConfig accepts `data_dir`, `splits_dir`,
`num_classes`, `image_size`, and `use_precomputed_aug`. With all of them at None/default,
the behavior is EXACTLY that of Stage 1 (paper dataset). Passing them allows training on
another dataset (e.g. CMPD300) reusing this same loop, without reimplementing it.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

import config
from src.dataset import load_split, make_dataloader, make_train_loader
from src.losses import build_loss
from src.models import build_model, count_parameters, trainable_parameters
from src.transforms import build_transforms
from src.utils import get_device, get_logger, save_json, set_seed


@dataclass
class RunConfig:
    """Configuration for one run (logged and saved with the checkpoint)."""
    model_name: str
    loss_kind: str = "ce"          # 'ce' | 'wce'
    use_aug: bool = False          # data augmentation in train
    seed: int = 0
    freeze_backbone: bool = config.FREEZE_BACKBONE
    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH_SIZE
    num_workers: int = config.NUM_WORKERS
    lr: float = config.LR
    momentum: float = config.MOMENTUM
    lr_step_size: int = config.LR_STEP_SIZE
    lr_gamma: float = config.LR_GAMMA
    pretrained: bool = True
    use_imagenet_norm: bool = config.USE_IMAGENET_NORM
    tag: str = ""
    # subset for smoke-tests (None = full dataset)
    max_train: int | None = None
    max_val: int | None = None
    # --- Stage 2 (None = Stage 1 defaults / paper dataset) ---
    data_dir: str | None = None          # image root (e.g. str(config.CMPD300_DIR))
    splits_dir: str | None = None        # folder containing the split JSONs
    num_classes: int | None = None       # number of classes (CMPD300: from label_map)
    image_size: int | None = None        # input side (Stage 2: 224)
    use_precomputed_aug: bool = True      # False for datasets without an aug cache (CMPD300)
    init_from: str | None = None          # warm-start backbone from a custom checkpoint


def _maybe_subset(entries: list[dict], n: int | None) -> list[dict]:
    return entries if n is None else entries[:n]


def run_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> tuple[float, float]:
    """One epoch. If optimizer is None → evaluation mode (no grad). Returns (loss, acc)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_run(rc: RunConfig, device: str | None = None,
                  ckpt_dir: Path = config.CHECKPOINTS_DIR) -> dict:
    """Train one run and return metrics + path to the best checkpoint."""
    device = device or get_device()
    set_seed(rc.seed)
    tag = rc.tag or f"{rc.model_name}_{rc.loss_kind}_{'aug' if rc.use_aug else 'noaug'}_s{rc.seed}"
    log = get_logger(f"train.{tag}", logfile=config.LOGS_DIR / f"{tag}.log")
    config.ensure_output_dirs()

    # --- Source resolution (Stage 2 if passed; otherwise Stage 1 defaults) ---
    data_dir = Path(rc.data_dir) if rc.data_dir else config.DATA_DIR
    splits_dir = Path(rc.splits_dir) if rc.splits_dir else config.SPLITS_DIR
    num_classes = rc.num_classes if rc.num_classes is not None else config.NUM_CLASSES
    image_size = rc.image_size if rc.image_size is not None else config.IMAGE_SIZE

    log.info(f"=== RUN {tag} | device={device} ===")
    log.info(f"config: {asdict(rc)}")
    log.info(f"data_dir={data_dir} | splits_dir={splits_dir} | "
             f"num_classes={num_classes} | image_size={image_size}")

    # ---- Data ----
    # PAPER: augmentation CREATES synthetic images and ENLARGES the dataset (keeps
    # originals at real brightness), it does not replace each image. Originals → clean
    # transform; synthetic copies → augmentation. See dataset.make_train_loader / DEVIATIONS D4.
    clean_tf = build_transforms(train=False, image_size=image_size,
                                use_imagenet_norm=rc.use_imagenet_norm)
    aug_tf = build_transforms(train=True, image_size=image_size,
                              use_imagenet_norm=rc.use_imagenet_norm)
    train_e = _maybe_subset(load_split("train", splits_dir=splits_dir), rc.max_train)
    val_e = _maybe_subset(load_split("val", splits_dir=splits_dir), rc.max_val)
    train_loader = make_train_loader(train_e, use_aug=rc.use_aug, clean_tf=clean_tf,
                                     aug_tf=aug_tf, seed=rc.seed,
                                     batch_size=rc.batch_size, num_workers=rc.num_workers,
                                     data_dir=data_dir,
                                     # with subset (smoke) or datasets without cache, do NOT
                                     # use the precomputed cache (it belongs to the paper dataset).
                                     use_precomputed=(rc.max_train is None and rc.use_precomputed_aug))
    val_loader = make_dataloader(val_e, clean_tf, shuffle=False, data_dir=data_dir,
                                 batch_size=rc.batch_size, num_workers=rc.num_workers)
    log.info(f"train imgs: {len(train_loader.dataset):,} (use_aug={rc.use_aug}) | "
             f"val imgs: {len(val_e):,}")

    # ---- Model / loss / optimizer ----
    model = build_model(rc.model_name, num_classes=num_classes,
                        freeze_backbone=rc.freeze_backbone, pretrained=rc.pretrained,
                        init_from=rc.init_from).to(device)
    tr, tot = count_parameters(model)
    log.info(f"trainable params: {tr:,}/{tot:,} ({100*tr/tot:.1f}%)")

    criterion = build_loss(rc.loss_kind, train_entries=train_e, device=device)
    optimizer = torch.optim.SGD(trainable_parameters(model), lr=rc.lr, momentum=rc.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=rc.lr_step_size, gamma=rc.lr_gamma)

    # ---- Loop ----
    best_val, best_epoch = -1.0, -1
    history = []
    ckpt_path = Path(ckpt_dir) / f"{tag}_best.pt"
    t0 = time.time()
    for epoch in range(1, rc.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": lr_now})
        log.info(f"ep {epoch:02d}/{rc.epochs} | train acc {tr_acc:.4f} loss {tr_loss:.4f} "
                 f"| val acc {va_acc:.4f} loss {va_loss:.4f} | lr {lr_now:.1e}")
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            torch.save({
                "model_state": model.state_dict(),
                "model_name": rc.model_name,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_acc": va_acc,
                "run_config": asdict(rc),
            }, ckpt_path)

    elapsed = time.time() - t0
    log.info(f"DONE {tag}: best val acc {best_val:.4f} @ ep {best_epoch} | {elapsed:.0f}s")
    return {
        "tag": tag,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(elapsed, 1),
        "history": history,
        "run_config": asdict(rc),
    }
