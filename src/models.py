"""models.py — VGG16_BN and ResNet-50 builders.

PAPER:
- Transfer learning from ImageNet.
- ONLY the fully-connected layers are fine-tuned (convolutional backbone frozen).
- VGG16_BN is the model that replicates 98.7%. ResNet-50 is our own backbone
  (for reuse in domain adaptation), also tested with full fine-tuning.

Loading pretrained weights without depending on internet (Kaggle): if `config.PRETRAINED_DIR`
has the corresponding .pth, it is loaded from there (`weights=None` + load_state_dict);
otherwise it is downloaded via the torchvision API.

Stage 2: `build_model(..., init_from=<checkpoint_path>)` initializes the BACKBONE
from a custom checkpoint (e.g. the winning ResNet-50 from the replication,
outputs/checkpoints/resnet50_backbone.pt), discarding its classification head.
Useful as a warm-start for the CMPD300 source encoder. The ImageNet load (or random
init) happens first and the backbone is overwritten afterwards; the head is always fresh.

Designed for the future embedding phase: `build_model(..., head=False)` (or
`strip_classifier`) returns the backbone without the classification layer, exposing
embeddings. Not used in this stage but ready to be extended.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tvm

import config
from src.utils import get_logger


def _load_pretrained_state(model_name: str) -> dict:
    """Returns the ImageNet state_dict: from local file if available, otherwise downloads."""
    fname = config.PRETRAINED_FILES[model_name]
    if config.PRETRAINED_DIR is not None:
        local = Path(config.PRETRAINED_DIR) / fname
        if local.is_file():
            return torch.load(local, map_location="cpu")
    # Fallback: download via torchvision (requires internet).
    weights = {
        "vgg16_bn": tvm.VGG16_BN_Weights.IMAGENET1K_V1,
        "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1,
    }[model_name]
    return tvm.get_model(model_name, weights=weights).state_dict()


def _apply_pretrained(model: nn.Module, model_name: str, pretrained: bool) -> None:
    if pretrained:
        model.load_state_dict(_load_pretrained_state(model_name))


def load_backbone_from_checkpoint(model: nn.Module, ckpt_path: str | Path,
                                  skip_prefixes: tuple[str, ...] = ("fc.", "classifier.")
                                  ) -> tuple[int, int]:
    """Load BACKBONE weights into `model` from a custom checkpoint.

    Discards the classification head (`skip_prefixes`) and any tensor whose shape
    does not match (nothing incompatible is overwritten). Accepts both a checkpoint
    saved by `train.py` ({"model_state": ...}) and a raw state_dict.

    Returns (n_loaded, n_skipped). Does not change `requires_grad` (freeze is applied
    separately, before or after).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"init_from: checkpoint {ckpt_path} does not exist. "
            f"Did you run the replication (03_train_resnet.py) and copy it to "
            f"resnet50_backbone.pt?"
        )
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj

    model_sd = model.state_dict()
    to_load, skipped = {}, 0
    for k, v in state.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped += 1
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped += 1
    model.load_state_dict(to_load, strict=False)

    log = get_logger("models.init_from")
    log.info(f"init_from {ckpt_path.name}: backbone loaded ({len(to_load)} tensors), "
             f"{skipped} skipped (head/shape-mismatch).")
    return len(to_load), skipped


def build_model(name: str,
                num_classes: int = config.NUM_CLASSES,
                freeze_backbone: bool = config.FREEZE_BACKBONE,
                pretrained: bool = True,
                init_from: str | Path | None = None) -> nn.Module:
    """Build the model with a `num_classes`-output head.

    Args:
        name: 'vgg16_bn' | 'resnet50'.
        num_classes: number of classes (PAPER: 268; CMPD300: from label_map).
        freeze_backbone: if True, freeze the convolutional backbone and train only
            the FC layers (PAPER). If False, full fine-tuning.
        pretrained: load ImageNet weights (True) or random init (False, for tests).
        init_from: path to a custom checkpoint to initialize the BACKBONE (Stage 2,
            warm-start). Applied AFTER ImageNet/random init and after replacing the head;
            the head is always fresh. If passed, you can set pretrained=False
            (the backbone comes from the checkpoint).
    """
    name = name.lower()
    if name == "vgg16_bn":
        model = tvm.vgg16_bn(weights=None)
        _apply_pretrained(model, name, pretrained)
        # Replace the last classifier layer (Linear 4096→1000) → 4096→num_classes.
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            # PAPER: freeze features (conv), train only classifier (FC).
            for p in model.features.parameters():
                p.requires_grad = False

    elif name == "resnet50":
        model = tvm.resnet50(weights=None)
        _apply_pretrained(model, name, pretrained)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            # Freeze everything except fc.
            for pname, p in model.named_parameters():
                p.requires_grad = pname.startswith("fc.")

    else:
        raise ValueError(f"Unsupported model: {name}. Use one of {config.MODELS}.")

    # Stage 2: overwrite the backbone with a custom checkpoint (warm-start).
    if init_from:
        load_backbone_from_checkpoint(model, init_from)

    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters with requires_grad=True (those seen by the optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
