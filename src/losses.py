"""losses.py — Cross-Entropy and Weighted Cross-Entropy.

PAPER (class imbalance optimization):
- Weighted CE: per-class weight w_i = N_max / N_i, with N_i = number of images of
  class i in train and N_max = maximum of those counts. The paper assumes N_max=70;
  the dataset maximum is also 70, so the empirical value matches (see
  DEVIATIONS.md). We compute N_max EMPIRICALLY from the train split.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

import config


def compute_class_counts(train_entries: list[dict],
                         num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """Per-class image count (N_i) in the train split."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for e in train_entries:
        counts[e["label"]] += 1
    return counts


def compute_class_weights(train_entries: list[dict],
                          num_classes: int = config.NUM_CLASSES,
                          nmax_override: int | None = config.WCE_NMAX_OVERRIDE) -> torch.Tensor:
    """Weighted CE weights: w_i = N_max / N_i (float32).

    In a full run all 268 classes are in train (guaranteed by the splits). If
    any class has 0 images (e.g. when subsetting in a smoke-test), it gets weight 0
    — it does not appear as a target, so it does not affect the loss — and a warning
    is emitted.
    """
    counts = compute_class_counts(train_entries, num_classes).float()
    nonzero = counts > 0
    if not bool(nonzero.all()):
        n_missing = int((~nonzero).sum())
        warnings.warn(f"{n_missing}/{num_classes} classes have no images in train "
                      f"(weight 0). This should NOT happen in a full run.")
    n_max = float(nmax_override) if nmax_override is not None else float(counts[nonzero].max())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    weights[nonzero] = n_max / counts[nonzero]
    return weights


def build_loss(kind: str,
               train_entries: list[dict] | None = None,
               num_classes: int = config.NUM_CLASSES,
               device: str = "cpu") -> nn.Module:
    """Build the loss function.

    Args:
        kind: 'ce' (Cross-Entropy) | 'wce' (Weighted Cross-Entropy).
        train_entries: required for 'wce' (to compute per-class weights).
        device: where to place the weight tensor.
    """
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "wce":
        if train_entries is None:
            raise ValueError("Weighted CE requires train_entries to compute weights.")
        weights = compute_class_weights(train_entries, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Unsupported loss: {kind}. Use 'ce' or 'wce'.")
