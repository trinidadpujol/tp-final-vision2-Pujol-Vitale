"""embeddings.py — Extractor de embeddings (Fase 6).

Toma un checkpoint entrenado por `train.py` (p.ej. `cmpd300_source.pt`), reconstruye el
modelo, le saca la cabeza de clasificación y expone el vector del global average pool
(**2048-d en ResNet-50**), **L2-normalizado** → distancia coseno = producto punto.

Usa el MISMO preprocesamiento con el que se entrenó el encoder (guardado en el run_config
del checkpoint: image_size / use_imagenet_norm), para que source y target se procesen igual.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from src.dataset import MuzzleDataset
from src.models import build_model
from src.transforms import build_transforms
from src.utils import get_device, get_logger


def _build_backbone(ckpt_path: Path, device: str):
    """Carga el checkpoint y devuelve (backbone_sin_cabeza, meta)."""
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model_name = obj.get("model_name", "resnet50") if isinstance(obj, dict) else "resnet50"
    num_classes = obj.get("num_classes", config.NUM_CLASSES) if isinstance(obj, dict) else config.NUM_CLASSES

    model = build_model(model_name, num_classes=num_classes,
                        freeze_backbone=False, pretrained=False)
    model.load_state_dict(state)

    if model_name == "resnet50":
        # todo menos la fc final → salida [B, 2048, 1, 1]
        backbone = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Extractor no soportado para {model_name} (usar resnet50).")

    backbone.eval().to(device)
    return backbone, (obj if isinstance(obj, dict) else {})


class EmbeddingExtractor:
    """Envuelve un encoder y produce embeddings L2-norm de un conjunto de entries."""

    def __init__(self, ckpt_path: Path = config.CHECKPOINTS_DIR / "cmpd300_source.pt",
                 device: str | None = None):
        self.device = device or get_device()
        self.log = get_logger("reid.embeddings")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint {ckpt_path}. ¿Corriste la Fase 5?")
        self.backbone, meta = _build_backbone(ckpt_path, self.device)
        rc = meta.get("run_config", {}) if isinstance(meta, dict) else {}
        self.image_size = rc.get("image_size") or config.IMAGE_SIZE_S2
        self.use_imagenet_norm = rc.get("use_imagenet_norm", config.USE_IMAGENET_NORM_S2)
        self.tf = build_transforms(train=False, image_size=self.image_size,
                                   use_imagenet_norm=self.use_imagenet_norm)
        self.log.info(f"encoder listo | image_size={self.image_size} | "
                      f"imagenet_norm={self.use_imagenet_norm} | device={self.device}")

    @torch.no_grad()
    def embed(self, entries: list[dict], data_dir: Path,
              batch_size: int = 64, num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """entries [{path,label}] → (embeddings [N,2048] L2-norm, labels [N])."""
        ds = MuzzleDataset(entries, transform=self.tf, data_dir=Path(data_dir))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available())
        embs, labs = [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            f = self.backbone(imgs).flatten(1)          # [B, 2048]
            f = F.normalize(f, dim=1)
            embs.append(f.cpu().numpy())
            labs.append(np.asarray(labels))
        return np.concatenate(embs), np.concatenate(labs)
