"""utils.py — Determinismo, logging e I/O. Único lugar donde se fijan seeds."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija las semillas de random, numpy y torch (CPU + CUDA) en un solo lugar.

    `torch` se importa adentro para que utils funcione aunque torch no esté
    instalado (p.ej. corriendo solo la inspección de datos de Fase 0).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    """'cuda' si hay GPU disponible, si no 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_logger(name: str = "cattle", logfile: Path | None = None) -> logging.Logger:
    """Logger legible a stdout (y opcionalmente a archivo)."""
    logger = logging.getLogger(name)
    if logger.handlers:  # ya configurado
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
