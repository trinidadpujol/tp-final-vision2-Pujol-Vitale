"""eval_reid.py — Métricas de re-identificación gallery/probe (Fase 6).

Protocolo estándar: cada probe se rankea contra toda la gallery por similitud coseno
(embeddings ya L2-normalizados → coseno = producto punto). Se reportan:
- Rank-1 / Rank-5 (CMC): fracción de probes cuyo top-1 / top-5 contiene su mismo individuo.
- mAP: mean Average Precision sobre todos los aciertos en la gallery.
"""
from __future__ import annotations

import numpy as np


def rank_metrics(probe_emb: np.ndarray, probe_lab: np.ndarray,
                 gal_emb: np.ndarray, gal_lab: np.ndarray,
                 topk: tuple[int, ...] = (1, 5)) -> dict:
    """Devuelve {rank1, rank5, mAP, n_probe, n_gallery, n_ids_gallery}."""
    sims = probe_emb @ gal_emb.T                    # [P, G] coseno
    order = np.argsort(-sims, axis=1)               # gallery ordenada por similitud desc.
    gl_sorted = gal_lab[order]                       # [P, G]
    matches = (gl_sorted == probe_lab[:, None]).astype(np.int32)

    res: dict = {}
    for k in topk:
        kk = min(k, matches.shape[1])
        res[f"rank{k}"] = float((matches[:, :kk].sum(1) > 0).mean())

    aps = []
    for i in range(len(probe_lab)):
        m = matches[i]
        total = int(m.sum())
        if total == 0:
            aps.append(0.0)
            continue
        cum = np.cumsum(m)
        prec_at_hit = cum / (np.arange(len(m)) + 1)
        aps.append(float((prec_at_hit * m).sum() / total))
    res["mAP"] = float(np.mean(aps)) if aps else 0.0

    res["n_probe"] = int(len(probe_lab))
    res["n_gallery"] = int(len(gal_lab))
    res["n_ids_gallery"] = int(len(np.unique(gal_lab)))
    return res
