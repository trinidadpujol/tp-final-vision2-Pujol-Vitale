"""eval_reid.py — Re-identification gallery/probe metrics (Phase 6).

Standard protocol: each probe is ranked against the entire gallery by cosine similarity
(L2-normalized embeddings → cosine = dot product). Reported metrics:
- Rank-1 / Rank-5 (CMC): fraction of probes whose top-1 / top-5 contains their individual.
- mAP: mean Average Precision over all hits in the gallery.
"""
from __future__ import annotations

import numpy as np


def rank_metrics(probe_emb: np.ndarray, probe_lab: np.ndarray,
                 gal_emb: np.ndarray, gal_lab: np.ndarray,
                 topk: tuple[int, ...] = (1, 5)) -> dict:
    """Return {rank1, rank5, mAP, n_probe, n_gallery, n_ids_gallery}."""
    sims = probe_emb @ gal_emb.T                    # [P, G] cosine
    order = np.argsort(-sims, axis=1)               # gallery sorted by similarity desc.
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
