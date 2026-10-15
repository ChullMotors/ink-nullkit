"""Spatial block bootstrap for AUROC (single and paired). Predictions are uint8 (0-255) probability maps."""
from __future__ import annotations
import numpy as np


def block_hists(pred: np.ndarray, label: np.ndarray, mask: np.ndarray, block: int = 256, min_px: int = 500):
    """Per-block positive/negative score histograms inside `mask`. Blocks with fewer than `min_px` masked pixels are skipped."""
    if pred.dtype != np.uint8:
        raise ValueError("pred must be uint8 (scale probabilities to 0-255)")
    H, W = pred.shape
    pos, neg = [], []
    for by in range(H // block):
        for bx in range(W // block):
            s = np.s_[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            m = mask[s]
            if m.sum() < min_px:
                continue
            p, l = pred[s][m], label[s][m] > 0
            pos.append(np.bincount(p[l], minlength=256))
            neg.append(np.bincount(p[~l], minlength=256))
    return np.array(pos, float), np.array(neg, float)


def auroc_from_hists(hp: np.ndarray, hn: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), from summed histograms."""
    cn = np.cumsum(hn) - hn
    return float((hp * (cn + 0.5 * hn)).sum() / max(hp.sum() * hn.sum(), 1))


def paired_auroc_diff(pred_a, pred_b, label, mask, block=256, n_boot=2000, seed=0):
    """AUROC(a) - AUROC(b) with a paired block-bootstrap 95% interval (same resampled blocks for both maps).
    Returns dict(auroc_a, auroc_b, diff, ci95, sd, n_blocks, mde_80pct). mde_80pct = 2.8 * sd, the smallest true difference
    detectable with 80% power at the 5% level for this evaluation region."""
    ha, hb = block_hists(pred_a, label, mask, block), block_hists(pred_b, label, mask, block)
    nb = len(ha[0])
    if nb < 5:
        raise ValueError(f"only {nb} usable blocks; need at least 5 for a bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    d = np.array([auroc_from_hists(ha[0][i].sum(0), ha[1][i].sum(0)) - auroc_from_hists(hb[0][i].sum(0), hb[1][i].sum(0)) for i in idx])
    a, b = auroc_from_hists(ha[0].sum(0), ha[1].sum(0)), auroc_from_hists(hb[0].sum(0), hb[1].sum(0))
    return dict(auroc_a=a, auroc_b=b, diff=a - b, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                sd=float(d.std()), n_blocks=nb, mde_80pct=float(2.8 * d.std()))
