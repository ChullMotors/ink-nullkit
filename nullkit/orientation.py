"""Label-free depth-direction selection for ink models.

Two independently trained models (for example two seeds, or two architectures) are run on the same surface volume in the forward
and in the reversed depth direction. The direction in which the two models agree more (Pearson correlation of their probability
maps over the object area) is the better candidate. Agreement is accumulated per spatial block so that a block bootstrap gives
an interval for the margin; no labels and no mesh geometry are used. Agreement shows that the models respond to the same
structure, not that the structure is ink: the check recommends a direction, it does not validate a prediction."""
from __future__ import annotations
import numpy as np

_STAT = ("n", "a", "b", "aa", "bb", "ab")


def _block_stats(a, b, mask, block):
    H, W = a.shape; rows = []
    for y in range(0, H, block):
        for x in range(0, W, block):
            m = mask[y:y + block, x:x + block]
            if not m.any(): rows.append((0, 0, 0, 0, 0, 0)); continue
            p = a[y:y + block, x:x + block][m].astype(np.float64); q = b[y:y + block, x:x + block][m].astype(np.float64)
            rows.append((p.size, p.sum(), q.sum(), (p * p).sum(), (q * q).sum(), (p * q).sum()))
    return np.array(rows, float)


def _corr(s):
    n, a, b, aa, bb, ab = s.sum(0)
    if n < 2: return float("nan")
    cov = ab / n - a * b / n ** 2; va = aa / n - (a / n) ** 2; vb = bb / n - (b / n) ** 2
    return float(cov / np.sqrt(max(va * vb, 1e-18)))


def choose_direction(fwd_a, fwd_b, rev_a, rev_b, mask=None, block=256, n_boot=1000, min_block_px=2000, seed=0, min_margin=0.1, min_agreement=0.5):
    """fwd_*/rev_*: 2D probability maps (uint8 or float) from model A and B in the forward and reversed direction.
    Returns dict(direction, decision, agreement_forward, agreement_reverse, margin, ci95, block_vote, n_blocks, confident, weak_agreement).
    `decision` is "forward", "reverse" or "undecided": it requires the margin interval to exclude zero, |margin| >= min_margin and the
    higher agreement >= min_agreement. min_margin and min_agreement were set after seeing weak cases (a scroll where every model scored at chance
    gave margin -0.04 with interval excluding zero), not tuned on held-out data; known-good renders had |margin| 0.24-0.28."""
    fwd_a, fwd_b, rev_a, rev_b = (np.asarray(x) for x in (fwd_a, fwd_b, rev_a, rev_b))
    if mask is None: mask = (fwd_a > 0) | (fwd_b > 0) | (rev_a > 0) | (rev_b > 0)
    sf, sr = _block_stats(fwd_a, fwd_b, mask, block), _block_stats(rev_a, rev_b, mask, block)
    keep = sf[:, 0] >= min_block_px; sf, sr = sf[keep], sr[keep]
    if len(sf) < 4: raise ValueError("fewer than 4 usable blocks; need a larger area")
    cf, cr = _corr(sf), _corr(sr)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(sf), size=(n_boot, len(sf)))
    marg = np.array([_corr(sf[i]) - _corr(sr[i]) for i in idx])
    per_f = np.array([_corr(s[None]) for s in sf]); per_r = np.array([_corr(s[None]) for s in sr])
    ci = [float(np.percentile(marg, 2.5)), float(np.percentile(marg, 97.5))]
    direction = "forward" if cf >= cr else "reverse"; confident = bool(ci[0] > 0 or ci[1] < 0)
    decision = direction if (confident and abs(cf - cr) >= min_margin and max(cf, cr) >= min_agreement) else "undecided"
    return dict(direction=direction, decision=decision, agreement_forward=cf, agreement_reverse=cr, margin=cf - cr, ci95=ci,
                block_vote=float((per_f > per_r).mean()), n_blocks=int(len(sf)), confident=confident, weak_agreement=bool(max(cf, cr) < 0.5))
