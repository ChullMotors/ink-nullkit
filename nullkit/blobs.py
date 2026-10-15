"""Letter-scale blob density: connected components of a thresholded probability map, per megapixel of a region."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def blob_density(prob: np.ndarray, threshold: float, region: np.ndarray, area_range=(300, 8000), block=256, n_boot=1000, seed=0):
    """Blobs (components with area in `area_range` pixels whose centroid lies in `region`) per megapixel of `region`,
    with a spatial block-bootstrap 95% interval. `prob` is a float map in [0, 1]. Returns (density, (lo, hi), n_blobs)."""
    cc, n = ndi.label(prob >= threshold)
    H, W = prob.shape
    nby, nbx = -(-H // block), -(-W // block)
    cnt = np.zeros((nby, nbx))
    nblobs = 0
    if n:
        ones = np.ones_like(prob, dtype=np.float32)
        idx = np.arange(1, n + 1)
        area = ndi.sum(ones, cc, idx)
        cen = np.array(ndi.center_of_mass(ones, cc, idx)).astype(int)
        keep = (area >= area_range[0]) & (area <= area_range[1])
        cen = cen[keep]
        cen = cen[region[cen[:, 0].clip(0, H - 1), cen[:, 1].clip(0, W - 1)]]
        nblobs = len(cen)
        np.add.at(cnt, (cen[:, 0] // block, cen[:, 1] // block), 1)
    ar = np.zeros((nby, nbx))
    for by in range(nby):
        for bx in range(nbx):
            ar[by, bx] = region[by * block:(by + 1) * block, bx * block:(bx + 1) * block].sum()
    ok = ar.ravel() > 0
    c, a = cnt.ravel()[ok], ar.ravel()[ok]
    if a.sum() == 0:
        raise ValueError("region is empty")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(c), size=(n_boot, len(c)))
    bs = c[i].sum(1) / a[i].sum(1) * 1e6
    return float(c.sum() / a.sum() * 1e6), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), nblobs
