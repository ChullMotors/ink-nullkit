"""Ink-free null input: keep each slice's 2D Fourier amplitude, replace phases with ONE random field shared across slices.

All stroke and letter structure is destroyed while intensity texture statistics and between-layer texture correlation are kept,
so any letter-shaped output blob on this input is a hallucination by construction. Each slice keeps its mean and standard deviation inside the valid area. It is an out-of-distribution stress test,
an upper bound on false-positive behaviour, not a natural blank-papyrus false-positive rate (use a real blank segment for that).
"""
from __future__ import annotations
import numpy as np


def phase_randomize(volume_zyx: np.ndarray, seed: int = 1234) -> np.ndarray:
    """volume_zyx: uint8 (Z, Y, X). Zeros (empty padding) are treated as outside the object and restored."""
    mask = volume_zyx > 0
    v = volume_zyx.astype(np.float32)
    # fill empty padding with the per-slice mean of the valid area so the padding step edge does not dominate the spectrum
    for z in range(v.shape[0]):
        if mask[z].any():
            v[z][~mask[z]] = v[z][mask[z]].mean()
    rng = np.random.default_rng(seed)
    # Phases of a real white-noise field are Hermitian-symmetric, so the inverse transform is real and the amplitude spectrum
    # (hence variance) is preserved exactly. The DC phase is forced to zero so each slice keeps its mean.
    ph = np.fft.fft2(rng.standard_normal(v.shape[1:]))
    ph = ph / np.maximum(np.abs(ph), 1e-12)
    ph[0, 0] = 1.0
    out = np.fft.ifft2(np.abs(np.fft.fft2(v, axes=(1, 2))) * ph[None], axes=(1, 2)).real
    # match each slice's mean and standard deviation inside the valid area to the source (random phases spread large-scale
    # structure, which would otherwise lower the contrast of the null relative to the real render)
    for z in range(out.shape[0]):
        m = mask[z]
        if m.sum() > 1:
            src_m, src_s = float(volume_zyx[z][m].mean()), float(volume_zyx[z][m].std())
            o = out[z][m]; out[z] = (out[z] - o.mean()) / max(float(o.std()), 1e-6) * src_s + src_m
    return (np.clip(np.rint(out), 0, 255) * mask).astype(np.uint8)
