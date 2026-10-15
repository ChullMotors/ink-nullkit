"""Ink-free null input: keep each slice's 2D Fourier amplitude, replace phases with ONE random field shared across slices.

All stroke and letter structure is destroyed while intensity texture statistics and between-layer texture correlation are kept,
so any letter-shaped output blob on this input is a hallucination by construction. It is an out-of-distribution stress test,
an upper bound on false-positive behaviour, not a natural blank-papyrus false-positive rate (use a real blank segment for that).
"""
from __future__ import annotations
import numpy as np


def phase_randomize(volume_zyx: np.ndarray, seed: int = 1234) -> np.ndarray:
    """volume_zyx: uint8 (Z, Y, X). Zeros (empty padding) are treated as outside the object and restored."""
    v = volume_zyx.astype(np.float32)
    mask = volume_zyx > 0
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(v.shape[1:]))
    out = np.fft.ifft2(np.abs(np.fft.fft2(v, axes=(1, 2))) * phase[None], axes=(1, 2)).real
    return (np.clip(np.rint(out), 0, 255) * mask).astype(np.uint8)
