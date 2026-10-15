# ink-nullkit

Small, dependency-light checks that tell you whether a difference between two ink-detection results is real, and whether a model draws letter-shaped blobs on input that contains no ink.

Written against the public Vesuvius Challenge `ink-labels` / `ink_9um` data and the `vesuvius.ink_detection` inference output (uint8 probability TIFFs). It does not import the villa package.

## Why

Held-out ink evaluation sets are small. On two public held-out segments (PHerc. 0841 w00, PHerc. 0500p2) there are only 39 and 77 independent 256 px blocks with labels, so paired AUROC differences below about 0.03 cannot be resolved (this agrees with the ~0.03 noise floor already reported by the validation-harness prize winner). Separately, letter-sized blobs in a prediction are not evidence of ink: one released seed draws them on ink-free input at the same rate as on the real render.

## Tools

- `nullkit.paired_auroc_diff` / `python -m nullkit.cli compare`: paired spatial block-bootstrap interval for AUROC(a) - AUROC(b) inside a supervision mask, plus the minimum effect detectable at 80% power for that region.
- `nullkit.phase_randomize` / `python -m nullkit.cli null`: ink-free copy of a surface volume (same per-slice Fourier amplitudes, one shared random phase field, padding restored). Run your model on it. Anything letter-shaped it outputs is a hallucination by construction. This is an out-of-distribution stress test, an upper bound; use a real blank-papyrus segment for a natural false-positive rate.
- `nullkit.blob_density` / `python -m nullkit.cli blobs`: letter-scale (300 to 8000 px) connected components per megapixel of a region, with a block-bootstrap interval. Compare the density on the real region with the density on the phase null.

## Example

```bash
python -m nullkit.cli compare s42_reverse.tif s43_reverse.tif w00_inklabels_v2.zarr w00_supervision_mask_v2.zarr --level 2 --z 32
python -m nullkit.cli null w00_9um.zarr w00_9um_null.zarr
python -m nullkit.cli blobs null_prediction.tif w00_occupied_region.zarr --threshold 0.7
```

## Measured with it (public data, `scrollprize/ink_9um` checkpoints at step 75000, depth reversed)

| Quantity | PHerc. 0841 w00 | PHerc. 0500p2 |
|---|---|---|
| Blobs per Mpx at 0.7, seed 42, real render | 7.5 [6.1, 8.9] | 5.4 [4.2, 6.8] |
| same, seed 42, phase null (no ink) | 9.5 [7.6, 11.3] | 5.9 [4.6, 7.4] |
| same, seed 43, real render | 7.6 [6.3, 9.0] | 4.8 [3.6, 6.0] |
| same, seed 43, phase null (no ink) | 0.8 [0.4, 1.3] | 0.5 [0.1, 0.9] |
| AUROC seed 42 / seed 43 vs labels | 0.812 / 0.818 | 0.869 / 0.858 |

Seeds with equal held-out AUROC differ by more than ten times in hallucination rate on the null. Label-based AUROC alone cannot see this.

## Limits

The phase null is not natural papyrus (mean model output rises on it), the blob thresholds are choices, and two segments are a small sample. Intervals resample spatial blocks and assume blocks are exchangeable, which spatially correlated labels only approximately satisfy.

## Tests

`PYTHONPATH=. pytest -q tests` (synthetic data only).

## License

MIT. See `LICENSE`.
