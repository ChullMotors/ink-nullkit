# ink-nullkit

Small, dependency-light checks that tell you whether a difference between two ink-detection results is real, and whether a model draws letter-shaped blobs on input that contains no ink.

Written against the public Vesuvius Challenge `ink-labels` / `ink_9um` data and the `vesuvius.ink_detection` inference output (uint8 probability TIFFs). It does not import the villa package.

## Why

Held-out ink evaluation sets are small. On two public held-out segments (PHerc. 0841 w00, PHerc. 0500p2) there are only 39 and 77 independent 256 px blocks with labels, so paired AUROC differences below about 0.03 cannot be resolved (this agrees with the ~0.03 noise floor already reported by the validation-harness prize winner). Separately, letter-sized blobs in a prediction are not evidence of ink: one released seed draws them on ink-free input at the same rate as on the real render.

## Tools

- `nullkit.paired_auroc_diff` / `python -m nullkit.cli compare`: paired spatial block-bootstrap interval for AUROC(a) - AUROC(b) inside a supervision mask, plus the minimum effect detectable at 80% power for that region.
- `nullkit.phase_randomize` / `python -m nullkit.cli null`: ink-free copy of a surface volume (same per-slice Fourier amplitudes, one shared Hermitian-symmetric random phase field, padding filled before the transform and restored after, per-slice mean and standard deviation matched). Run your model on it. Anything letter-shaped it outputs is a hallucination by construction. This is an out-of-distribution stress test, an upper bound; use a real blank-papyrus segment for a natural false-positive rate.
- `nullkit.blob_density` / `python -m nullkit.cli blobs`: letter-scale (300 to 8000 px) connected components per megapixel of a region, with a block-bootstrap interval. Compare the density on the real region with the density on the phase null.

## Example

```bash
python -m nullkit.cli compare s42_reverse.tif s43_reverse.tif w00_inklabels_v2.zarr w00_supervision_mask_v2.zarr --level 2 --z 32
python -m nullkit.cli null w00_9um.zarr w00_9um_null.zarr
python -m nullkit.cli blobs null_prediction.tif w00_occupied_region.zarr --threshold 0.7
```

## Measured with it (v0.1.1 null; public data; `scrollprize/ink_9um` checkpoints at step 75000, depth reversed)

Letter-sized blobs per megapixel (95% block-bootstrap interval), threshold 0.7:

| | PHerc. 0841 w00 | PHerc. 0500p2 |
|---|---|---|
| seed 42, real render | 7.5 [6.1, 8.9] | 5.4 [4.2, 6.8] |
| seed 42, ink-free null | 6.3 [4.8, 7.9] | 4.4 [3.3, 5.6] |
| seed 43, real render | 7.6 [6.3, 9.0] | 4.8 [3.6, 6.0] |
| seed 43, ink-free null | 0.5 [0.2, 1.0] | 0.4 [0.1, 0.7] |
| AUROC seed 42 / seed 43 vs labels | 0.812 / 0.818 | 0.869 / 0.858 |

At threshold 0.5 the ordering is not the same: seed 43 draws 16.9 (0841) and 12.7 (0500p2) blobs per megapixel on the null, seed 42 draws 6.1 and 13.7. So screen across thresholds rather than at one, and note that equal held-out AUROC hid a large difference in behaviour on the null at 0.7. A v0.1.0 table computed with a flawed null was withdrawn (see CHANGELOG).

## Limits

The phase null is not natural papyrus (mean model output rises on it), the blob thresholds are choices, and two segments are a small sample. Intervals resample spatial blocks and assume blocks are exchangeable, which spatially correlated labels only approximately satisfy.

## Tests

`PYTHONPATH=. pytest -q tests` (synthetic data only).

## License

MIT. See `LICENSE`.
