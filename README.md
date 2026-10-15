# ink-nullkit

Small, dependency-light checks that tell you whether a difference between two ink-detection results is real, and whether a model draws letter-shaped blobs on input that contains no ink.

Written against the public Vesuvius Challenge `ink-labels` / `ink_9um` data and the `vesuvius.ink_detection` inference output (uint8 probability TIFFs). It does not import the villa package.

## Why

Held-out ink evaluation sets are small. On two public held-out segments (PHerc. 0841 w00, PHerc. 0500p2) there are only 39 and 77 independent 256 px blocks with labels, so paired AUROC differences below about 0.03 cannot be resolved (this agrees with the ~0.03 noise floor already reported by the validation-harness prize winner). Separately, letter-sized blobs in a prediction are not evidence of ink: one released seed draws them on ink-free input at the same rate as on the real render.

## Tools

- `nullkit.paired_auroc_diff` / `python -m nullkit.cli compare`: paired spatial block-bootstrap interval for AUROC(a) - AUROC(b) inside a supervision mask, plus the minimum effect detectable at 80% power for that region.
- `nullkit.phase_randomize` / `python -m nullkit.cli null`: ink-free copy of a surface volume (same per-slice Fourier amplitudes, one shared Hermitian-symmetric random phase field, padding filled before the transform and restored after, per-slice mean and standard deviation matched). Run your model on it. Anything letter-shaped it outputs is a hallucination by construction. This is an out-of-distribution stress test, an upper bound; use a real blank-papyrus segment for a natural false-positive rate.
- `nullkit.blob_density` / `python -m nullkit.cli blobs`: letter-scale (300 to 8000 px) connected components per megapixel of a region, with a block-bootstrap interval. Compare the density on the real region with the density on the phase null.

- `nullkit.choose_direction` / `python -m nullkit.cli orientation fwd_a.tif fwd_b.tif rev_a.tif rev_b.tif`: label-free choice of depth direction. Run two independently trained models on the same surface volume forward and reversed; the direction in which their probability maps correlate more is the better candidate. Returns the margin with a block-bootstrap interval, the fraction of blocks that vote for each direction, and flags `confident` (interval excludes zero) and `weak_agreement` (best agreement below 0.5).

## Example

```bash
python -m nullkit.cli compare s42_reverse.tif s43_reverse.tif w00_inklabels_v2.zarr w00_supervision_mask_v2.zarr --level 2 --z 32
python -m nullkit.cli null w00_9um.zarr w00_9um_null.zarr
python -m nullkit.cli blobs null_prediction.tif w00_occupied_region.zarr --threshold 0.7
```

## Measured with it (v0.1.2; public data; `scrollprize/ink_9um` seeds 42/43 at step 75000 and `domenicor046/reader-v2` step 40000)

Inputs are the official 109-slice (0841 w00) and 118-slice (0500p2) surface volumes from the open-data bucket, prepared with the official `prepare_9um_isotropic_input` (level 2), forward direction, scored inside the official 20260918 supervision masks. Letter-sized blobs per megapixel (95% block-bootstrap interval) on the real render and on the ink-free phase null:

| threshold 0.7 | 0841 real | 0841 null | 0500p2 real | 0500p2 null |
|---|---|---|---|---|
| seed 42 | 6.4 [5.1, 7.7] | 7.2 [5.6, 9.0] | 5.6 [4.4, 6.9] | 6.5 [5.1, 8.1] |
| seed 43 | 6.9 [5.7, 8.1] | 0.6 [0.2, 1.1] | 5.4 [4.1, 6.8] | 0.4 [0.1, 0.8] |
| reader-v2 | 4.8 [3.6, 6.0] | 0.0 [0.0, 0.0] | 3.4 [2.4, 4.4] | 0.0 [0.0, 0.0] |

| threshold 0.5 | 0841 real | 0841 null | 0500p2 real | 0500p2 null |
|---|---|---|---|---|
| seed 42 | 13.0 | 5.8 | 12.1 | 12.4 |
| seed 43 | 13.0 | 14.5 | 11.8 | 13.0 |
| reader-v2 | 8.9 | 1.0 | 5.2 | 2.3 |

Caveat: reader-v2 was trained on other segments of the 0500P2 scroll (per its published training config), so its 0500p2 column is not held out by scroll; 0841 is. AUROC vs the official labels (forward): seed 42 0.853 / 0.840, seed 43 0.891 / 0.857, reader-v2 0.904 / 0.908 (0841 / 0500p2). Equal AUROC hid large differences in null behaviour: seed 42 draws as many letter-sized blobs on ink-free data as on the real render at 0.7, seed 43 only behaves at 0.7, and reader-v2 stays low on the null at both thresholds.

**Depth direction depends on the render, not just the model.** An earlier version of this table was computed on older pooled renders of the same segments where the reversed direction scored best (AUROC about 0.81 to 0.87) and forward was at chance. On the official prepared inputs the forward direction scores best and reversed is near chance (0.53 to 0.58). Always check the direction on your own render with a quick forward/reverse comparison before drawing conclusions.

## Depth-direction check, measured

Seed 42 vs seed 43 of `scrollprize/ink_9um`, forward vs reversed, 256 px blocks. Truth is the direction with the better AUROC against the labels (used only for validation):

| render | agreement forward / reversed | margin [95% interval] | choice | truth |
|---|---|---|---|---|
| PHerc. 0841 w00, official surface volume | 0.715 / 0.434 | +0.28 [+0.23, +0.33] | forward | forward |
| PHerc. 0500p2, official surface volume | 0.663 / 0.413 | +0.25 [+0.19, +0.30] | forward | forward |
| PHerc. 0139 w035 (training corpus segment) | 0.665 / 0.410 | +0.25 [+0.22, +0.30] | forward | forward |
| PHerc. 0841 w00, older pooled mirror render | 0.443 / 0.679 | -0.24 [-0.28, -0.19] | reverse | reverse |
| PHerc. 0500p2, older pooled mirror render | 0.458 / 0.701 | -0.24 [-0.31, -0.18] | reverse | reverse |

On ink-free phase nulls of the two official renders the margin is -0.002 and -0.013 with intervals including zero (abstains). On a native 9.366 um render of 0841 made with a home-made sampler it chose reverse (margin -0.15 [-0.20, -0.11]), i.e. that sampler's default normal was opposite to the official convention. It chose correctly on all five known renders, and the same segments needed opposite directions depending on which render they came from. Limits: both models come from one training recipe and corpus, so shared bias could mimic agreement; only two render conventions were available; and when agreement is weak it should abstain. On six PHerc. 0800 segments without labels it was confident on one and undecided on five, which is the intended behaviour. Agreement does not show that the structure is ink, so combine it with the null screen above.

## Limits

The phase null is not natural papyrus (mean model output rises on it), the blob thresholds are choices, and two segments are a small sample. Intervals resample spatial blocks and assume blocks are exchangeable, which spatially correlated labels only approximately satisfy.

## Tests

`PYTHONPATH=. pytest -q tests` (synthetic data only).

## License

MIT. See `LICENSE`.
