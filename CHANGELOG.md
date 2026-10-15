# Changelog

## 0.2.0
- Added `choose_direction` and the `orientation` CLI command: label-free depth-direction choice from two models' forward and reversed maps, with block-bootstrap interval and abstention flags. Tests added.

## 0.1.2
- README results recomputed on the official surface volumes with the official 9 um preparation (forward direction) and now include reader-v2. The earlier table came from older pooled renders in which the reversed direction was correct; that was a property of those renders, noted in the README. No code change.

## 0.1.1
- Fix: `phase_randomize` used non-Hermitian random phases and discarded the imaginary part, so the null lost power (mean intensity about half the source). Now uses phases of a real white-noise field, fills padding before the transform, and matches each slice's mean and standard deviation. Added tests. The v0.1.0 results table was withdrawn and has been regenerated in the README with the corrected null.

## 0.1.0
- First release.
