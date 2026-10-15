# Changelog

## 0.1.1
- Fix: `phase_randomize` used non-Hermitian random phases and discarded the imaginary part, so the null lost power (mean intensity about half the source). Now uses phases of a real white-noise field, fills padding before the transform, and matches each slice's mean and standard deviation. Added tests. The v0.1.0 results table is withdrawn pending regeneration.

## 0.1.0
- First release.
