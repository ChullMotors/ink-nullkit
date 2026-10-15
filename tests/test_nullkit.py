import numpy as np
from nullkit import paired_auroc_diff, blob_density, phase_randomize


def _scene(seed=0, H=1024, W=1024):
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W), bool)
    for _ in range(40):
        y, x = rng.integers(40, H - 40), rng.integers(40, W - 40); lab[y - 12:y + 12, x - 4:x + 4] = True
    return lab, np.ones((H, W), bool), rng


def test_identical_maps_diff_is_zero():
    lab, m, rng = _scene()
    p = np.clip(lab * 120 + rng.normal(90, 30, lab.shape), 0, 255).astype(np.uint8)
    r = paired_auroc_diff(p, p, lab, m, n_boot=200)
    assert r["diff"] == 0 and r["ci95"] == [0.0, 0.0]


def test_clear_improvement_detected_and_noise_not():
    lab, m, rng = _scene()
    base = rng.normal(90, 30, lab.shape)
    weak = np.clip(base + lab * 30, 0, 255).astype(np.uint8)
    strong = np.clip(base + lab * 90, 0, 255).astype(np.uint8)
    r = paired_auroc_diff(strong, weak, lab, m, n_boot=300)
    assert r["ci95"][0] > 0
    other = np.clip(rng.normal(90, 30, lab.shape) + lab * 30, 0, 255).astype(np.uint8)
    r2 = paired_auroc_diff(weak, other, lab, m, n_boot=300)
    assert r2["ci95"][0] < 0 < r2["ci95"][1] or abs(r2["diff"]) < 0.02


def test_phase_null_destroys_structure_keeps_spectrum_and_mask():
    rng = np.random.default_rng(1)
    v = np.zeros((5, 256, 256), np.uint8); v[:, 20:240, 20:240] = 100
    v[:, 100:140, 120:130] = 220  # a bar "stroke"
    n = phase_randomize(v)
    assert n.shape == v.shape and (n[v == 0] == 0).all()
    inside = np.s_[:, 100:140, 120:130]
    assert abs(float(n[inside].mean()) - float(n[:, 20:240, 20:240].mean())) < 40  # bar no longer stands out
    a0, a1 = np.abs(np.fft.fft2(v[0].astype(float))), np.abs(np.fft.fft2(n[0].astype(float)))
    assert np.corrcoef(a0.ravel(), a1.ravel())[0, 1] > 0.5


def test_blob_density_counts_letter_sized_blobs_only():
    p = np.zeros((1024, 1024), np.float32)
    p[100:140, 100:120] = 1  # 800 px: counted
    p[500:503, 500:503] = 1  # 9 px: ignored
    p[600:900, 600:900] = 1  # 90000 px: ignored (too big)
    d, ci, n = blob_density(p, 0.5, np.ones_like(p, bool), n_boot=50)
    assert n == 1 and d > 0
