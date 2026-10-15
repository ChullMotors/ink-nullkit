"""Command line: python -m nullkit.cli {null,compare,blobs} ..."""
from __future__ import annotations
import argparse, json
import numpy as np


def _read_tif(p):
    import tifffile
    return tifffile.imread(p)


def _read_mask(p, level, z):
    import zarr
    a = zarr.open(p, mode="r")[level]
    a = a[z] if z is not None else a[:]
    return np.squeeze(np.asarray(a)) > 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nullkit")
    sp = ap.add_subparsers(dest="cmd", required=True)
    n = sp.add_parser("null", help="write a phase-randomized, ink-free copy of a surface-volume zarr (array '0', ZYX uint8)")
    n.add_argument("src"); n.add_argument("dst"); n.add_argument("--seed", type=int, default=1234)
    c = sp.add_parser("compare", help="paired block-bootstrap AUROC difference between two prediction TIFFs")
    c.add_argument("pred_a"); c.add_argument("pred_b"); c.add_argument("labels"); c.add_argument("mask")
    c.add_argument("--level", default="0"); c.add_argument("--z", type=int); c.add_argument("--block", type=int, default=256)
    b = sp.add_parser("blobs", help="letter-scale blob density of a prediction TIFF in a region")
    b.add_argument("pred"); b.add_argument("region"); b.add_argument("--level", default="0"); b.add_argument("--z", type=int)
    b.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args(argv)
    if a.cmd == "null":
        import zarr
        from .phase_null import phase_randomize
        v = np.asarray(zarr.open(a.src, mode="r")["0"])
        g = zarr.open_group(a.dst, mode="w", zarr_format=2)
        g.create_array("0", shape=v.shape, chunks=(v.shape[0], 128, 128), dtype=np.uint8, fill_value=0)[:] = phase_randomize(v, a.seed)
    elif a.cmd == "compare":
        from .bootstrap import paired_auroc_diff
        pa, pb = _read_tif(a.pred_a), _read_tif(a.pred_b)
        lab, m = _read_mask(a.labels, a.level, a.z)[:pa.shape[0], :pa.shape[1]], _read_mask(a.mask, a.level, a.z)[:pa.shape[0], :pa.shape[1]]
        print(json.dumps(paired_auroc_diff(pa, pb, lab, m, a.block), indent=1))
    else:
        from .blobs import blob_density
        p = _read_tif(a.pred).astype(np.float32) / 255
        r = _read_mask(a.region, a.level, a.z)[:p.shape[0], :p.shape[1]]
        d, ci, n = blob_density(p, a.threshold, r)
        print(json.dumps(dict(blobs_per_megapixel=d, ci95=ci, n_blobs=n)))


if __name__ == "__main__":
    main()
