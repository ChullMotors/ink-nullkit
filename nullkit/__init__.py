"""ink-nullkit: block-bootstrap comparisons and ink-free null controls for ink-detection models."""
from .bootstrap import auroc_from_hists, block_hists, paired_auroc_diff
from .blobs import blob_density
from .phase_null import phase_randomize
from .orientation import choose_direction

__all__ = ["auroc_from_hists", "block_hists", "paired_auroc_diff", "blob_density", "phase_randomize", "choose_direction"]
