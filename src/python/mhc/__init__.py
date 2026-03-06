__version__ = "0.1.0"

from .layer import MHCLayer

try:
    from .ops import (
        sinkhorn_knopp,
        rmsnorm,
        mhc_layer_fused,
        mhc_layer_fused_dynamic,
        mhc_layer_fused_dynamic_inference,
    )
except ImportError:  # mhc_cuda not built; fused ops unavailable
    sinkhorn_knopp = None
    rmsnorm = None
    mhc_layer_fused = None
    mhc_layer_fused_dynamic = None
    mhc_layer_fused_dynamic_inference = None

__all__ = [
    "__version__",
    "MHCLayer",
    "sinkhorn_knopp",
    "rmsnorm",
    "mhc_layer_fused",
    "mhc_layer_fused_dynamic",
    "mhc_layer_fused_dynamic_inference",
]
