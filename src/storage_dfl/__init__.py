"""Decision-focused scenario construction for shared-storage planning."""

import os

# On Windows, the conda NumPy build and the pip PyTorch wheel can each ship an
# Intel OpenMP runtime.  BO first reaches MKL through ``numpy.linalg`` only after
# its initial evaluations, so the duplicate-runtime failure otherwise appears
# late in a run.  The GP matrices are tiny; using MKL's sequential layer avoids
# loading its OpenMP runtime without enabling the unsafe KMP duplicate override.
if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

__all__ = ["config", "pipeline"]
