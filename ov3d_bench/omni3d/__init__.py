"""Vendored Omni3D / Cube R-CNN evaluation code.

LICENCE: everything in this subpackage is derived from
https://github.com/facebookresearch/omni3d and is licensed CC-BY-NC 4.0
(see LICENSE.md here), NOT the Apache 2.0 that covers the rest of OV3D-Bench.

The import is deferred rather than eager. `omni3d_eval` pulls in PyTorch3D, which
is the most fragile dependency in the stack, so importing it at package level would
mean a mismatched PyTorch3D breaks `--help`, the ontology, the loaders and every
other thing that never touches 3D IoU. Call `load()` at the point of use instead.
"""

_CACHE = {}

_HINT = (
    "3D IoU needs PyTorch3D, and it must be a build matching your torch "
    "installation (CPU vs CUDA, and the torch version). A CUDA PyTorch3D against a "
    "CPU torch fails with a missing libc10_cuda.so. See the README for install "
    "instructions.\n  original error: {err}"
)


def load():
    """Return (Omni3Deval, box3d_overlap), importing PyTorch3D on first use."""
    if not _CACHE:
        try:
            from .omni3d_eval import Omni3Deval, box3d_overlap
        except ImportError as exc:  # missing, or built against a different torch
            raise ImportError(_HINT.format(err=exc)) from exc
        _CACHE["Omni3Deval"] = Omni3Deval
        _CACHE["box3d_overlap"] = box3d_overlap
    return _CACHE["Omni3Deval"], _CACHE["box3d_overlap"]


def available():
    """True if the 3D IoU backend can be imported."""
    try:
        load()
        return True
    except ImportError:
        return False
