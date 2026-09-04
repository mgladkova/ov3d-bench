"""Vendored Omni3D / Cube R-CNN evaluation code.

LICENCE: everything in this subpackage is derived from
https://github.com/facebookresearch/omni3d and is licensed CC-BY-NC 4.0
(see LICENSE.md here), NOT the Apache 2.0 that covers the rest of OV3D-Bench.

It is isolated here, and reached only through `ov3d_bench.metrics.eval_ap3d`,
so that a future permissive reimplementation is a drop-in replacement.
"""
from .omni3d_eval import Omni3Deval, box3d_overlap  # noqa: F401
