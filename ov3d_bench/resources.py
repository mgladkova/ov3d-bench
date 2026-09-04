"""Locate data files shipped inside the package.

These live under `ov3d_bench/data/` rather than at the repository root so they
survive a wheel install, where nothing outside the package is present.
"""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def resource_path(name):
    """Absolute path to a shipped data file, e.g. resource_path('datasets.json')."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"packaged data file not found: {name} (looked in {DATA_DIR})")
    return path
