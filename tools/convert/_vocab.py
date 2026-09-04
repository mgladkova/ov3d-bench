"""Loads the shipped 273-class vocabulary, replacing omni3d-xl's class_mapping.py."""
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "ov3d_bench", "data", "class_mapping.json")
with open(_PATH) as _f:
    _DATA = json.load(_f)

CLASSES = _DATA["classes"]
CLASS_MAPPING = _DATA["class_mapping"]
