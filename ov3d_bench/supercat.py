"""Confusion aggregated onto a coarse, vocabulary-agnostic ontology.

Fine-grained confusion matrices cannot be compared across datasets whose
vocabularies differ by two orders of magnitude. Projecting onto a small shared
ontology answers a question that survives that difference: when a detector is
wrong, is it wrong within an object family or across families? A strong diagonal
means coarse identity is recovered and the residual error is fine-grained naming.
"""
import json
import os

import numpy as np

from .resources import resource_path

DEFAULT_ONTOLOGY = resource_path("supercategories.json")


def load_ontology(path=None):
    """Return (supercategories, rules, overrides)."""
    with open(path or DEFAULT_ONTOLOGY) as f:
        data = json.load(f)
    rules = [(name, tuple(keys)) for name, keys in data["rules"]]
    return data["supercategories"], rules, data.get("overrides", {})


def to_super(name, rules, fallback="other", overrides=None):
    """Assign a fine-grained class to a super-category.

    An exact-name override wins outright; otherwise the first matching substring
    rule wins. Overrides exist because some names embed a substring belonging to
    another family, e.g. `message_board_trailer` is road-side signage rather than a
    vehicle, and `mobile_pedestrian_crossing_sign` is a sign rather than a person.
    """
    key = str(name).lower()
    if overrides and key in overrides:
        return overrides[key]
    for super_name, keys in rules:
        if any(k in key for k in keys):
            return super_name
    return fallback


def aggregate_confusions(confusions, ontology_path=None):
    """Sum row-normalised super-category confusion over many fine-grained matrices.

    `confusions` is an iterable of the `iou3d_confusion` blocks that `eval`
    produces. Counts are pooled before normalising, so datasets contribute in
    proportion to how many boxes they actually localised.
    """
    supers, rules, overrides = load_ontology(ontology_path)
    index = {s: i for i, s in enumerate(supers)}
    fallback = supers[-1]
    totals = np.zeros((len(supers), len(supers)), dtype=float)
    unmapped = {}

    for block in confusions:
        labels_gt = block.get("labels_gt")
        labels_pred = block.get("labels_pred")
        matrix = block.get("matrix")
        if not labels_gt or not matrix:
            continue
        counts = np.asarray(matrix, dtype=float)
        pred_super = [index[to_super(p, rules, fallback, overrides)] for p in labels_pred]
        for i, gt_name in enumerate(labels_gt):
            row = counts[i]
            if not row.any():
                continue
            gt_super = to_super(gt_name, rules, fallback, overrides)
            if gt_super == fallback:
                unmapped[gt_name] = unmapped.get(gt_name, 0.0) + float(row.sum())
            gi = index[gt_super]
            for j, value in enumerate(row):
                if value > 0:
                    totals[gi, pred_super[j]] += value

    row_sums = totals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = totals / row_sums
    cross_rate = 1.0 - (np.trace(totals) / totals.sum()) if totals.sum() else float("nan")

    return {
        "supercategories": supers,
        "matrix": totals.tolist(),
        "matrix_norm": normalized.tolist(),
        "cross_supercategory_rate": float(cross_rate),
        "unmapped_gt_mass": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])),
    }


def run_supercat(results_paths, ontology_path=None):
    """Aggregate over a list of results.json files produced by `eval`."""
    blocks = []
    for path in results_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            block = json.load(f).get("iou3d_confusion")
        if block:
            blocks.append(block)
    if not blocks:
        raise ValueError("no results.json with an iou3d_confusion block was found")
    out = aggregate_confusions(blocks, ontology_path)
    out["n_sources"] = len(blocks)
    return out


def format_matrix(result):
    supers = result["supercategories"]
    norm = result["matrix_norm"]
    lines = ["Row-normalised super-category confusion (diagonal = same super-category):",
             "            " + " ".join(f"{s[:6]:>6}" for s in supers)]
    for i, s in enumerate(supers):
        lines.append(f"{s:>10}  " + " ".join(f"{norm[i][j]:6.2f}" for j in range(len(supers))))
    lines.append(f"cross-super-category rate = {result['cross_supercategory_rate']:.3f}")
    return "\n".join(lines)
