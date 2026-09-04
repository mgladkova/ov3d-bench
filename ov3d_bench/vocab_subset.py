"""Sensitivity of reported AP to which categories are scored.

The base/novel protocol reports AP over a held-out subset of the vocabulary, and
those numbers are not comparable across papers because each withholds different
classes. This isolates the evaluation-side effect: hold the model and its
predictions fixed, then re-score over subsets.

The re-scoring is exact, not sampled inference. Omni3D's aggregate AP is the
unweighted mean of per-category AP over classes with ground truth, and per-category
AP does not depend on which other classes are present, so AP(subset) is just the
mean of those categories' APs.
"""
import numpy as np

from . import io as bio
from . import metrics

DEFAULT_SUBSET_SIZES = (5, 10, 20, 40)


def run_vocab_subset(gt_json, pred, target_cats=None, dataset_name=None,
                     bbox_format="xywh", corner_order="omni3d", max_dets_per_image=100,
                     subset_sizes=DEFAULT_SUBSET_SIZES, n_samples=5000, seed=0,
                     strict_categories=True):
    """Report AP over random, adverse and favourable subsets of the vocabulary."""
    gt_data, images, gt_by_image, target_categories, id_to_name, name_to_id = bio.load_gt(
        gt_json, target_cats, dataset_name=dataset_name, strict=strict_categories
    )
    by_image = bio.load_predictions(pred, 0.0, corner_order, max_per_image=max_dets_per_image)
    target_ids = {name_to_id[n] for n in target_categories if n in name_to_id}

    aggregate, per_cat = metrics.per_category_ap(
        gt_json, target_categories,
        metrics.prepare_omni3d_results(by_image, bbox_format, target_ids),
        id_to_name,
    )

    valid = np.array([v for v in per_cat.values() if v == v])   # drop NaN (no GT)
    rng = np.random.default_rng(seed)

    rows = []
    for k in subset_sizes:
        if k > len(valid):
            continue
        sampled = np.array([rng.choice(valid, size=k, replace=False).mean()
                            for _ in range(n_samples)])
        ordered = np.sort(valid)
        best, worst = ordered[-k:].mean(), ordered[:k].mean()
        rows.append({
            "k": k,
            "rand_mean": float(sampled.mean()),
            "rand_std": float(sampled.std()),
            "rand_min": float(sampled.min()),
            "rand_max": float(sampled.max()),
            "best": float(best),
            "worst": float(worst),
            "cv": float(100 * sampled.std() / sampled.mean()) if sampled.mean() else float("nan"),
            "ratio": float(best / worst) if worst else float("inf"),
        })

    return {
        "dataset": dataset_name,
        "pred": pred,
        "aggregate": aggregate,
        "n_valid": int(len(valid)),
        "percat": {k: v for k, v in per_cat.items() if v == v},
        "rows": rows,
    }
