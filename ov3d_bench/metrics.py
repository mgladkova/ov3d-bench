"""The benchmark's diagnostics: mAP3D, class-agnostic recall, and confusion.

`eval_ap3d` is the only entry point into the vendored CC-BY-NC evaluator
(`ov3d_bench.omni3d`). Everything else here is Apache 2.0 and depends on it only
through that call, so a permissive reimplementation of AP3D is a drop-in swap.
"""
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .io import DEFAULT_FILTER_SETTINGS, Omni3DDataset, prediction_bbox2d_xywh
from .omni3d import load as _load_omni3d_backend

AP_METRICS = ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf"]


def group_predictions(by_image, bbox_format="xywh"):
    """Normalise raw prediction instances into the shape the matchers expect."""
    grouped = defaultdict(list)
    for image_id, instances in by_image.items():
        for inst in instances:
            grouped[image_id].append({
                "category_id": inst.get("category_id"),
                "score": inst.get("score", 0.0),
                "bbox2d": prediction_bbox2d_xywh(inst, bbox_format),
                "bbox3d": inst.get("bbox3D"),
                "raw": inst,
            })
    return grouped


# --------------------------------------------------------------------------
# Assignment rules
# --------------------------------------------------------------------------

def hungarian_match(iou, min_iou):
    """Globally IoU-optimal and confidence-blind."""
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(1.0 - iou)
    return [(r, c, float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]


def greedy_confidence_match(iou, scores, min_iou):
    """nuScenes-style: predictions claim ground truth in descending confidence.

    Not globally optimal. A low-confidence prediction can lose a ground-truth box
    to a higher-confidence one that fits it worse, so confidence affects whether a
    box is credited at all.
    """
    if iou.size == 0:
        return []
    matches, taken = [], set()
    for r in sorted(range(iou.shape[0]), key=lambda i: scores[i], reverse=True):
        best_c, best_iou = None, -1.0
        for c in range(iou.shape[1]):
            if c not in taken and iou[r, c] > best_iou:
                best_c, best_iou = c, iou[r, c]
        if best_c is not None and best_iou >= min_iou:
            taken.add(best_c)
            matches.append((r, best_c, float(best_iou)))
    return matches


def _iou3d(pred_boxes, gt_boxes):
    _, box3d_overlap = _load_omni3d_backend()
    pred = torch.tensor(pred_boxes, dtype=torch.float32)
    gt = torch.tensor(gt_boxes, dtype=torch.float32)
    return box3d_overlap(pred, gt).cpu().numpy()


# --------------------------------------------------------------------------
# Diagnostic 1: class-agnostic localization
# --------------------------------------------------------------------------

def eval_box_iou3d(gt_by_image, preds_by_image, iou_min=0.0, class_agnostic=False,
                   matching_mode="hungarian"):
    """Recall of ground-truth boxes matched by some prediction.

    With `class_agnostic`, a single assignment per image runs over all boxes
    regardless of category, so a box fails only if it is poorly *located*, never
    because it is poorly *labelled*. That is what isolates geometry from semantics.
    """
    if matching_mode not in ("hungarian", "greedy"):
        raise ValueError(f"bad matching_mode: {matching_mode}")

    total_iou, matched, total_preds, total_gts = 0.0, 0, 0, 0

    for image_id, gts in gt_by_image.items():
        gt_items = [g for g in gts if g.get("bbox3d") is not None]
        pred_items = [p for p in preds_by_image.get(image_id, []) if p.get("bbox3d") is not None]
        total_gts += len(gt_items)
        total_preds += len(pred_items)
        if not gt_items or not pred_items:
            continue

        if class_agnostic:
            groups = [(gt_items, pred_items)]
        else:
            gts_by_class = defaultdict(list)
            for g in gt_items:
                gts_by_class[g.get("category_id")].append(g)
            preds_by_class = defaultdict(list)
            for p in pred_items:
                preds_by_class[p.get("category_id")].append(p)
            groups = [(v, preds_by_class.get(k, [])) for k, v in gts_by_class.items()]

        for gts_c, preds_c in groups:
            if not gts_c or not preds_c:
                continue
            iou = _iou3d([p["bbox3d"] for p in preds_c], [g["bbox3d"] for g in gts_c])
            if matching_mode == "greedy":
                found = greedy_confidence_match(iou, [p.get("score", 0.0) for p in preds_c], iou_min)
            else:
                found = hungarian_match(iou, iou_min)
            for _, _, value in found:
                total_iou += value
                matched += 1

    return {
        "avg_iou3d": total_iou / matched if matched else 0.0,
        "recall": matched / total_gts if total_gts else 0.0,
        "matches": matched,
        "total_preds": total_preds,
        "total_gts": total_gts,
    }


# --------------------------------------------------------------------------
# Diagnostic 2: semantic confusion
# --------------------------------------------------------------------------

def eval_iou3d_confusion(gt_by_image, preds_by_image, target_categories, id_to_name,
                         iou3d_min=0.0):
    """Row-normalised confusion over ground truth that was localised successfully.

    Each ground-truth box is matched one-to-one to its highest-IoU prediction and
    unmatched boxes are discarded, so the matrix shows only semantic error. The
    diagonal is per-class recall among localised boxes.
    """
    index = {name: i for i, name in enumerate(target_categories)}
    n = len(target_categories)
    matrix = np.zeros((n, n), dtype=np.float32)
    matched_per_class = np.zeros(n, dtype=np.int32)

    for image_id, gts in gt_by_image.items():
        gt_items = [g for g in gts if g.get("bbox3d") is not None]
        pred_items = [p for p in preds_by_image.get(image_id, []) if p.get("bbox3d") is not None]
        if not gt_items or not pred_items:
            continue

        iou = _iou3d([p["bbox3d"] for p in pred_items], [g["bbox3d"] for g in gt_items])
        rows, cols = linear_sum_assignment(1.0 - iou)
        for pred_idx, gt_idx in zip(rows, cols):
            if iou[pred_idx, gt_idx] < iou3d_min:
                continue
            gt_label = id_to_name.get(gt_items[gt_idx]["category_id"], "unknown")
            if gt_label not in index:
                continue
            pred_label = id_to_name.get(pred_items[pred_idx]["category_id"], "unknown")
            pred_col = index.get(pred_label, -1)
            if pred_col >= 0:
                matrix[index[gt_label], pred_col] += 1
                matched_per_class[index[gt_label]] += 1

    normalized = matrix.copy()
    for i in range(n):
        if matched_per_class[i] > 0:
            normalized[i] /= float(matched_per_class[i])

    return {
        "labels_gt": target_categories,
        "labels_pred": target_categories,
        "matrix": matrix.tolist(),
        "matrix_norm": normalized.tolist(),
        "matched_gts_per_class": matched_per_class.tolist(),
    }


# --------------------------------------------------------------------------
# Diagnostic 3: mAP3D  (the only path into the vendored evaluator)
# --------------------------------------------------------------------------

def prepare_omni3d_results(by_image, bbox_format, target_id_set):
    """Flatten predictions into the evaluator's result records."""
    results = []
    for image_id, instances in by_image.items():
        for inst in instances:
            bbox3d = inst.get("bbox3D")
            bbox = prediction_bbox2d_xywh(inst, bbox_format)
            if bbox3d is None or bbox is None:
                continue
            if target_id_set and inst.get("category_id") not in target_id_set:
                continue
            results.append({
                "image_id": image_id,
                "category_id": inst.get("category_id"),
                "bbox": bbox,
                "score": inst.get("score", 0.0),
                "depth": float(np.array(bbox3d)[:, 2].mean()),
                "bbox3D": bbox3d,
            })
    return results


def eval_ap3d(gt_json_path, target_categories, omni_results):
    """Omni3D mAP3D: AP averaged over 3D IoU thresholds 0.05 to 0.5."""
    if not omni_results:
        return {m: float("nan") for m in AP_METRICS}

    settings = dict(DEFAULT_FILTER_SETTINGS)
    settings["category_names"] = target_categories

    Omni3Deval, _ = _load_omni3d_backend()
    gt = Omni3DDataset(gt_json_path, filter_settings=settings)
    dt = gt.loadRes(omni_results)

    ev = Omni3Deval(gt, dt, iouType="bbox", mode="3D")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return {m: float(ev.stats[i] * 100 if ev.stats[i] >= 0 else np.nan)
            for i, m in enumerate(AP_METRICS)}


def per_category_ap(gt_json_path, target_categories, omni_results, id_to_name):
    """Aggregate mAP3D plus AP for each category.

    Per-category AP is independent of which other categories are in the evaluation
    set, which is what makes vocabulary-subset re-scoring exact rather than an
    approximation. Categories with no ground truth come back as NaN.
    """
    if not omni_results:
        return float("nan"), {}

    settings = dict(DEFAULT_FILTER_SETTINGS)
    settings["category_names"] = target_categories

    Omni3Deval, _ = _load_omni3d_backend()
    gt = Omni3DDataset(gt_json_path, filter_settings=settings)
    dt = gt.loadRes(omni_results)
    ev = Omni3Deval(gt, dt, iouType="bbox", mode="3D")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    params = ev.params
    precision = ev.eval["precision"]          # (T, R, K, A, M)
    max_det_idx = [i for i, m in enumerate(params.maxDets) if m == 100][0]

    per = {}
    for k, cat_id in enumerate(params.catIds):
        sliced = precision[:, :, k, 0, max_det_idx]
        aps = [np.mean(row[row > -1]) for row in sliced if len(row[row > -1])]
        name = id_to_name.get(cat_id, cat_id)
        per[name] = float(np.mean(aps) * 100) if aps else float("nan")

    return float(ev.stats[0]) * 100, per
