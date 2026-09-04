"""One-pass evaluation: mAP3D, class-agnostic recall and confusion together."""
import json
import os

from . import io as bio
from . import metrics


def apply_target_aware_filter(gt_by_image, by_image):
    """The target-aware oracle: keep only predictions whose class occurs in that image's GT.

    This is the protocol used by OVMono3D and DetAny3D. It deletes hallucinated
    classes instead of penalising them, which is why it inflates AP. Provided so
    the inflation can be measured, not because it is recommended.
    """
    kept = dropped = 0
    for image_id, instances in by_image.items():
        gt_classes = {g.get("category_id") for g in gt_by_image.get(image_id, [])}
        keep = [i for i in instances if i.get("category_id") in gt_classes]
        dropped += len(instances) - len(keep)
        kept += len(keep)
        by_image[image_id] = keep
    return kept, dropped


def run_eval(gt_json, pred, target_cats=None, dataset_name=None, bbox_format="xywh",
             corner_order="omni3d", score_threshold=0.0, max_dets_per_image=100,
             iou3d_min=0.0, class_agnostic=False, matching_mode="hungarian",
             target_aware=False, remap_sequential_ids=False, stream=False,
             strict_categories=True, outdir=None, visualize=False, image_root=None,
             vis_every_n=1, vis_max_images=0, vis_nms_iou=0.5,
             remap=None, remap_model=None, remap_crop_scale=1.2,
             remap_similarity_threshold=0.05, remap_prototype_threshold=0.8,
             remap_max_dets=100, remap_legacy_compat=False):
    """Evaluate one prediction file and return every diagnostic in one dict."""
    gt_data, images, gt_by_image, target_categories, id_to_name, name_to_id = bio.load_gt(
        gt_json, target_cats, dataset_name=dataset_name, strict=strict_categories
    )

    if stream:
        if not pred.endswith(".jsonl"):
            raise ValueError("streaming requires a .jsonl prediction file")
        by_image = bio.stream_predictions_jsonl(
            pred, score_threshold, corner_order, max_dets_per_image
        )
    else:
        by_image = bio.load_predictions(
            pred, score_threshold, corner_order, max_per_image=max_dets_per_image
        )

    if remap_sequential_ids:
        by_image = bio.remap_sequential_category_ids(by_image, target_categories, name_to_id)

    remap_stats = None
    if remap:
        if not image_root:
            raise ValueError("remapping needs --image-root")
        from .remap import (build_class_prototypes, load_display_names, load_encoder,
                            remap_predictions)

        defaults = {"siglip": "google/siglip2-so400m-patch16-naflex",
                    "clip": "openai/clip-vit-large-patch14"}
        model_name = remap_model or defaults[remap]
        model, processor, tokenizer = load_encoder(model_name, None)
        target_ids_for_proto = {name_to_id[n] for n in target_categories if n in name_to_id}
        prototypes = build_class_prototypes(
            gt_data, images, image_root, model, processor, None,
            crop_scale=remap_crop_scale, allowed_category_ids=target_ids_for_proto,
        )
        by_image, remap_stats = remap_predictions(
            by_image, images, target_categories, name_to_id, image_root,
            model_name=model_name, bbox_format=bbox_format, crop_scale=remap_crop_scale,
            max_dets=remap_max_dets, similarity_threshold=remap_similarity_threshold,
            prototype_threshold=remap_prototype_threshold, class_prototypes=prototypes,
            display_names=load_display_names(), model=model, processor=processor,
            tokenizer=tokenizer, legacy_compat=remap_legacy_compat,
        )

    protocol = {"target_aware": target_aware}
    if remap_stats is not None:
        protocol["remap"] = {"backend": remap, **remap_stats}
    if target_aware:
        kept, dropped = apply_target_aware_filter(gt_by_image, by_image)
        protocol.update(kept=kept, dropped=dropped)

    grouped = metrics.group_predictions(by_image, bbox_format)

    box_iou3d = metrics.eval_box_iou3d(
        gt_by_image, grouped, iou_min=iou3d_min,
        class_agnostic=class_agnostic, matching_mode=matching_mode,
    )
    confusion = metrics.eval_iou3d_confusion(
        gt_by_image, grouped, target_categories, id_to_name, iou3d_min=iou3d_min
    )
    target_ids = {name_to_id[n] for n in target_categories if n in name_to_id}
    ap3d = metrics.eval_ap3d(
        gt_json, target_categories,
        metrics.prepare_omni3d_results(by_image, bbox_format, target_ids),
    )

    results = {
        "box_iou3d": box_iou3d,
        "iou3d_confusion": confusion,
        "ap3d": ap3d,
        "protocol": protocol,
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    if visualize:
        if not outdir:
            raise ValueError("--visualize needs --outdir")
        if not image_root:
            raise ValueError("--visualize needs --image-root")
        from .vis import visualize as render

        results["visualized"] = render(
            gt_data, images, by_image, id_to_name, image_root,
            os.path.join(outdir, "visualizations"),
            score_threshold=score_threshold, nms_iou=vis_nms_iou,
            every_n=vis_every_n, max_images=vis_max_images, bbox_format=bbox_format,
        )

    return results


def run_per_category(gt_json, pred, target_cats=None, dataset_name=None,
                     bbox_format="xywh", corner_order="omni3d", max_dets_per_image=100,
                     strict_categories=True):
    """Per-category AP under both protocols, and the inflation the oracle buys.

    The target-aware gain concentrates on classes the detector confuses with a
    dominant neighbour: the oracle deletes those false positives wherever the
    confused class is absent from an image.
    """
    import copy

    gt_data, images, gt_by_image, target_categories, id_to_name, name_to_id = bio.load_gt(
        gt_json, target_cats, dataset_name=dataset_name, strict=strict_categories
    )
    by_image = bio.load_predictions(pred, 0.0, corner_order, max_per_image=max_dets_per_image)
    target_ids = {name_to_id[n] for n in target_categories if n in name_to_id}

    def score(predictions):
        return metrics.per_category_ap(
            gt_json, target_categories,
            metrics.prepare_omni3d_results(predictions, bbox_format, target_ids),
            id_to_name,
        )

    dataset_level = copy.deepcopy(by_image)
    agg_dlp, per_dlp = score(dataset_level)

    target_aware = copy.deepcopy(by_image)
    kept, dropped = apply_target_aware_filter(gt_by_image, target_aware)
    agg_tap, per_tap = score(target_aware)

    rows = []
    for name in per_dlp:
        d, t = per_dlp.get(name, float("nan")), per_tap.get(name, float("nan"))
        rows.append({"category": name, "dataset_level": d, "target_aware": t, "delta": t - d})
    rows.sort(key=lambda r: (r["delta"] if r["delta"] == r["delta"] else -1e9), reverse=True)

    return {
        "dataset": dataset_name,
        "aggregate": {"dataset_level": agg_dlp, "target_aware": agg_tap,
                      "inflation": (agg_tap / agg_dlp) if agg_dlp else float("nan")},
        "target_aware_filter": {"kept": kept, "dropped": dropped},
        "per_category": rows,
    }
