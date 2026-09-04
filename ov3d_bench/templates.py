"""Prompt-template robustness: how much does mAP3D move when only the wording does?

Two modes, because detectors consume a prompt in fundamentally different ways.

  predset      One prediction file per template. Required for prompt-conditioned
               detectors (GroundingDINO, SAM 3 style), which re-detect for every
               prompt, so a single prediction set cannot be re-scored.

  contrastive  One prediction set, re-scored against each template's text
               embedding. Valid only for encoders that remap fixed detections
               (SigLIPv2, CLIP), where the boxes never depend on the wording.
               Provided by `ov3d_bench.remap`.

Reported as mAP3D per template group and as the coefficient of variation across
all templates. Lower CV is more robust.
"""
import json
import os
import statistics

from . import io as bio
from . import metrics

DEFAULT_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "data", "prompt_templates.json")


def load_template_bank(path=None):
    """Load templates grouped by descriptiveness. Keys starting with _ are metadata."""
    with open(path or DEFAULT_TEMPLATES) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def coefficient_of_variation(values):
    """Standard deviation over mean, as a percentage. The scalar robustness summary."""
    usable = [v for v in values if v == v]
    if len(usable) < 2:
        return float("nan")
    mean = statistics.mean(usable)
    if not mean:
        return float("nan")
    return 100.0 * statistics.pstdev(usable) / mean


def summarize_groups(ap_by_template, groups):
    """Aggregate per-template AP into per-group statistics plus an overall CV.

    `ap_by_template` maps a template string to its mAP3D; `groups` maps a group
    name to the templates in it.
    """
    per_group, every = {}, []
    for name, templates in groups.items():
        aps = [ap_by_template[t] for t in templates if t in ap_by_template]
        if not aps:
            continue
        every.extend(aps)
        per_group[name] = {
            "mean": statistics.mean(aps),
            "std": statistics.pstdev(aps),
            "n": len(aps),
            "aps": aps,
        }
    return {"per_group": per_group, "cv_percent": coefficient_of_variation(every)}


def run_templates_predset(gt_json, preds, target_cats=None, dataset_name=None,
                          bbox_format="xywh", corner_order="omni3d",
                          score_threshold=0.0, max_dets_per_image=100,
                          labels=None, strict_categories=True):
    """Score one prediction file per template and summarise the spread."""
    gt_data, images, gt_by_image, target_categories, id_to_name, name_to_id = bio.load_gt(
        gt_json, target_cats, dataset_name=dataset_name, strict=strict_categories
    )
    target_ids = {name_to_id[n] for n in target_categories if n in name_to_id}

    per_run = []
    for i, pred_path in enumerate(preds):
        by_image = bio.load_predictions(
            pred_path, score_threshold, corner_order, max_per_image=max_dets_per_image
        )
        ap = metrics.eval_ap3d(
            gt_json, target_categories,
            metrics.prepare_omni3d_results(by_image, bbox_format, target_ids),
        )
        per_run.append({
            "pred_path": pred_path,
            "label": labels[i] if labels and i < len(labels) else os.path.basename(pred_path),
            "metrics": ap,
        })

    keys = metrics.AP_METRICS
    values = {k: [r["metrics"][k] for r in per_run if r["metrics"][k] == r["metrics"][k]]
              for k in keys}
    mean = {k: statistics.mean(v) if v else float("nan") for k, v in values.items()}
    std = {k: statistics.pstdev(v) if len(v) > 1 else 0.0 for k, v in values.items()}

    return {
        "mode": "predset",
        "dataset": dataset_name,
        "num_runs": len(per_run),
        "metrics_mean": mean,
        "metrics_std": std,
        "cv_percent": coefficient_of_variation(values["AP"]),
        "per_run": per_run,
    }


def run_templates_contrastive(gt_json, pred, image_root, target_cats=None,
                              dataset_name=None, model_name=None, backend="siglip",
                              bbox_format="xywh", corner_order="omni3d",
                              max_dets=100, crop_scale=1.2, similarity_threshold=0.05,
                              templates_path=None, device=None, strict_categories=True):
    """Re-score ONE prediction set against every template's text embedding.

    Each crop is encoded once and reused for all templates, so this measures the
    text encoder's prompt sensitivity in isolation: the boxes are identical across
    templates and only the wording changes. Valid only for encoders that remap
    fixed detections; a prompt-conditioned detector must use predset mode instead.
    """
    import torch

    from . import io as bio
    from . import metrics
    from .remap import (_encode_images, _resolve_image, extract_crop, load_display_names,
                        load_encoder, prompt_text)
    from PIL import Image

    defaults = {"siglip": "google/siglip2-so400m-patch16-naflex",
                "clip": "openai/clip-vit-large-patch14"}
    model_name = model_name or defaults[backend]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    gt_data, images, gt_by_image, target_categories, id_to_name, name_to_id = bio.load_gt(
        gt_json, target_cats, dataset_name=dataset_name, strict=strict_categories
    )
    target_ids = {name_to_id[n] for n in target_categories if n in name_to_id}
    by_image = bio.load_predictions(pred, 0.0, corner_order, max_per_image=max_dets)
    by_image = {i: v for i, v in by_image.items() if i in gt_by_image}

    model, processor, tokenizer = load_encoder(model_name, device)
    display_names = load_display_names()

    # Encode every crop once.
    layout, feature_blocks = [], []
    for image_id, instances in by_image.items():
        path = _resolve_image(images.get(image_id), image_root)
        if path is None:
            continue
        image = Image.open(path).convert("RGB")
        ordered = sorted(instances, key=lambda i: i.get("score", 0.0), reverse=True)[:max_dets]
        crops, kept = [], []
        for inst in ordered:
            crop = extract_crop(image, bio.prediction_bbox2d_xywh(inst, bbox_format), crop_scale)
            if crop is not None:
                crops.append(crop)
                kept.append(inst)
        if not crops:
            continue
        feature_blocks.append(_encode_images(model, processor, crops, device))
        layout.extend(kept)
    if not layout:
        raise ValueError("no crops could be encoded; check --image-root")
    features = torch.cat(feature_blocks)

    groups = load_template_bank(templates_path)
    ap_by_template, per_template = {}, {}
    for group, templates in groups.items():
        for template in templates:
            texts = [prompt_text(n, display_names, template) for n in target_categories]
            tokens = tokenizer(texts, padding="max_length", truncation=True,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                text_features = model.get_text_features(**tokens).float().cpu()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = features @ text_features.T
            best = similarity.argmax(dim=-1)
            peak = similarity.max(dim=-1).values
            for row, inst in enumerate(layout):
                if float(peak[row]) < similarity_threshold:
                    inst["category_id"] = -1
                else:
                    inst["category_id"] = name_to_id.get(target_categories[int(best[row])], -1)

            ap = metrics.eval_ap3d(
                gt_json, target_categories,
                metrics.prepare_omni3d_results(by_image, bbox_format, target_ids),
            )["AP"]
            ap_by_template[template] = ap
            per_template[f"{group} | {template}"] = ap

    summary = summarize_groups(ap_by_template, groups)
    return {
        "mode": "contrastive",
        "backend": backend,
        "model": model_name,
        "dataset": dataset_name,
        "n_images": len(feature_blocks),
        "n_crops": len(layout),
        "per_template": per_template,
        **summary,
    }
