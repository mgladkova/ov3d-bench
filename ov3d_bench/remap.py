"""Training-free open-vocabulary remapping of a frozen closed-vocabulary detector.

Take a detector's 3D boxes as given, crop each one from the image, encode the crop
with a contrastive vision-language encoder, and reassign it to whichever category
in the target vocabulary its embedding is closest to. Localization is untouched, so
comparing before and after isolates semantics from geometry. A detector trained
only on cars, pedestrians and cyclists can then be scored on trams.

This corrects two behaviours of the original research implementation.
`legacy_compat=True` reproduces the original behaviour for comparison.

  1. Identity and display text were conflated. A typo table meant to give the text
     encoder readable names was loaded inverted, and the prompt string and the
     category id were then both derived from it, so the encoder received the
     ground-truth spelling, typo included. Here identity always resolves by
     ground-truth name and display text is a separate lookup.

  2. A missing class prototype deleted the detection. Because identity resolved
     through the inverted names, three ScanNet-200 slots had no prototype and so
     silently swallowed every crop assigned to them. Here a missing prototype
     simply skips gating for that class.
"""
import hashlib
import json
import os
from collections import defaultdict

from .resources import resource_path

DEFAULT_PROMPT_NAMES = resource_path("prompt_names.json")


def load_display_names(path=None):
    """Ground-truth category name -> readable text for a prompt. Never for identity."""
    path = path or DEFAULT_PROMPT_NAMES
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("names", data) if isinstance(data, dict) else {}


def prompt_text(name, display_names, template="{name}"):
    """Render one category as the string handed to the text encoder."""
    readable = display_names.get(name, name).replace("_", " ")
    return template.format(name=readable)


def build_target_encoding(target_categories, name_to_id, display_names=None,
                          template="{name}", legacy_compat=False, legacy_typo_map=None):
    """Map the target vocabulary to (prompt strings, category ids).

    Identity is ALWAYS the target category itself. The research code routed it
    through an inverted typo map, so a name could resolve to a *different* class id
    whose prototype did not exist, and every crop assigned there was discarded.
    `legacy_compat` reproduces that for comparison.
    """
    display_names = display_names or {}
    inverted = {v: k for k, v in (legacy_typo_map or {}).items()} if legacy_compat else {}

    texts, category_ids = [], []
    for name in target_categories:
        if legacy_compat:
            resolved = inverted.get(name, name)
            texts.append(template.format(name=resolved.replace("_", " ")))
            category_ids.append(name_to_id.get(resolved, -1))
        else:
            texts.append(prompt_text(name, display_names, template))
            category_ids.append(name_to_id.get(name, -1))
    return texts, category_ids


def load_encoder(model_name, device):
    """Load a SigLIP or CLIP checkpoint. `transformers` is the [remap] extra."""
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "remapping needs `transformers`; install the [remap] extra"
        ) from exc

    if str(device) == "cpu":
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_name, device_map="auto").eval()
    return (model,
            AutoProcessor.from_pretrained(model_name),
            AutoTokenizer.from_pretrained(model_name, model_max_length=64))


def extract_crop(image, bbox_xywh, crop_scale=1.2, min_size=5):
    """Crop a box with context, clamped to the image. None if degenerate."""
    if bbox_xywh is None:
        return None
    x0, y0, w, h = bbox_xywh
    new_w, new_h = w * crop_scale, h * crop_scale
    if new_w < min_size or new_h < min_size:
        return None

    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    width, height = image.size
    left = max(0, int(cx - new_w / 2.0))
    top = max(0, int(cy - new_h / 2.0))
    right = min(width - 1, int(left + new_w))
    bottom = min(height - 1, int(top + new_h))
    left = max(0, right - int(new_w))
    top = max(0, bottom - int(new_h))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _encode_images(model, processor, crops, device, batch_size=64):
    import torch

    out = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).float().cpu()
        out.append(feats / feats.norm(dim=-1, keepdim=True))
    return torch.cat(out) if out else None


def build_class_prototypes(gt_data, images, image_root, model, processor, device,
                           crop_scale=1.2, allowed_category_ids=None,
                           max_samples_per_class=500):
    """Mean normalised embedding of ground-truth crops, per class.

    Used to reject an assignment whose crop looks nothing like that class. A class
    with no usable crops simply gets no prototype, and is then left ungated.
    """
    import torch
    from PIL import Image

    from .io import DEFAULT_FILTER_SETTINGS, pick_bbox2d

    categories = gt_data.get("categories", [])
    wanted = {c["id"] for c in categories
              if allowed_category_ids is None or c["id"] in allowed_category_ids}
    counts = defaultdict(int)

    crops, crop_class_ids = [], []
    for anno in gt_data.get("annotations", []):
        class_id = anno.get("category_id")
        if class_id not in wanted or counts[class_id] >= max_samples_per_class:
            continue
        info = images.get(anno.get("image_id"))
        path = _resolve_image(info, image_root)
        if path is None:
            continue
        bbox = pick_bbox2d(anno, DEFAULT_FILTER_SETTINGS)
        if bbox is None:
            continue
        try:
            with Image.open(path) as handle:
                crop = extract_crop(handle.convert("RGB"), bbox, crop_scale)
        except Exception:
            continue
        if crop is not None:
            crops.append(crop)
            crop_class_ids.append(class_id)
            counts[class_id] += 1

    if not crops:
        return {}

    features = _encode_images(model, processor, crops, device)
    by_class = defaultdict(list)
    for i, class_id in enumerate(crop_class_ids):
        by_class[class_id].append(features[i])

    prototypes = {}
    for class_id, feats in by_class.items():
        mean = torch.stack(feats).mean(dim=0)
        norm = mean.norm()
        if norm > 0:
            prototypes[class_id] = mean / norm
    return prototypes


def _resolve_image(info, image_root):
    if info is None:
        return None
    path = info.get("file_path") or info.get("file_name")
    if path is None:
        return None
    full = path if (image_root is None and os.path.isabs(path)) \
        else os.path.join(image_root or "", path)
    return full if os.path.exists(full) else None


def remap_predictions(pred_by_image, images, target_categories, name_to_id,
                      image_root, model_name="google/siglip2-so400m-patch16-naflex",
                      device=None, bbox_format="xywh", crop_scale=1.2, max_dets=100,
                      similarity_threshold=0.05, prototype_threshold=0.8,
                      class_prototypes=None, display_names=None,
                      template="{name}", model=None, processor=None, tokenizer=None,
                      legacy_compat=False, legacy_typo_map=None, limit_images=0):
    """Reassign each detection to its best-matching category. Modifies in place.

    Returns (pred_by_image, stats). With `legacy_compat`, reproduces the research
    code's behaviour: identity resolved through the inverted typo map, and a
    missing prototype discarding the detection.
    """
    import torch
    from PIL import Image

    from .io import prediction_bbox2d_xywh

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, processor, tokenizer = load_encoder(model_name, device)
    display_names = {} if display_names is None else display_names

    texts, category_ids = build_target_encoding(
        target_categories, name_to_id, display_names, template,
        legacy_compat, legacy_typo_map,
    )

    tokens = tokenizer(texts, padding="max_length", truncation=True,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**tokens).float().cpu()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    stats = defaultdict(int)
    for n_image, (image_id, instances) in enumerate(pred_by_image.items()):
        if limit_images and stats["images"] >= limit_images:
            break
        if not instances:
            continue
        path = _resolve_image(images.get(image_id), image_root)
        if path is None:
            continue
        image = Image.open(path).convert("RGB")

        ordered = sorted(instances, key=lambda i: i.get("score", 0.0), reverse=True)[:max_dets]
        crops, crop_idx = [], []
        for idx, inst in enumerate(ordered):
            crop = extract_crop(image, prediction_bbox2d_xywh(inst, bbox_format), crop_scale)
            if crop is not None:
                crops.append(crop)
                crop_idx.append(idx)
        if not crops:
            continue

        stats["images"] += 1
        stats["crops"] += len(crops)
        features = _encode_images(model, processor, crops, device)
        similarity = features @ text_features.T
        best = similarity.argmax(dim=-1)

        for row, (idx, class_idx) in enumerate(zip(crop_idx, best.tolist())):
            if float(similarity[row, class_idx]) < similarity_threshold:
                ordered[idx]["category_id"] = -1
                stats["below_similarity"] += 1
                continue

            category_id = category_ids[class_idx]
            prototype = class_prototypes.get(category_id) if class_prototypes else None

            if prototype is None:
                if legacy_compat and class_prototypes:
                    # The bug: no prototype meant the detection was thrown away.
                    ordered[idx]["category_id"] = -1
                    stats["no_prototype_discarded"] += 1
                    continue
                stats["no_prototype_ungated"] += 1
            elif float(torch.dot(features[row], prototype)) < prototype_threshold:
                ordered[idx]["category_id"] = -1
                stats["below_prototype"] += 1
                continue

            ordered[idx]["category_id"] = category_id
            stats["applied"] += 1

    return pred_by_image, dict(stats)


def cache_key(pred_path, gt_json, target_categories, **options):
    """Stable key for a remap result, so an expensive pass is computed once."""
    blob = json.dumps({
        "pred": os.path.abspath(pred_path),
        "pred_mtime": os.path.getmtime(pred_path) if os.path.exists(pred_path) else None,
        "gt": os.path.abspath(gt_json),
        "categories": list(target_categories),
        "options": {k: options[k] for k in sorted(options)},
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
