"""Loading of ground truth, predictions and target vocabularies.

The dataset class here is an independent implementation of the Omni3D annotation
filtering rules (it replaces the Meta-derived loader), so this file is Apache 2.0.
Its filtering semantics are pinned by the golden regression fixtures: a change in
which annotations survive `_is_ignore` moves every reported number at once.
"""
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO

# Omni3D's evaluation defaults. Callers should not need to vary these; they are
# spelled out rather than hidden so the filtering contract is visible.
DEFAULT_FILTER_SETTINGS = {
    "category_names": [],
    "ignore_names": [],
    "truncation_thres": 0.99,
    "visibility_thres": 0.01,
    "min_height_thres": 0.0,
    "max_height_thres": 1.50,
    "modal_2D_boxes": False,
    "trunc_2D_boxes": False,
    "max_depth": 1e8,
}


def xyxy_to_xywh(box):
    x0, y0, x1, y1 = box
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def xywh_to_xyxy(box):
    x0, y0, w, h = box
    return [x0, y0, x0 + w, y0 + h]


def pick_bbox2d(anno, filter_settings):
    """Choose the 2D box, in xywh, honouring the modal/truncated preferences."""
    if filter_settings.get("modal_2D_boxes") and anno.get("bbox2D_tight", [-1])[0] != -1:
        return xyxy_to_xywh(anno["bbox2D_tight"])
    trunc = anno.get("bbox2D_trunc")
    if filter_settings.get("trunc_2D_boxes") and trunc and not all(v == -1 for v in trunc):
        return xyxy_to_xywh(trunc)
    if anno.get("bbox2D_proj", [-1])[0] != -1:
        return xyxy_to_xywh(anno["bbox2D_proj"])
    if anno.get("bbox"):
        return anno["bbox"]
    return None


def bbox3d_of(anno):
    return anno.get("bbox3D_cam") or anno.get("bbox3D")


def _is_ignore(anno, filter_settings, image_height):
    """True when an annotation is present but must not be scored for or against."""
    if anno.get("behind_camera", False) or not bool(anno.get("valid3D", True)):
        return True

    dims = anno.get("dimensions", [1, 1, 1])
    if min(dims) <= 0:
        return True
    if anno.get("center_cam", [0, 0, 0])[2] > filter_settings["max_depth"]:
        return True
    if anno.get("lidar_pts", 1) == 0 or anno.get("segmentation_pts", 1) == 0:
        return True
    if anno.get("depth_error", 0.0) > 0.5:
        return True

    bbox2d = pick_bbox2d(anno, filter_settings)
    if bbox2d is None:
        return True
    height = bbox2d[3]
    if height <= filter_settings["min_height_thres"] * image_height:
        return True
    if height >= filter_settings["max_height_thres"] * image_height:
        return True

    truncation = anno.get("truncation", -1)
    if truncation >= 0 and truncation >= filter_settings["truncation_thres"]:
        return True
    visibility = anno.get("visibility", -1)
    if visibility >= 0 and visibility <= filter_settings["visibility_thres"]:
        return True

    return anno.get("category_name") in filter_settings.get("ignore_names", [])


class Omni3DDataset(COCO):
    """COCO-compatible view of Omni3D-format annotations.

    Restricts categories to `filter_settings["category_names"]` when given, marks
    ignored annotations, and materialises the `bbox`, `bbox3D` and `depth` fields
    the evaluator expects.
    """

    def __init__(self, annotation_files, filter_settings=None):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        if isinstance(annotation_files, str):
            annotation_files = [annotation_files]

        merged, cat_by_id = None, {}
        for path in annotation_files:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data.get("info"), list):
                data["info"] = data["info"][0]
            data["info"]["known_category_ids"] = [c["id"] for c in data["categories"]]
            if merged is None:
                merged = data
            else:
                if isinstance(merged.get("info"), dict):
                    merged["info"] = [merged["info"]]
                merged["info"].append(data["info"])
                merged["annotations"] += data["annotations"]
                merged["images"] += data["images"]
            for cat in data["categories"]:
                cat_by_id.setdefault(cat["id"], cat)

        self.dataset = merged
        ordered = [cat_by_id[i] for i in sorted(cat_by_id)]

        if filter_settings is None:
            self.dataset["categories"] = ordered
            self.createIndex()
            return

        settings = dict(DEFAULT_FILTER_SETTINGS)
        settings.update(filter_settings)
        wanted = settings.get("category_names") or []
        if wanted:
            self.dataset["categories"] = [c for c in ordered if c["name"] in wanted]
        else:
            self.dataset["categories"] = ordered
            wanted = [c["name"] for c in ordered]
            settings["category_names"] = wanted
        keep_names = set(settings.get("ignore_names", [])) | set(wanted)

        id_to_name = {c["id"]: c["name"] for c in self.dataset["categories"]}
        heights = {im["id"]: im["height"] for im in self.dataset["images"]}

        valid = []
        for anno in self.dataset["annotations"]:
            anno["category_name"] = anno.get(
                "category_name", id_to_name.get(anno.get("category_id"), "")
            )
            ignore = _is_ignore(anno, settings, heights.get(anno["image_id"], 0))

            bbox2d = pick_bbox2d(anno, settings)
            if bbox2d is None:
                continue
            bbox3d = bbox3d_of(anno)
            if bbox3d is None:
                continue

            anno["bbox"] = bbox2d
            anno["area"] = bbox2d[2] * bbox2d[3]
            anno["iscrowd"] = False
            anno["ignore"] = anno["ignore2D"] = anno["ignore3D"] = ignore
            anno["bbox3D"] = bbox3d
            center = anno.get("center_cam")
            anno["depth"] = center[2] if center else float(np.array(bbox3d)[:, 2].mean())

            if anno["category_name"] in keep_names:
                valid.append(anno)

        self.dataset["annotations"] = valid
        self.createIndex()


def load_target_categories(value, gt_categories, dataset_name=None):
    """Resolve the dataset-level vocabulary from a list, a txt file or a json map."""
    if value is None:
        return [c["name"] for c in gt_categories]

    if os.path.isfile(value):
        if value.endswith(".json"):
            with open(value) as f:
                data = json.load(f)
            if isinstance(data, dict):
                # shipped form: {"_note": ..., "datasets": {name: [...]}}
                if isinstance(data.get("datasets"), dict):
                    data = data["datasets"]
                if "categories" in data:
                    return data["categories"]
                if dataset_name and dataset_name in data:
                    return data[dataset_name]
                if len(data) == 1:
                    only = next(iter(data.values()))
                    if isinstance(only, list):
                        return only
                raise ValueError(f"no category list for dataset {dataset_name!r} in {value}")
            if isinstance(data, list):
                return data
            raise ValueError(f"unsupported target-category json: {value}")
        with open(value) as f:
            return [line.strip() for line in f if line.strip()]

    return [item.strip() for item in value.split(",") if item.strip()]


def load_gt(gt_json_path, target_categories, dataset_name=None, strict=True):
    """Load ground truth grouped by image, restricted to the target vocabulary.

    `strict` raises when a target name is absent from the GT categories. That
    mismatch silently drops a whole class from evaluation, so it defaults to loud.
    """
    with open(gt_json_path) as f:
        data = json.load(f)

    categories = data["categories"]
    dataset_name = dataset_name or os.path.basename(gt_json_path).split("_")[0]
    target_categories = load_target_categories(target_categories, categories, dataset_name)

    id_to_name = {c["id"]: c["name"] for c in categories}
    name_to_id = {c["name"]: c["id"] for c in categories}

    missing = [n for n in target_categories if n not in name_to_id]
    if missing and strict:
        raise ValueError(
            f"{len(missing)} target categories are absent from {os.path.basename(gt_json_path)} "
            f"and would be silently excluded from evaluation: {missing}. "
            "Fix the target list, or pass strict=False to reproduce legacy behaviour."
        )

    target_ids = {name_to_id[n] for n in target_categories if n in name_to_id}
    images = {im["id"]: im for im in data["images"]}

    gt_by_image = defaultdict(list)
    for anno in data["annotations"]:
        cid = anno.get("category_id")
        if cid not in target_ids or id_to_name.get(cid) == "dontcare":
            continue
        bbox3d = bbox3d_of(anno)
        if bbox3d is None:
            continue
        gt_by_image[anno["image_id"]].append({
            "category_id": cid,
            "category_name": id_to_name.get(cid, "unknown"),
            "bbox2d": pick_bbox2d(anno, DEFAULT_FILTER_SETTINGS),
            "bbox3d": bbox3d,
        })

    return data, images, gt_by_image, target_categories, id_to_name, name_to_id


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

_PRED_COUNTER = itertools.count()


def normalize_bbox3d_corner_order(bbox3d, corner_order="omni3d"):
    """Map a prediction's 8x3 corners into the Omni3D/vis4d ordering."""
    if bbox3d is None or corner_order == "omni3d":
        return bbox3d
    if corner_order == "wilddet3d":
        return np.array(bbox3d)[[4, 5, 1, 0, 6, 7, 3, 2], :].tolist()
    raise ValueError(f"unknown corner order: {corner_order}")


def read_predictions_file(pred_path):
    """Read a predictions file whole. Prefer `stream_predictions_jsonl` when large."""
    if pred_path.endswith(".pth"):
        import torch  # deferred: only .pth needs torch

        return torch.load(pred_path, map_location="cpu")
    if pred_path.endswith(".jsonl"):
        with open(pred_path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(pred_path) as f:
        return json.load(f)


def _keep_top_k(instances, score_threshold, max_per_image, corner_order):
    kept = []
    for inst in instances:
        if inst.get("score", 0.0) < score_threshold:
            continue
        if inst.get("bbox3D") is not None:
            inst["bbox3D"] = normalize_bbox3d_corner_order(inst["bbox3D"], corner_order)
        kept.append(inst)
    kept.sort(key=lambda i: i.get("score", 0.0), reverse=True)
    return kept[:max_per_image] if max_per_image > 0 else kept


def load_predictions(pred_path, score_threshold=0.0, corner_order="omni3d",
                     predictions=None, max_per_image=100):
    """Group predictions by image id.

    Accepts either per-frame records `{"image_id", "instances": [...]}` or a flat
    list of instances each carrying `image_id`.
    """
    if predictions is None:
        predictions = read_predictions_file(pred_path)
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list of frames or of instances")

    by_image = defaultdict(list)
    if predictions and "instances" in predictions[0]:
        for frame in predictions:
            by_image[frame["image_id"]] = _keep_top_k(
                frame.get("instances", []), score_threshold, max_per_image, corner_order
            )
        return by_image

    for inst in predictions:
        if inst.get("score", 0.0) < score_threshold:
            continue
        by_image[inst["image_id"]].append(inst)
    for image_id, insts in by_image.items():
        by_image[image_id] = _keep_top_k(insts, score_threshold, max_per_image, corner_order)
    return by_image


def stream_predictions_jsonl(pred_path, score_threshold=0.0, corner_order="omni3d",
                             max_per_image=100):
    """Stream a .jsonl prediction file, holding only the top-k per image.

    `torch.load` on a .pth expands roughly sevenfold in memory (a 2 GB file has
    been measured at ~16 GB resident), so JSONL plus this reader is the supported
    path for large prediction sets.
    """
    import heapq

    heaps = defaultdict(list)

    def push(image_id, inst):
        if max_per_image <= 0:
            return
        score = inst.get("score", 0.0)
        entry = (score, next(_PRED_COUNTER), inst)
        heap = heaps[image_id]
        if len(heap) < max_per_image:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "instances" in rec:
                for inst in rec.get("instances", []):
                    if inst.get("score", 0.0) >= score_threshold:
                        push(rec["image_id"], inst)
            elif "image_id" in rec and rec.get("score", 0.0) >= score_threshold:
                push(rec["image_id"], rec)

    by_image = {}
    for image_id, heap in heaps.items():
        insts = [inst for _, _, inst in sorted(heap, key=lambda e: e[0], reverse=True)]
        for inst in insts:
            if inst.get("bbox3D") is not None:
                inst["bbox3D"] = normalize_bbox3d_corner_order(inst["bbox3D"], corner_order)
        by_image[image_id] = insts
    return by_image


def remap_sequential_category_ids(by_image, target_categories, name_to_id):
    """Map contiguous 0..K-1 prediction ids onto ground-truth category ids."""
    ids = [i["category_id"] for insts in by_image.values() for i in insts
           if i.get("category_id") is not None]
    if not ids:
        return by_image
    if not all(0 <= i < len(target_categories) for i in ids):
        raise ValueError("prediction category_id values are not contiguous indices")

    lookup = {idx: name_to_id.get(name, idx) for idx, name in enumerate(target_categories)}
    for insts in by_image.values():
        for inst in insts:
            cid = inst.get("category_id")
            if cid in lookup:
                inst["category_id"] = lookup[cid]
    return by_image


def prediction_bbox2d_xywh(inst, bbox_format="xywh"):
    bbox = inst.get("bbox")
    if bbox is None:
        return None
    return bbox if bbox_format == "xywh" else xyxy_to_xywh(bbox)
