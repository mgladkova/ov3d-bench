"""Tier-1 visualization: 3D boxes drawn in image space, ground truth beside predictions.

This is the sanity check to run first on a new detector. If boxes land on objects
here but mAP3D is low, the failure is semantic rather than geometric, which is the
distinction the rest of the benchmark quantifies.

Independent implementation (Apache 2.0). The 8-corner edge list is a property of
the Omni3D/vis4d corner ordering that `io.normalize_bbox3d_corner_order` produces.
"""
import colorsys
import os

import cv2
import numpy as np

from .io import bbox3d_of, pick_bbox2d, prediction_bbox2d_xywh, xywh_to_xyxy
from .io import DEFAULT_FILTER_SETTINGS

# Edges of a cuboid under the Omni3D/vis4d corner ordering.
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (4, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

GT_COLOR = (60, 220, 60)      # BGR, green
NEAR_PLANE = 0.05


def color_for(category_id):
    """Stable, well-separated BGR colour per category id."""
    hue = (int(category_id) * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _clip_to_near_plane(v0, v1, z=NEAR_PLANE, eps=1e-4):
    """Trim a segment to the near plane, or drop it if wholly behind the camera."""
    z0, z1 = v0[-1], v1[-1]
    if z0 < z and z1 < z:
        return None
    if z0 < z or z1 < z:
        s = (z - z0) / max(z1 - z0, eps)
        moved = v0 + s * (v1 - v0)
        return (moved, v1) if z0 < z else (v0, moved)
    return v0, v1


def draw_box3d(image, K, corners3d, color, thickness=2, max_behind=0.1):
    """Project an 8x3 cuboid and draw its edges. Skips boxes mostly behind the camera."""
    corners = np.asarray(corners3d, dtype=np.float64)
    if corners.shape != (8, 3):
        return False
    if np.mean(corners[:, 2] < NEAR_PLANE) > max_behind:
        return False

    height, width = image.shape[:2]
    drawn = False
    for i, j in BOX_EDGES:
        seg = _clip_to_near_plane(corners[i], corners[j])
        if seg is None:
            continue
        pts = []
        for v in seg:
            p = K @ v
            depth = max(float(v[2]), 1e-4)
            pts.append((p[0] / depth, p[1] / depth))
        if not all(np.isfinite(np.array(pts)).ravel()):
            continue
        p0 = (int(pts[0][0]), int(pts[0][1]))
        p1 = (int(pts[1][0]), int(pts[1][1]))
        if cv2.clipLine((0, 0, width, height), p0, p1)[0]:
            cv2.line(image, p0, p1, color, thickness, lineType=cv2.LINE_AA)
            drawn = True
    return drawn


def draw_label(image, text, corners3d, K, color):
    """Put a category label at the top-front corner of a projected box."""
    corners = np.asarray(corners3d, dtype=np.float64)
    visible = corners[corners[:, 2] > NEAR_PLANE]
    if len(visible) == 0:
        return
    projected = np.stack([(K @ v)[:2] / max(float(v[2]), 1e-4) for v in visible])
    x, y = projected[:, 0].min(), projected[:, 1].min()
    x = int(np.clip(x, 0, image.shape[1] - 1))
    y = int(np.clip(y, 12, image.shape[0] - 1))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)


def nms_2d(instances, iou_threshold=0.5, bbox_format="xywh"):
    """Suppress overlapping predictions, highest score first."""
    ordered = sorted(instances, key=lambda i: i.get("score", 0.0), reverse=True)
    boxes = []
    for inst in ordered:
        b = prediction_bbox2d_xywh(inst, bbox_format)
        boxes.append(xywh_to_xyxy(b) if b else None)

    kept = []
    for idx, box in enumerate(boxes):
        if box is None or all(_iou2d(box, boxes[k]) < iou_threshold for k in kept):
            kept.append(idx)
    return [ordered[i] for i in kept]


def _iou2d(a, b):
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _panel(image, title):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_pair(image, K, gt_annos, predictions, id_to_name, bbox_format="xywh",
                show_scores=True):
    """Ground truth on the left, predictions on the right, same image underneath."""
    left, right = image.copy(), image.copy()
    for anno in gt_annos:
        corners = bbox3d_of(anno)
        if corners is None:
            continue
        if draw_box3d(left, K, corners, GT_COLOR):
            draw_label(left, str(anno.get("category_name", "")), corners, K, GT_COLOR)
    for inst in predictions:
        corners = inst.get("bbox3D")
        if corners is None:
            continue
        cid = inst.get("category_id")
        color = color_for(cid if cid is not None else 0)
        if draw_box3d(right, K, corners, color):
            name = id_to_name.get(cid, str(cid))
            text = f"{name} {inst.get('score', 0.0):.2f}" if show_scores else str(name)
            draw_label(right, text, corners, K, color)
    return np.hstack([_panel(left, "Ground truth"), _panel(right, "Predictions")])


def resolve_image_path(image_info, image_root):
    path = image_info.get("file_path") or image_info.get("file_name") or ""
    if image_root:
        candidate = os.path.join(image_root, path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(image_root, os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
    return path if os.path.exists(path) else None


def visualize(gt_data, images, predictions_by_image, id_to_name, image_root, outdir,
              score_threshold=0.0, nms_iou=0.5, every_n=1, max_images=0,
              bbox_format="xywh"):
    """Write GT-vs-prediction overlays. Returns the number of images written."""
    os.makedirs(outdir, exist_ok=True)

    annos_by_image = {}
    for anno in gt_data.get("annotations", []):
        annos_by_image.setdefault(anno["image_id"], []).append(anno)

    written = 0
    for n, image_id in enumerate(sorted(predictions_by_image)):
        if n % max(every_n, 1):
            continue
        if max_images and written >= max_images:
            break
        info = images.get(image_id)
        if info is None:
            continue
        path = resolve_image_path(info, image_root)
        if path is None:
            continue
        image = cv2.imread(path)
        if image is None:
            continue

        K = np.array(info["K"], dtype=np.float64)
        preds = [p for p in predictions_by_image[image_id]
                 if p.get("score", 0.0) >= score_threshold]
        preds = nms_2d(preds, nms_iou, bbox_format)

        panel = render_pair(image, K, annos_by_image.get(image_id, []), preds,
                            id_to_name, bbox_format)
        cv2.imwrite(os.path.join(outdir, f"{image_id}.jpg"), panel)
        written += 1
    return written
