#!/usr/bin/env python3
"""Generate the tiny synthetic dataset the smoke test runs against.

Synthetic on purpose. Subsampling a real dataset would embed CC-BY-NC annotations
in an Apache-licensed repository, and would make the test depend on data the user
has to download. Everything here is generated from a fixed seed, so the fixture is
reproducible and the expected metric values are stable.

Regenerate with:  python tests/fixtures/make_fixture.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ov3d_bench.omni3d.geometry import get_cuboid_verts_faces  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = ["chair", "table", "sofa", "bed", "lamp"]
N_IMAGES, W, H = 24, 640, 480
K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]


def corners_and_box2d(center, dims, rng):
    """8x3 camera-frame corners plus their projected 2D extent.

    `get_cuboid_verts_faces` returns corners in the camera frame. Its sibling
    `get_cuboid_verts` applies K and returns projected coordinates, which are not
    what `bbox3D_cam` holds.
    """
    verts, _ = get_cuboid_verts_faces([*center, *dims], R=np.eye(3))
    verts = np.array(verts, dtype=float).reshape(-1, 3)
    projected = np.stack([(np.array(K) @ v)[:2] / max(v[2], 1e-4) for v in verts])
    x0, y0 = projected[:, 0].min(), projected[:, 1].min()
    x1, y1 = projected[:, 0].max(), projected[:, 1].max()
    return verts.tolist(), [float(x0), float(y0), float(x1), float(y1)]


def main():
    rng = np.random.default_rng(0)
    images, annotations, predictions = [], [], []
    ann_id = 0

    for image_id in range(N_IMAGES):
        images.append({
            "id": image_id, "dataset_id": 0, "width": W, "height": H, "K": K,
            "file_path": f"synthetic/{image_id:04d}.jpg",
            "src_90_rotate": 0, "src_flagged": False, "incomplete": False,
        })
        instances = []
        for _ in range(int(rng.integers(2, 5))):
            cat = int(rng.integers(0, len(CATEGORIES)))
            center = [float(rng.uniform(-2, 2)), float(rng.uniform(-1, 1)),
                      float(rng.uniform(4, 12))]
            dims = [float(rng.uniform(0.5, 1.6)) for _ in range(3)]
            verts, box_xyxy = corners_and_box2d(center, dims, rng)

            annotations.append({
                "id": ann_id, "image_id": image_id, "dataset_id": 0,
                "category_id": cat, "category_name": CATEGORIES[cat],
                "bbox3D_cam": verts, "center_cam": center, "dimensions": dims,
                "R_cam": np.eye(3).tolist(), "bbox2D_proj": box_xyxy,
                "bbox2D_tight": box_xyxy, "bbox2D_trunc": box_xyxy,
                "behind_camera": False, "valid3D": True, "depth_error": 0.0,
                "truncation": 0.0, "visibility": 1.0,
                "lidar_pts": 100, "segmentation_pts": 100,
            })
            ann_id += 1

            # 80% detected. Of those, 70% labelled correctly, the rest confused
            # with a neighbouring class, so the confusion matrix is non-trivial.
            if rng.random() < 0.8:
                jitter = rng.normal(0, 0.05, 3)
                pred_center = [c + j for c, j in zip(center, jitter)]
                pred_verts, pred_box = corners_and_box2d(pred_center, dims, rng)
                label = cat if rng.random() < 0.7 else int((cat + 1) % len(CATEGORIES))
                instances.append({
                    "category_id": label, "score": float(rng.uniform(0.3, 0.99)),
                    "bbox": [pred_box[0], pred_box[1],
                             pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]],
                    "bbox3D": pred_verts,
                })
        # a false positive on some images, so precision is not perfect
        if rng.random() < 0.4:
            center = [float(rng.uniform(-2, 2)), float(rng.uniform(-1, 1)),
                      float(rng.uniform(4, 12))]
            dims = [float(rng.uniform(0.5, 1.6)) for _ in range(3)]
            verts, box = corners_and_box2d(center, dims, rng)
            instances.append({
                "category_id": int(rng.integers(0, len(CATEGORIES))),
                "score": float(rng.uniform(0.1, 0.5)),
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "bbox3D": verts,
            })
        predictions.append({"image_id": image_id, "instances": instances})

    gt = {
        "info": {"source": "Synthetic", "split": "test", "name": "Synthetic-test",
                 "id": 0, "version": "1.0"},
        "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CATEGORIES)],
    }
    with open(os.path.join(HERE, "mini_gt.json"), "w") as f:
        json.dump(gt, f, indent=1)
    with open(os.path.join(HERE, "mini_pred.json"), "w") as f:
        json.dump(predictions, f, indent=1)
    print(f"  {len(images)} images, {len(annotations)} GT boxes, "
          f"{sum(len(p['instances']) for p in predictions)} predictions")


if __name__ == "__main__":
    main()
