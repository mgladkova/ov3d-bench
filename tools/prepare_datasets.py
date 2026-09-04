#!/usr/bin/env python3
"""Convert official Omni3D jsons into OV3D-Bench's shared 273-class space.

Get the inputs with Omni3D's own script:

    wget https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip && unzip Omni3D_json.zip

Each source dataset ships its own category ids (KITTI has 8, SUNRGBD 83, ...).
The benchmark scores every dataset in one shared vocabulary, so each annotation is
folded onto a canonical class and re-indexed: `category_id == classes.index(name)`.

Split semantics follow the original pipeline: Omni3D's train and val are merged
into the output train file, and Omni3D's test becomes the output val file, which
is what the benchmark uses and what `<Dataset>_test.json` is renamed from.

IDS ARE NOT PORTABLE. Image and annotation ids come from counters running across
every dataset in DATASET_ORDER, so omitting one, or reordering, shifts every id
that follows. Objectron is in that order even though the benchmark never scores it.
Predictions are keyed by image_id, so a divergent rebuild silently mismatches every
prediction. Always run `tools/verify_datasets.py` afterwards.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOCAB = os.path.join(HERE, "..", "data", "class_mapping.json")

# Load-bearing: fixes the id space. Objectron is scored by nothing but must be
# processed, because dropping it shifts the ids of every dataset after it.
DATASET_ORDER = ["nuScenes", "ARKitScenes", "Objectron", "KITTI", "Hypersim", "SUNRGBD"]

# Directory name used inside Omni3D's `file_path` values, per dataset.
SOURCE_DIRNAME = {
    "ARKitScenes": "ARKitScenes",
    "Hypersim": "hypersim",
    "KITTI": "KITTI_object",
    "nuScenes": "nuScenes",
    "Objectron": "objectron",
    "SUNRGBD": "SUNRGBD",
}


def load_vocabulary(path=None):
    with open(path or DEFAULT_VOCAB) as f:
        data = json.load(f)
    return data["classes"], data["class_mapping"]


def new_stats():
    return {"n_datasets": 0, "n_ims": 0, "n_anns": 0}


def _index_by_image(payload):
    annos = {}
    for anno in payload["annotations"]:
        annos.setdefault(anno["image_id"], []).append(anno)
    return {im["id"]: im for im in payload["images"]}, annos


def _empty(dataset, split, dataset_id, classes):
    return {
        "info": {"source": dataset, "split": split, "name": f"{dataset}-{split}",
                 "id": dataset_id, "version": "1.0"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)],
    }


def convert_dataset(dataset, json_dir, stats, classes, class_mapping,
                    image_src=None, image_dst=None):
    """Convert one Omni3D dataset. Returns (train_payload, val_payload)."""
    class_index = {name: i for i, name in enumerate(classes)}
    dataset_id = stats["n_datasets"]

    train_out = _empty(dataset, "train", dataset_id, classes)
    val_out = _empty(dataset, "val", dataset_id, classes)

    # Omni3D train + val -> our train;  Omni3D test -> our val.
    plan = [("train", train_out), ("val", train_out), ("test", val_out)]
    src_dir = SOURCE_DIRNAME.get(dataset, dataset)

    for split, target in plan:
        path = os.path.join(json_dir, f"{dataset}_{split}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing Omni3D input: {path}")
        with open(path) as f:
            payload = json.load(f)

        images, annos_by_image = _index_by_image(payload)
        for image_id in images:
            image = dict(images[image_id])
            source_path = image["file_path"]
            image["file_path"] = source_path.replace(src_dir, dataset, 1)
            image["dataset_id"] = dataset_id
            image["id"] = stats["n_ims"]

            if image_src and image_dst:
                destination = os.path.join(image_dst, image["file_path"].lstrip("/"))
                if not os.path.isfile(destination):
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    origin = os.path.join(image_src, source_path.lstrip("/"))
                    if os.path.isfile(origin):
                        shutil.copyfile(origin, destination)

            for anno in annos_by_image.get(image_id, []):
                canonical = class_mapping.get(anno["category_name"])
                if canonical is None:
                    raise KeyError(
                        f"{dataset}: no canonical class for {anno['category_name']!r}. "
                        "Extend class_mapping in data/class_mapping.json."
                    )
                anno = dict(anno)
                anno["category_name"] = canonical
                anno["category_id"] = class_index[canonical]
                anno["image_id"] = stats["n_ims"]
                anno["id"] = stats["n_anns"]
                anno["dataset_id"] = dataset_id
                target["annotations"].append(anno)
                stats["n_anns"] += 1

            target["images"].append(image)
            stats["n_ims"] += 1

    stats["n_datasets"] += 1
    return train_out, val_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True, help="unzipped official Omni3D jsons")
    ap.add_argument("--out-dir", required=True, help="where converted jsons are written")
    ap.add_argument("--vocab", default=None, help="override data/class_mapping.json")
    ap.add_argument("--datasets", default=",".join(DATASET_ORDER),
                    help="processing order. Changing this CHANGES ALL IDS")
    ap.add_argument("--image-src", default=None, help="copy images from here")
    ap.add_argument("--image-dst", default=None, help="copy images to here")
    args = ap.parse_args()

    classes, class_mapping = load_vocabulary(args.vocab)
    order = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if order != DATASET_ORDER:
        print("WARNING: dataset order differs from the reference. Image and annotation "
              "ids will NOT match the published benchmark.", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    stats = new_stats()
    for dataset in order:
        train_out, val_out = convert_dataset(
            dataset, args.json_dir, stats, classes, class_mapping,
            args.image_src, args.image_dst,
        )
        for split, payload in (("train", train_out), ("val", val_out)):
            target = os.path.join(args.out_dir, f"{dataset}_{split}.json")
            with open(target, "w") as f:
                json.dump(payload, f)
            print(f"  wrote {target}  ({len(payload['images'])} images, "
                  f"{len(payload['annotations'])} annotations)")

    print(f"\ntotals: {stats['n_datasets']} datasets, {stats['n_ims']} images, "
          f"{stats['n_anns']} annotations")
    print("AV2 and ScanNet-200 are not derivable from Omni3D; see docs/DATA.md.")
    print("Now run: python tools/verify_datasets.py --anno-dir <out-dir> --suffix _val.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
