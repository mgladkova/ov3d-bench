#!/usr/bin/env python3
"""Verify regenerated OV3D-Bench jsons against the reference manifest.

Global image/annotation ids come from running counters across every dataset in a
fixed processing order, so a rebuild that omits a dataset (Objectron is processed
even though the benchmark never uses it) or reorders them will silently produce
different ids. Predictions are keyed by image_id, so that divergence shows up much
later as empty or nonsensical results. Run this immediately after generating.

Usage:
    python tools/verify_datasets.py --anno-dir <dir with *_test.json>
"""
import argparse
import hashlib
import json
import os
import sys

DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "reference", "dataset_manifest.json"
)


def summarize(path):
    with open(path) as f:
        d = json.load(f)
    ims, anns, cats = d["images"], d["annotations"], d["categories"]
    iid = [i["id"] for i in ims]
    aid = [a["id"] for a in anns]
    blob = json.dumps([[c["id"], c["name"]] for c in cats], sort_keys=True)
    return {
        "dataset_id": sorted({i["dataset_id"] for i in ims}),
        "split": d["info"]["split"],
        "source": d["info"]["source"],
        "n_images": len(ims),
        "n_annotations": len(anns),
        "n_categories": len(cats),
        "image_id_min": min(iid),
        "image_id_max": max(iid),
        "ann_id_min": min(aid),
        "ann_id_max": max(aid),
        "categories_sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-dir", required=True, help="directory holding <Dataset>_test.json")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--suffix", default="_test.json")
    args = ap.parse_args()

    with open(args.manifest) as f:
        ref = json.load(f)["datasets"]

    failures, missing = [], []
    for ds, expected in sorted(ref.items()):
        path = os.path.join(args.anno_dir, ds + args.suffix)
        if not os.path.isfile(path):
            missing.append(ds)
            print(f"[MISSING] {ds}: {path}")
            continue
        got = summarize(path)
        diffs = [(k, expected[k], got[k]) for k in expected if expected[k] != got[k]]
        if diffs:
            failures.append(ds)
            print(f"[FAIL] {ds}")
            for k, e, g in diffs:
                print(f"         {k}: expected {e!r}, got {g!r}")
        else:
            print(f"[OK]   {ds}  images={got['n_images']} anns={got['n_annotations']}")

    if failures or missing:
        print(f"\nFAILED. mismatched={failures} missing={missing}")
        print("Most likely cause: the generation order or the set of processed datasets "
              "differed from the reference (Objectron must be included).")
        return 1
    print(f"\nAll {len(ref)} datasets match the reference manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
