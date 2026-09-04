#!/usr/bin/env python3
"""Compare your run against the reference results shipped with the benchmark.

Confirms your setup reproduces the published numbers. A mismatch usually means a
different prediction file, a different protocol flag, or a PyTorch3D build whose
3D IoU differs.

    python tools/compare_results.py --results out/results.json --key detany3d_KITTI_hungarian
    python tools/compare_results.py --list
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "..", "results", "reference_results.json")

FIELDS = [("mAP3D", "ap3d", "AP"), ("AP15", "ap3d", "AP15"), ("AP25", "ap3d", "AP25"),
          ("AP50", "ap3d", "AP50"), ("recall", "box_iou3d", "recall"),
          ("avg_tp_iou3d", "box_iou3d", "avg_iou3d"),
          ("matches", "box_iou3d", "matches"), ("total_gts", "box_iou3d", "total_gts")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="results.json written by `ov3d-bench eval --outdir`")
    ap.add_argument("--key", help="entry in the reference, e.g. detany3d_KITTI_hungarian")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="absolute tolerance on mAP-like values (default 0.01)")
    ap.add_argument("--list", action="store_true", help="show available keys and exit")
    args = ap.parse_args()

    with open(args.reference) as f:
        reference = json.load(f)

    if args.list:
        for section in ("eval", "vocab_subset", "templates"):
            keys = sorted(reference.get(section, {}))
            print(f"\n{section}  ({len(keys)})")
            for k in keys:
                print(f"  {k}")
        return 0

    if not args.results or not args.key:
        ap.error("--results and --key are required (or use --list)")
    if args.key not in reference["eval"]:
        print(f"no such key: {args.key}. Use --list.", file=sys.stderr)
        return 2

    expected = reference["eval"][args.key]
    with open(args.results) as f:
        got = json.load(f)

    print(f"\n  {args.key}")
    print(f"  {'field':16} {'reference':>12} {'yours':>12} {'delta':>10}")
    worst = 0.0
    for label, section, field in FIELDS:
        if label not in expected or section not in got:
            continue
        ref, mine = expected[label], got[section][field]
        delta = mine - ref
        worst = max(worst, abs(delta) if isinstance(ref, float) else 0.0)
        flag = "" if (isinstance(ref, int) and ref == mine) or abs(delta) <= args.tolerance else "  <-- differs"
        print(f"  {label:16} {ref:>12.4f} {mine:>12.4f} {delta:>+10.4f}{flag}")

    if worst <= args.tolerance:
        print(f"\n  matches the reference within {args.tolerance}")
        return 0
    print(f"\n  largest difference {worst:.4f} exceeds tolerance {args.tolerance}.\n"
          "  Check that you used the same predictions and the same protocol flags\n"
          "  (see _protocol in the reference file), and that PyTorch3D is a build\n"
          "  matched to your torch version.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
