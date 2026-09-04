"""Command line entry point: `ov3d-bench <command>`."""
import argparse
import json
import os
import sys

from .eval import run_eval, run_per_category
from .resources import resource_path


def _add_eval_args(p):
    p.add_argument("--gt-json", required=True, help="Omni3D-format ground-truth json")
    p.add_argument("--pred", required=True, help="predictions (.json, .jsonl or .pth)")
    p.add_argument("--target-cats", default=None,
                   help="dataset-level vocabulary: comma list, .txt, or dataset->list .json. "
                        "Defaults to the vocabularies shipped with the benchmark")
    p.add_argument("--dataset-name", default=None,
                   help="key into the target-category json (inferred from the GT filename)")
    p.add_argument("--outdir", default=None, help="write results.json here")

    p.add_argument("--pred-bbox-format", choices=["xywh", "xyxy"], default="xywh")
    p.add_argument("--pred-corner-order", choices=["omni3d", "wilddet3d"], default="omni3d",
                   help="corner ordering of the predicted 8x3 bbox3D")
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--max-dets-per-image", type=int, default=100)
    p.add_argument("--remap-from-sequential-ids", action="store_true",
                   help="predictions use contiguous 0..K-1 ids rather than GT category ids")
    p.add_argument("--stream", action="store_true",
                   help="stream a .jsonl prediction file instead of loading it whole")

    p.add_argument("--iou3d-min", type=float, default=0.0)
    p.add_argument("--class-agnostic", action="store_true",
                   help="match across all categories at once, isolating localization "
                        "from semantics (3D Class-Agnostic Recall)")
    p.add_argument("--matching-mode", choices=["hungarian", "greedy"], default="hungarian",
                   help="hungarian: IoU-optimal and confidence-blind. greedy: nuScenes-style")
    p.add_argument("--target-aware", action="store_true",
                   help="target-aware oracle: keep only predictions whose class occurs in "
                        "that image's GT. Measures how much the oracle inflates AP")
    p.add_argument("--remap", choices=["siglip", "clip"], default=None,
                   help="remap detections to the target vocabulary with a contrastive "
                        "encoder, turning a closed-vocabulary detector open-vocabulary. "
                        "Needs --image-root and the [remap] extra")
    p.add_argument("--remap-model", default=None, help="override the encoder checkpoint")
    p.add_argument("--remap-legacy-compat", action="store_true",
                   help="reproduce the research code's original remapping behaviour")
    p.add_argument("--visualize", action="store_true",
                   help="write GT-vs-prediction 3D box overlays into <outdir>/visualizations")
    p.add_argument("--image-root", default=None, help="root the GT file_path entries resolve against")
    p.add_argument("--vis-every-n", type=int, default=1)
    p.add_argument("--vis-max-images", type=int, default=0, help="0 = no limit")
    p.add_argument("--vis-nms-iou", type=float, default=0.5)
    p.add_argument("--no-strict-categories", action="store_true",
                   help="silently drop target names absent from the GT (legacy behaviour) "
                        "instead of raising")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ov3d-bench",
        description="Diagnostic benchmark for open-vocabulary monocular 3D detection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser(
        "eval", help="mAP3D, class-agnostic recall and the confusion matrix in one pass"
    )
    _add_eval_args(p_eval)

    p_pc = sub.add_parser("per-category",
                          help="per-category AP under dataset-level vs target-aware prompting")
    for a in ("--gt-json", "--pred"):
        p_pc.add_argument(a, required=True)
    p_pc.add_argument("--target-cats", default=None)
    p_pc.add_argument("--dataset-name", default=None)
    p_pc.add_argument("--out-json", default=None)
    p_pc.add_argument("--no-strict-categories", action="store_true")

    p_vs = sub.add_parser("vocab-subset",
                          help="sensitivity of AP to which categories are scored")
    for a in ("--gt-json", "--pred"):
        p_vs.add_argument(a, required=True)
    p_vs.add_argument("--target-cats", default=None)
    p_vs.add_argument("--dataset-name", default=None)
    p_vs.add_argument("--subset-sizes", default="5,10,20,40")
    p_vs.add_argument("--n-samples", type=int, default=5000)
    p_vs.add_argument("--seed", type=int, default=0)
    p_vs.add_argument("--out-json", default=None)
    p_vs.add_argument("--no-strict-categories", action="store_true")

    p_sc = sub.add_parser("supercat",
                          help="confusion aggregated onto the coarse shared ontology")
    p_sc.add_argument("--results", nargs="+", required=True,
                      help="results.json files produced by `eval`")
    p_sc.add_argument("--ontology", default=None, help="override data/supercategories.json")
    p_sc.add_argument("--out-json", default=None)

    p_tp = sub.add_parser("templates",
                          help="prompt-template robustness (predset mode)")
    p_tp.add_argument("--gt-json", required=True)
    p_tp.add_argument("--preds", nargs="+", required=True,
                      help="predset: one prediction file per template. "
                           "contrastive: a single prediction file")
    p_tp.add_argument("--target-cats", default=None)
    p_tp.add_argument("--dataset-name", default=None)
    p_tp.add_argument("--mode", choices=["predset", "contrastive"], default="predset",
                      help="predset: one prediction file per template, required for "
                           "prompt-conditioned detectors. contrastive: one prediction set "
                           "re-scored per template, for remapping encoders")
    p_tp.add_argument("--backend", choices=["siglip", "clip"], default="siglip")
    p_tp.add_argument("--image-root", default=None, help="required for contrastive mode")
    p_tp.add_argument("--out-json", default=None)
    p_tp.add_argument("--no-strict-categories", action="store_true")

    args = parser.parse_args(argv)

    # The dataset-level vocabulary is a property of the benchmark, so the
    # shipped lists are the default rather than something to pass every time.
    if getattr(args, "target_cats", None) is None and hasattr(args, "target_cats"):
        args.target_cats = resource_path("datasets.json")

    if args.command == "eval":
        results = run_eval(
            gt_json=args.gt_json, pred=args.pred, target_cats=args.target_cats,
            dataset_name=args.dataset_name, bbox_format=args.pred_bbox_format,
            corner_order=args.pred_corner_order, score_threshold=args.score_threshold,
            max_dets_per_image=args.max_dets_per_image, iou3d_min=args.iou3d_min,
            class_agnostic=args.class_agnostic, matching_mode=args.matching_mode,
            target_aware=args.target_aware,
            remap_sequential_ids=args.remap_from_sequential_ids, stream=args.stream,
            strict_categories=not args.no_strict_categories, outdir=args.outdir,
            visualize=args.visualize, image_root=args.image_root,
            vis_every_n=args.vis_every_n, vis_max_images=args.vis_max_images,
            vis_nms_iou=args.vis_nms_iou, remap=args.remap,
            remap_model=args.remap_model, remap_legacy_compat=args.remap_legacy_compat,
        )
        b, a = results["box_iou3d"], results["ap3d"]
        label = "3D Class-Agnostic Recall" if args.class_agnostic else "Recall (per class)"
        print(f"\n  mAP3D                    {a['AP']:.2f}")
        print(f"  {label:24} {b['recall']:.4f}   ({b['matches']}/{b['total_gts']} GT boxes)")
        print(f"  Mean TP 3D IoU           {b['avg_iou3d']:.4f}")
        if args.outdir:
            print(f"\n  wrote {args.outdir}/results.json")
        if results.get("visualized"):
            print(f"  wrote {results['visualized']} overlays to {args.outdir}/visualizations")
        return 0

    def dump(payload):
        if getattr(args, "out_json", None):
            os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n  wrote {args.out_json}")

    strict = not getattr(args, "no_strict_categories", False)

    if args.command == "per-category":
        r = run_per_category(gt_json=args.gt_json, pred=args.pred,
                             target_cats=args.target_cats, dataset_name=args.dataset_name,
                             strict_categories=strict)
        a = r["aggregate"]
        print(f"\n  aggregate mAP3D   dataset-level {a['dataset_level']:.2f}   "
              f"target-aware {a['target_aware']:.2f}   ({a['inflation']:.2f}x)")
        print(f"  oracle dropped {r['target_aware_filter']['dropped']} predictions\n")
        print(f"  {'category':22} {'dataset':>9} {'target-aw':>10} {'delta':>8}")
        for row in r["per_category"][:15]:
            print(f"  {str(row['category']):22} {row['dataset_level']:9.2f} "
                  f"{row['target_aware']:10.2f} {row['delta']:+8.2f}")
        dump(r)
        return 0

    if args.command == "vocab-subset":
        from .vocab_subset import run_vocab_subset
        sizes = tuple(int(x) for x in args.subset_sizes.split(","))
        r = run_vocab_subset(gt_json=args.gt_json, pred=args.pred,
                             target_cats=args.target_cats, dataset_name=args.dataset_name,
                             subset_sizes=sizes, n_samples=args.n_samples, seed=args.seed,
                             strict_categories=strict)
        print(f"\n  full-vocabulary mAP3D {r['aggregate']:.2f} over {r['n_valid']} categories with GT\n")
        print(f"  {'k':>4} {'random mean±std':>18} {'adverse':>9} {'favourable':>11} {'fav/adv':>9}")
        for row in r["rows"]:
            print(f"  {row['k']:>4} {row['rand_mean']:>10.2f}±{row['rand_std']:<7.2f} "
                  f"{row['worst']:9.2f} {row['best']:11.2f} {row['ratio']:>9.1f}x")
        dump(r)
        return 0

    if args.command == "supercat":
        from .supercat import run_supercat, format_matrix
        r = run_supercat(args.results, args.ontology)
        print(f"\n  aggregated over {r['n_sources']} results files\n")
        print(format_matrix(r))
        dump(r)
        return 0

    if args.command == "templates":
        if args.mode == "contrastive":
            from .templates import run_templates_contrastive
            r = run_templates_contrastive(
                gt_json=args.gt_json, pred=args.preds[0], image_root=args.image_root,
                target_cats=args.target_cats, dataset_name=args.dataset_name,
                backend=args.backend, strict_categories=strict)
            print(f"\n  {args.backend} over {r['n_crops']} crops   CV {r['cv_percent']:.0f}%\n")
            for g, v in r["per_group"].items():
                print(f"    {g:16} {v['mean']:6.2f} +/- {v['std']:.2f}  (n={v['n']})")
            dump(r)
            return 0
        from .templates import run_templates_predset
        r = run_templates_predset(gt_json=args.gt_json, preds=args.preds,
                                  target_cats=args.target_cats,
                                  dataset_name=args.dataset_name, strict_categories=strict)
        print(f"\n  {r['num_runs']} templates   mAP3D {r['metrics_mean']['AP']:.2f} "
              f"± {r['metrics_std']['AP']:.2f}   CV {r['cv_percent']:.0f}%\n")
        for run in r["per_run"]:
            print(f"    {run['label'][:48]:48} {run['metrics']['AP']:6.2f}")
        dump(r)
        return 0

    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
