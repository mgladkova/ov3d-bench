# 🧭 Usage

Every command, what it answers, and the Python equivalent. Paths below assume
`GT=<Dataset>_test.json` and a prediction file in the format the README documents.

The dataset-level vocabulary defaults to the lists shipped with the benchmark, so
`--target-cats` is only needed to override it. Pass `--dataset-name` when the
ground-truth filename does not begin with the dataset name.

---

## 📊 `eval` — the main pass

mAP3D, recall and the confusion matrix from one matching pass.

```bash
ov3d-bench eval --gt-json $GT --pred preds.pth --dataset-name ARKitScenes \
                --class-agnostic --iou3d-min 0.15 --outdir out/
```

**Is my detector's problem geometry or naming?** Run it twice. `--class-agnostic
--iou3d-min 0.15` gives 3D class-agnostic recall, which credits a box that is well
placed no matter what label it carries. Compare that against mAP3D from the same
run: a large gap means localization is fine and the vocabulary is the bottleneck.

Useful flags:

| Flag | Effect |
|---|---|
| `--class-agnostic` | match across all categories at once, isolating localization |
| `--matching-mode greedy` | nuScenes-style confidence-ordered assignment instead of IoU-optimal |
| `--target-aware` | apply the oracle that keeps only classes present in each image |
| `--remap siglip` | remap detections with a contrastive encoder (needs `--image-root`, `[remap]`) |
| `--visualize` | write ground-truth-vs-prediction overlays to `<outdir>/visualizations` |
| `--stream` | stream a `.jsonl` prediction file instead of loading it whole |

```python
from ov3d_bench.eval import run_eval
r = run_eval(gt_json=GT, pred="preds.pth", dataset_name="ARKitScenes",
             class_agnostic=True, iou3d_min=0.15)
r["ap3d"]["AP"], r["box_iou3d"]["recall"]
```

---

## 🎯 `per-category` — what the target-aware oracle buys

Per-class AP under dataset-level and target-aware prompting, sorted by the gain.

```bash
ov3d-bench per-category --gt-json $GT --pred preds.pth --dataset-name KITTI
```

The oracle deletes predictions whose class is absent from an image rather than
penalizing them, so the gain concentrates on classes the detector confuses with a
dominant neighbour. On KITTI, `car` barely moves while `truck` gains ten points.

```python
from ov3d_bench.eval import run_per_category
run_per_category(gt_json=GT, pred="preds.pth", dataset_name="KITTI")
```

---

## 🎲 `vocab-subset` — how much AP depends on which classes are scored

Re-scores fixed predictions over subsets of the vocabulary, which is the
evaluation-side analogue of choosing a `novel` split.

```bash
ov3d-bench vocab-subset --gt-json $GT --pred preds.pth --dataset-name SUNRGBD \
                        --subset-sizes 5,10,20,40 --n-samples 5000
```

Reports the mean and spread over random subsets alongside the adverse and
favourable bounds, so you can see how far a reported number can be moved by
category selection alone. The re-scoring is exact, not sampled inference:
per-category AP does not depend on which other classes are present.

```python
from ov3d_bench.vocab_subset import run_vocab_subset
run_vocab_subset(gt_json=GT, pred="preds.pth", dataset_name="SUNRGBD")
```

---

## 🧩 `supercat` — confusion across different vocabularies

Takes **`results.json` files from `eval`**, not predictions, and projects their
confusion matrices onto a coarse shared ontology.

```bash
ov3d-bench supercat --results out/*/results.json --out-json supercat.json
```

Fine-grained matrices cannot be compared across datasets whose vocabularies differ
by two orders of magnitude. This answers the question that survives: when the
detector is wrong, is it wrong within an object family or across families? Override
the ontology with `--ontology`; a novel vocabulary lands mostly in `other` until
rules are added for it.

```python
from ov3d_bench.supercat import run_supercat, format_matrix
print(format_matrix(run_supercat(["out/a/results.json", "out/b/results.json"])))
```

---

## 💬 `templates` — prompt robustness

**Pick the mode by how your detector consumes a prompt.**

`predset` for prompt-conditioned detectors such as GroundingDINO or SAM 3 style
models. They re-detect for every prompt, so one prediction set cannot be re-scored
and you must supply one file per template:

```bash
ov3d-bench templates --gt-json $GT --dataset-name KITTI \
                     --preds tmpl_00.pth tmpl_01.pth ... tmpl_14.pth
```

`contrastive` for encoders that remap fixed detections, such as SigLIPv2 or CLIP.
The boxes never depend on the wording, so each crop is encoded once and reused
across all templates:

```bash
ov3d-bench templates --mode contrastive --backend siglip \
                     --gt-json $GT --dataset-name KITTI --preds preds.pth \
                     --image-root /path/to/images
```

Both report mAP3D per template group and the coefficient of variation across all
templates. Lower CV is more robust.

```python
from ov3d_bench.templates import run_templates_predset, run_templates_contrastive
```

---

## 🔁 Remapping a closed-vocabulary detector

Turns a frozen closed-vocabulary detector into an open-vocabulary one without
training: boxes are kept as they are, each is cropped and encoded, and reassigned
to the nearest category in the target vocabulary.

```bash
ov3d-bench eval --gt-json $GT --pred cubercnn_preds.pth --dataset-name ScanNet \
                --remap siglip --image-root /path/to/images --class-agnostic
```

Because localization is untouched, comparing before and after isolates semantics
from geometry. `--remap-legacy-compat` reproduces the original research
implementation's behaviour.

```python
from ov3d_bench.remap import remap_predictions, build_class_prototypes
```

---

## 🧪 Checking a rebuilt dataset

```bash
python tools/verify_datasets.py --anno-dir datasets/Omni3D/ --suffix _val.json
```

Image ids are not portable between differently-ordered rebuilds, and a mismatch
produces meaningless numbers rather than an error. See [DATA.md](DATA.md).
