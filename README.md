# OV3D-Bench

Diagnostic benchmark for open-vocabulary monocular 3D detectors.

Reports mAP3D alongside diagnostics that a single number hides: 3D class-agnostic
recall, which credits a well-localized box regardless of its predicted label;
confusion matrices over successfully localized objects; the inflation caused by
target-aware prompting; prompt-template robustness; and the sensitivity of AP to
which categories are scored.

## Install

```bash
pip install -e .            # evaluation
pip install -e '.[remap]'   # + SigLIPv2/CLIP remapping
```

## Use

```bash
ov3d-bench eval --gt-json <Dataset>_test.json --pred predictions.pth \
                --target-cats data/datasets.json --dataset-name <Dataset> \
                --class-agnostic --iou3d-min 0.15
```

Other commands: `per-category`, `vocab-subset`, `supercat`, `templates`.
Data preparation is described in [docs/DATA.md](docs/DATA.md).

## Reproducing the paper

Categories are scored over each dataset's full declared vocabulary. Some numbers
therefore differ slightly from the paper's Tables 2 and 7, which omitted a small
number of categories, and instead match Table 5. The differences are at most
0.22 mAP3D and affect no ranking.

## Licence

Apache 2.0, except `ov3d_bench/omni3d/`, which is derived from Omni3D / Cube R-CNN
and remains CC-BY-NC 4.0. See [NOTICE](NOTICE).
