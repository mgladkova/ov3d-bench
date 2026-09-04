# OV3D-Bench

Diagnostic benchmark for open-vocabulary monocular 3D object detection.

A single mAP number hides where a detector actually fails. OV3D-Bench reports
mAP3D alongside diagnostics that separate the causes: **3D class-agnostic recall**,
which credits a well-localized box regardless of its predicted label; **confusion
matrices** over objects that were localized successfully; the **inflation caused by
target-aware prompting**; **prompt-template robustness**; and the **sensitivity of
AP to which categories are scored**.

## 📦 Install

```bash
pip install -e .            # evaluation
pip install -e '.[remap]'   # + SigLIPv2/CLIP remapping
pip install -e '.[data]'    # + building datasets from raw sources
```

**PyTorch3D is required** and is not installable from PyPI. It provides the 3D IoU
used by every metric and needs a build matched to your torch and CUDA version, so
it cannot be declared in `pyproject.toml`. Either install it yourself, or use the
conda environment file, which pins the combination this benchmark was verified
against (Python 3.10, torch 2.0.1, CUDA 11.8, PyTorch3D 0.7.4):

```bash
conda env create -f environment.yml && conda activate ov3d-bench
```

Check the install:

```bash
pytest tests/test_smoke.py        # synthetic fixture, no downloads
```

If you have the published predictions, `tools/compare_results.py` also checks your
numbers against `results/reference_results.json`. See
[docs/USAGE.md](docs/USAGE.md#-verifying-your-setup).

## 🚀 Use

```bash
ov3d-bench eval --gt-json ARKitScenes_test.json --pred predictions.pth \
                --dataset-name ARKitScenes --class-agnostic --iou3d-min 0.15
```

The dataset-level vocabulary defaults to the lists shipped with the benchmark.

Other commands: `per-category`, `vocab-subset`, `supercat`, `templates`, and
remapping a closed-vocabulary detector. **[docs/USAGE.md](docs/USAGE.md) documents
every command with a worked example and its Python equivalent.**

### 📄 Prediction format

`.json`, `.jsonl` (streamable) or `.pth`, as either per-frame records
`{"image_id", "instances": [...]}` or a flat list of instances. Each instance:

| Field | Meaning |
|---|---|
| `image_id` | matches the ground-truth image id |
| `category_id` | ground-truth category id, or `0..K-1` with `--remap-from-sequential-ids` |
| `score` | confidence |
| `bbox` | 2D box, `xywh` by default or `xyxy` via `--pred-bbox-format` |
| `bbox3D` | 8x3 camera-frame corners, Omni3D ordering |

Large prediction sets should use `.jsonl` with `--stream`: `torch.load` on a `.pth`
expands roughly sevenfold in memory.

## 🗂️ Data

The evaluation jsons are built locally from each source dataset under its own
terms. See [docs/DATA.md](docs/DATA.md), and always run
`tools/verify_datasets.py` after a rebuild, because image ids are not portable
between differently-ordered rebuilds.

## 🔬 Reproducing the paper

Categories are scored over each dataset's full declared vocabulary. Some numbers
therefore differ slightly from Tables 2 and 7, which omitted a small number of
categories, and instead match Table 5. Differences are at most 0.22 mAP3D and
affect no ranking.

## 📚 Citation

```bibtex
@article{gladkova2026ov3d,
  title={OV3D-Bench: A Diagnostic Benchmark for Open-Vocabulary Monocular 3D Detection},
  author={Gladkova, Mariia and Peri, Neehar and Khatri, Ishan and Ramanan, Deva and Cremers, Daniel},
  journal={arXiv preprint arXiv:2608.17110},
  year={2026}
}
```

The mAP3D metric and the evaluation jsons derive from Omni3D, which its licence
requires you to cite as well:

```bibtex
@inproceedings{brazil2023omni3d,
  title={{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  author={Brazil, Garrick and Kumar, Abhinav and Straub, Julian and Ravi, Nikhila
          and Johnson, Justin and Gkioxari, Georgia},
  booktitle={CVPR},
  year={2023}
}
```

## ⚖️ Licence

Apache 2.0, except `ov3d_bench/omni3d/`, which derives from Omni3D / Cube R-CNN and
remains CC-BY-NC 4.0. Because Omni3D's annotations are also CC-BY-NC, evaluation on
the five in-domain datasets is non-commercial regardless. See [NOTICE](NOTICE).
