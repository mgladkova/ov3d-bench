# Preparing the data

OV3D-Bench ships **no annotations and no images**. Every dataset is obtained from
its original source under that source's own terms, and converted locally.

The benchmark scores seven datasets in one shared 273-class vocabulary. Five come
from Omni3D and are produced by a converter in this repo. Two, Argoverse 2 and
ScanNet-200, are not part of Omni3D and need their raw releases.

> **The evaluation jsons are not stock Omni3D.** Omni3D gives each dataset its own
> category ids; the benchmark re-indexes everything into a single vocabulary where
> `category_id == classes.index(name)`. Stock Omni3D jsons will not work.

---

## 1. The five Omni3D datasets

KITTI, nuScenes, SUNRGBD, ARKitScenes and Hypersim.

Download the official annotations, the same archive Omni3D's own script fetches:

```bash
wget https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip
unzip Omni3D_json.zip -d omni3d_json/
```

Convert them into the shared vocabulary:

```bash
python tools/prepare_datasets.py \
    --json-dir omni3d_json/ \
    --out-dir  datasets/Omni3D/
```

Each dataset yields `<Dataset>_train.json` and `<Dataset>_val.json`. **The
benchmark evaluates on the `_val.json` files**, which the published release
distributes renamed to `_test.json`; rename them if you want the published names.

Split semantics, inherited from the original pipeline: Omni3D's *train* and *val*
are merged into the output train file, and Omni3D's *test* becomes the output val
file.

Images are downloaded separately from each source dataset and passed to the
evaluator with `--image-root`. `prepare_datasets.py` can also assemble a combined
image tree with `--image-src` and `--image-dst`, but this is optional.

---

## 2. Argoverse 2 and ScanNet-200

Neither is part of Omni3D, so both are built from raw sensor data.

Download each after accepting its terms, then run the matching converter in
`tools/convert/`. Both need extra dependencies that the benchmark itself does not:

| Converter | Needs |
|---|---|
| `tools/convert/av2.py` | the [av2-api](https://github.com/argoverse/av2-api) devkit, `shapely`, `pyquaternion`, `opencv-python` |
| `tools/convert/scannet.py` | ScanNet's label-mapping TSV, `opencv-python` |

Install those into the same environment before running either.

---

## 3. Verify, always

**Image and annotation ids are not portable.** They come from counters running
across every dataset in a fixed order, so omitting a dataset or reordering shifts
every id that follows. Objectron is processed even though the benchmark never
scores it, precisely because dropping it would shift everything after it.

Predictions are keyed by `image_id`. A divergent rebuild does not fail loudly, it
silently mismatches every prediction and reports meaningless numbers. So after any
rebuild:

```bash
python tools/verify_datasets.py --anno-dir datasets/Omni3D/ --suffix _val.json
```

This compares image counts, annotation counts and id ranges against
`tests/reference/dataset_manifest.json`, the reference taken from the published
release. Anything other than a clean pass means your ids differ from everyone
else's, and your numbers are not comparable.

---

## 4. Licences

Each dataset stays under its own licence, and you accept those directly with the
provider. In particular:

| Source | Note |
|---|---|
| Omni3D / Cube R-CNN | CC-BY-NC 4.0 |
| nuScenes | non-commercial |
| KITTI | CC-BY-NC-SA |
| ScanNet | requires a signed terms-of-use agreement; annotations are not redistributable |
| Argoverse 2 | CC-BY-NC 4.0 |

Because Omni3D's annotations are CC-BY-NC, evaluations on the five in-domain
datasets are non-commercial regardless of this repository's Apache 2.0 licence.
See `NOTICE`.

If you use OV3D-Bench, cite Omni3D as well as this benchmark.
