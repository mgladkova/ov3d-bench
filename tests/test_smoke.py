"""Smoke test: every diagnostic runs and returns the values it did before.

Runs against a tiny synthetic dataset in tests/fixtures, so it needs no download,
no GPU and no licensed data, and finishes in seconds. It is a regression guard and
an install check: if PyTorch3D is missing or mismatched, this is where you find out.

The expected values below were produced by this code on this fixture. They are not
ground truth about detection, only a record of current behaviour, so a change here
means the package changed and you should know why.
"""
import os

import pytest

from ov3d_bench.omni3d import available as backend_available

# 3D IoU comes from PyTorch3D, which must be built against your exact torch
# (CPU vs CUDA included). Tests that compute a metric need it; the rest do not,
# and stay active so a broken PyTorch3D still leaves meaningful coverage.
needs_iou = pytest.mark.skipif(
    not backend_available(),
    reason="PyTorch3D unavailable or mismatched with torch; metric tests skipped",
)

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "fixtures", "mini_gt.json")
PRED = os.path.join(HERE, "fixtures", "mini_pred.json")
CATS = "chair,table,sofa,bed,lamp"
KW = dict(gt_json=GT, pred=PRED, target_cats=CATS, dataset_name="Synthetic")

TOL = 1e-4


def test_fixture_present():
    assert os.path.exists(GT) and os.path.exists(PRED)


@needs_iou
def test_eval_class_agnostic():
    from ov3d_bench.eval import run_eval

    r = run_eval(class_agnostic=True, iou3d_min=0.15, **KW)
    box, ap = r["box_iou3d"], r["ap3d"]
    assert box["total_gts"] == 67
    assert box["matches"] == 53
    assert box["recall"] == pytest.approx(0.7910447761194029, abs=TOL)
    assert box["avg_iou3d"] == pytest.approx(0.7997, abs=1e-3)
    assert ap["AP"] == pytest.approx(38.2857, abs=1e-3)


@needs_iou
def test_confusion_matrix_is_populated():
    """Off-diagonal mass must be non-zero, or the fixture stops testing semantics."""
    import numpy as np

    from ov3d_bench.eval import run_eval

    r = run_eval(class_agnostic=True, iou3d_min=0.15, **KW)
    conf = r["iou3d_confusion"]
    matrix = np.array(conf["matrix"])
    assert conf["labels_gt"] == ["chair", "table", "sofa", "bed", "lamp"]
    assert matrix.sum() == 53
    assert np.trace(matrix) / matrix.sum() == pytest.approx(0.6792, abs=1e-3)
    assert np.trace(matrix) < matrix.sum(), "no confusions: fixture is degenerate"


@needs_iou
def test_target_aware_inflates_ap():
    """The oracle deletes hallucinated classes, so AP must go up, never down."""
    from ov3d_bench.eval import run_eval

    plain = run_eval(iou3d_min=0.5, **KW)["ap3d"]["AP"]
    oracle = run_eval(iou3d_min=0.5, target_aware=True, **KW)
    assert oracle["protocol"]["dropped"] == 14
    assert oracle["ap3d"]["AP"] == pytest.approx(46.9953, abs=1e-3)
    assert oracle["ap3d"]["AP"] > plain


@needs_iou
def test_per_category():
    from ov3d_bench.eval import run_per_category

    r = run_per_category(**KW)
    assert len(r["per_category"]) == 5
    assert r["aggregate"]["dataset_level"] == pytest.approx(38.2857, abs=1e-3)
    assert r["aggregate"]["inflation"] > 1.0


@needs_iou
def test_vocab_subset():
    from ov3d_bench.vocab_subset import run_vocab_subset

    r = run_vocab_subset(subset_sizes=(2, 3), n_samples=200, **KW)
    assert r["n_valid"] == 5
    assert r["aggregate"] == pytest.approx(38.2857, abs=1e-3)
    assert len(r["rows"]) == 2
    for row in r["rows"]:
        assert row["worst"] <= row["rand_mean"] <= row["best"]


def test_supercat_ontology_orderings():
    """Substring rules must not capture names belonging to another family."""
    from ov3d_bench.supercat import load_ontology, to_super

    supers, rules, overrides = load_ontology()
    assert "barricades" in supers and "electronics" in supers
    # orderings that regressed before: these must not be captured by earlier rules
    assert to_super("message_board_trailer", rules, "other", overrides) == "barricades"
    assert to_super("mobile_pedestrian_crossing_sign", rules, "other", overrides) == "barricades"
    assert to_super("official_signaler", rules, "other", overrides) == "person"
    assert to_super("tv_stand", rules, "other", overrides) == "table"


@needs_iou
def test_supercat_aggregation():
    from ov3d_bench.eval import run_eval
    from ov3d_bench.supercat import aggregate_confusions

    r = run_eval(class_agnostic=True, iou3d_min=0.15, **KW)
    agg = aggregate_confusions([r["iou3d_confusion"]])
    assert 0.0 <= agg["cross_supercategory_rate"] <= 1.0


def test_strict_categories_rejects_unknown_name():
    """A target name absent from the ground truth must raise, never pass silently.

    No 3D IoU involved: the check happens while loading ground truth.
    """
    from ov3d_bench.eval import run_eval

    kw = dict(KW, target_cats="chair,table,not_a_real_class")
    with pytest.raises(ValueError, match="absent from"):
        run_eval(iou3d_min=0.15, **kw)


@needs_iou
def test_strict_categories_escape_hatch():
    from ov3d_bench.eval import run_eval

    kw = dict(KW, target_cats="chair,table,not_a_real_class")
    r = run_eval(iou3d_min=0.15, strict_categories=False, **kw)
    assert r["ap3d"]["AP"] == r["ap3d"]["AP"]  # not NaN


@needs_iou
def test_cli_entry_point():
    from ov3d_bench.cli import main

    assert main(["eval", "--gt-json", GT, "--pred", PRED, "--target-cats", CATS,
                 "--dataset-name", "Synthetic", "--class-agnostic",
                 "--iou3d-min", "0.15"]) == 0


def test_package_imports_without_iou_backend():
    """--help, loaders and the ontology must work even if PyTorch3D is broken."""
    import ov3d_bench.cli  # noqa: F401
    import ov3d_bench.io  # noqa: F401
    import ov3d_bench.supercat  # noqa: F401


def test_shipped_vocabularies_load():
    from ov3d_bench.io import load_target_categories
    from ov3d_bench.resources import resource_path

    path = resource_path("datasets.json")
    for dataset, expected in (("KITTI", 6), ("nuScenes", 9), ("ARKitScenes", 15)):
        assert len(load_target_categories(path, [], dataset)) == expected
