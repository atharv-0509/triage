import numpy as np
import pandas as pd
import pytest

from metapilot import metrics


def _toy():
    perf = pd.DataFrame({"a": [0.90, 0.60], "b": [0.80, 0.70], "c": [0.70, 0.65]}, index=["t1", "t2"])
    scores = pd.DataFrame({"a": [0.0, 3.0], "b": [2.0, 2.0], "c": [1.0, 1.0]}, index=["t1", "t2"])
    return perf, scores


def test_regret_at_k():
    perf, scores = _toy()
    # t1: recommended order b, c, a -> best 0.90; t2: order a, b, c -> best 0.70
    assert metrics.regret_at_k(scores, perf, 1).tolist() == pytest.approx([10.0, 10.0])
    assert metrics.regret_at_k(scores, perf, 2).tolist() == pytest.approx([10.0, 0.0])
    assert metrics.regret_at_k(scores, perf, 3).tolist() == pytest.approx([0.0, 0.0])


def test_regret_curve_is_non_increasing():
    perf, scores = _toy()
    curve = metrics.regret_curve(scores, perf).mean()
    assert np.all(np.diff(curve.to_numpy()) <= 1e-12)
    assert curve.iloc[-1] == pytest.approx(0.0)


def test_perfect_selector():
    perf, _ = _toy()
    s = metrics.summarise(perf, perf, n_boot=50)
    assert s["regret@1"] == 0 and s["top1_hit"] == 100 and s["spearman"] == pytest.approx(1.0)


def test_hit_tolerance_boundary():
    perf = pd.DataFrame({"a": [0.800], "b": [0.795]}, index=["t"])
    scores = pd.DataFrame({"a": [0.0], "b": [1.0]}, index=["t"])
    assert metrics.summarise(scores, perf, n_boot=10)["top1_hit"] == 100


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).normal(3, 1, size=200)
    lo, hi = metrics.bootstrap_ci(v, 500)
    assert lo < v.mean() < hi
