import numpy as np
import pandas as pd
import pytest

from metapilot import metafeatures
from metapilot.augment import make_tasks
from metapilot.config import Config
from metapilot.portfolio import PORTFOLIOS, build_pipeline, get_portfolio
from metapilot.sources import Dataset, make_moons_circles, make_synthetic


def _mixed_frame(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "num": rng.normal(size=n),
        "num_missing": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
        "color": rng.choice(["red", "green", "blue", None], size=n),
        "zip": [f"z{int(v)}" for v in rng.integers(0, 60, size=n)],  # high cardinality
        "flag": rng.random(n) < 0.5,
    })
    y = np.where(X["num"] + (X["color"] == "red") > 0.3, "yes", "no")
    return X, y


def test_synthetic_sources_shapes():
    rng = np.random.default_rng(0)
    for ds in make_synthetic(5, rng) + make_moons_circles(4, rng):
        assert len(ds.X) == len(ds.y) and ds.X.shape[1] >= 2
        assert ds.y.nunique() >= 2


def test_make_tasks_respects_constraints():
    X, y = _mixed_frame(400)
    cfg = Config(subsamples_per_parent=5, max_rows=100, min_class_count=6)
    tasks = make_tasks(Dataset("mixed", X, pd.Series(y), "test"), cfg, np.random.default_rng(0))
    assert len(tasks) == 5
    for t in tasks:
        assert len(t.X) <= 100
        counts = np.bincount(t.y)
        assert counts.min() >= 6 and set(np.unique(t.y)) == set(range(len(counts)))
        assert t.X.shape[1] >= 1


@pytest.mark.parametrize("spec", PORTFOLIOS["pilot"], ids=lambda s: s.name)
def test_pilot_pipelines_handle_mixed_data(spec):
    X, y = _mixed_frame()
    tasks = make_tasks(Dataset("mixed", X, pd.Series(y), "test"),
                       Config(subsamples_per_parent=1, col_frac=(1.0, 1.0), row_frac=(1.0, 1.0)),
                       np.random.default_rng(0))
    t = tasks[0]
    pipe = build_pipeline(spec, t.X.iloc[:100], seed=0)
    pipe.fit(t.X.iloc[:100], t.y[:100])
    assert pipe.predict(t.X.iloc[100:]).shape == (len(t.X) - 100,)


def test_portfolio_sizes():
    assert len(get_portfolio("pilot")) == 18
    assert 36 <= len(get_portfolio("full")) <= 42


def test_metafeatures_complete_and_finite():
    X, y = _mixed_frame()
    mf = metafeatures.extract(X, pd.factorize(y)[0], landmarker_rows=100)
    assert list(mf) == metafeatures.FEATURES and len(mf) == 14
    assert all(np.isfinite(v) for v in mf.values())
    assert 0 <= mf["minority_majority_ratio"] <= 1 and 0 < mf["pca_frac_95"] <= 1
