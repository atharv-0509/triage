import numpy as np
import pandas as pd
import pytest

from metapilot.recommend import Recommender


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.choice(["x", "y", None], size=n)})
    y = np.where(X["a"] + (X["c"] == "x") > 0.2, "pos", "neg")
    y[:3] = "rare"  # a class too small to keep
    return X, pd.Series(y)


def test_recommender(tiny_meta):
    rec = Recommender(tiny_meta)
    X, y = _frame()
    r = rec.recommend(X, y, verify_k=2, n_neighbours=3)
    assert sorted(r.ranking["pipeline"]) == sorted(rec.meta.pipelines)
    assert r.ranking["score"].is_monotonic_decreasing
    assert r.dropped_classes == 1 and r.shape[2] == 2
    assert len(r.neighbours) == 3 and r.neighbours["distance"].is_monotonic_increasing
    assert list(r.verified["pipeline"]) == list(r.ranking["pipeline"][:2])
    assert r.verified["cv_accuracy"].between(0, 1).all()
    with pytest.raises(ValueError, match="two classes"):
        rec.recommend(X, pd.Series(["a"] * len(X)))


def test_demo_handlers(tiny_meta, tmp_path):
    pytest.importorskip("gradio")
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("demo", pathlib.Path(__file__).parents[1] / "app" / "demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    app = demo.build(str(tiny_meta))
    fns = {f.fn.__name__: f.fn for f in app.fns.values()}
    X, y = _frame()
    csv = tmp_path / "d.csv"
    X.assign(label=y).to_csv(csv, index=False)
    update, preview = fns["on_input"](str(csv), None)
    assert update["value"] == "label" and len(preview) == 8
    summary, ranking, neighbours, verified, mf = fns["on_run"](str(csv), None, "label", False)
    assert "Top recommendation" in summary and len(mf) == 14 and verified.empty
