import numpy as np
import pandas as pd

from metapilot.metalevel import (MetaData, family_win_share, fill_missing, gap_to_best, long_format,
                                 make_folds, relevance_labels)


def test_fill_missing_uses_task_minimum():
    perf = pd.DataFrame({"a": [0.9, np.nan], "b": [np.nan, 0.5], "c": [0.7, 0.6]})
    filled = fill_missing(perf)
    assert filled.loc[0, "b"] == 0.7 and filled.loc[1, "a"] == 0.5


def test_targets():
    perf = pd.DataFrame({"a": [0.9, 0.5], "b": [0.8, 0.5], "c": [0.7, 0.5]})
    assert gap_to_best(perf).max(axis=1).tolist() == [0, 0]
    rel = relevance_labels(perf)
    assert rel.loc[0].tolist() == [10, 5, 0]
    assert rel.loc[1].tolist() == [10, 10, 10]  # all tied -> all best


def test_long_format():
    mf = pd.DataFrame({"f1": [1.0, 2.0]}, index=["t1", "t2"])
    X = long_format(mf, ["p", "q", "r"])
    assert X.shape == (6, 4)
    assert X.loc[("t2", "q"), "f1"] == 2.0 and X.loc[("t2", "q"), "pipe=q"] == 1.0
    assert X[["pipe=p", "pipe=q", "pipe=r"]].sum(axis=1).eq(1).all()


def _meta(n_parents=10, per_parent=3):
    ids = [f"p{i}__s{j}" for i in range(n_parents) for j in range(per_parent)]
    rng = np.random.default_rng(0)
    tasks = pd.DataFrame({"parent": [i.split("__")[0] for i in ids]}, index=ids)
    perf = pd.DataFrame(rng.random((len(ids), 3)), index=ids, columns=["a", "b", "c"])
    mf = pd.DataFrame(rng.random((len(ids), 2)), index=ids, columns=["f1", "f2"])
    return MetaData(tasks, mf, perf, perf, {"a": "x", "b": "x", "c": "y"})


def test_folds_grouped_by_parent_and_reused(tmp_path):
    meta = _meta()
    path = tmp_path / "folds.csv"
    folds = make_folds(meta, 5, seed=0, path=path)
    assert (meta.groups.groupby(folds).nunique().sum()) == meta.groups.nunique()  # no parent in two folds
    assert set(folds) == set(range(5))
    again = make_folds(meta, 5, seed=123, path=path)  # saved assignment wins
    assert again.equals(folds)


def test_family_win_share_splits_ties():
    meta = _meta(n_parents=1, per_parent=2)
    meta.perf.iloc[0] = [0.9, 0.9, 0.5]   # family x wins (two x pipelines tie)
    meta.perf.iloc[1] = [0.9, 0.5, 0.9]   # x and y tie
    share = family_win_share(meta)
    assert share["x"] == 0.75 and share["y"] == 0.25
