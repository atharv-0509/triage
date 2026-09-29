import json
import shutil

import numpy as np
import pandas as pd
import pytest

from metapilot import baselines, experiment, metadataset, metafeatures
from metapilot.config import Config
from metapilot.sources import Dataset


def _cfg(**kw):
    base = dict(sources=["synthetic", "moons_circles"], n_synthetic=3, n_moons_circles=1,
                subsamples_per_parent=2, max_rows=120, n_jobs=1, outer_folds=2, bootstrap=50)
    return Config(**{**base, **kw})


def test_task_list_is_stable_across_calls():
    a = [t.task_id for t in metadataset.generate_tasks(_cfg())]
    b = [t.task_id for t in metadataset.generate_tasks(_cfg())]
    assert a == b and len(a) == 8


def test_shards_partition_tasks():
    ids = [t.task_id for t in metadataset.generate_tasks(_cfg())]
    owners = [[i for i in range(3) if metadataset.in_shard(t, (i, 3))] for t in ids]
    assert all(len(o) == 1 for o in owners)
    with pytest.raises(ValueError):
        metadataset.parse_shard("3/3")


def test_sharded_build_matches_single_build_and_resumes(tmp_path):
    cfg = _cfg()
    single = metadataset.build(cfg, tmp_path / "single")

    merged = tmp_path / "merged"
    for i in range(2):  # two "machines"
        metadataset.build(cfg, tmp_path / f"m{i}", shard=(i, 2))
        shutil.copytree(tmp_path / f"m{i}" / "parts", merged / "parts", dirs_exist_ok=True)
    shutil.copy(tmp_path / "m0" / "config.yaml", merged / "config.yaml")
    shutil.copy(tmp_path / "m0" / "portfolio.csv", merged / "portfolio.csv")
    metadataset.assemble(merged)

    a = pd.read_csv(single / "performance.csv", index_col="task_id").sort_index()
    b = pd.read_csv(merged / "performance.csv", index_col="task_id").sort_index()
    pd.testing.assert_frame_equal(a, b)

    # Resume: a second build finds every part and runs nothing.
    mtimes = {p: p.stat().st_mtime_ns for p in (single / "parts").glob("*.json")}
    metadataset.build(cfg, single)
    assert {p: p.stat().st_mtime_ns for p in (single / "parts").glob("*.json")} == mtimes

    # A part from a different config is rejected at assembly.
    part = next((merged / "parts").glob("*.json"))
    r = json.loads(part.read_text())
    r["config_key"] = "other"
    part.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="different config"):
        metadataset.assemble(merged)


def test_duplicate_parents_are_dropped():
    X = pd.DataFrame({"a": range(20), "b": range(20)})
    y = pd.Series([0, 1] * 10)
    kept = metadataset.dedupe_parents([Dataset("p", X, y, "s"), Dataset("q", X * 2, y, "s"),
                                       Dataset("r", X.iloc[:18], y.iloc[:18], "s")])
    assert [d.name for d in kept] == ["p", "r"]


def test_metafeatures_on_wide_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3000)))
    mf = metafeatures.extract(X, np.repeat([0, 1], 20), landmarker_rows=40)
    assert all(np.isfinite(v) for v in mf.values()) and mf["cols_rows_ratio"] == 75


def test_seeds_and_flaml_row(tmp_path):
    cfg = _cfg()
    meta = metadataset.build(cfg, tmp_path / "meta")
    perf = pd.read_csv(meta / "performance.csv", index_col="task_id")
    fake = pd.DataFrame({"accuracy": perf.max(axis=1) - 0.01, "seconds": 3.0, "budget_per_fold": 1.0,
                         "best_estimators": "lgbm"}, index=perf.index)
    fake.to_csv(meta / "flaml.csv")

    summary = experiment.run(meta, tmp_path / "run", cfg, extra_seeds=(1,))
    assert summary.loc["FLAML (1s/fold)", "regret@1"] == pytest.approx(1.0)
    assert (summary.drop("FLAML (1s/fold)")["cost@3_s"] >= summary.drop("FLAML (1s/fold)")["cost@1_s"]).all()
    seeds = pd.read_csv(tmp_path / "run" / "summary_seeds.csv", index_col="selector")
    assert "regret@1_std" in seeds and len(seeds) == 5
    assert (meta / "folds_seed1.csv").exists()
    assert "FLAML" in experiment.format_summary(summary)


def test_flaml_runs_on_real_tasks(tmp_path):
    pytest.importorskip("flaml")
    cfg = _cfg(cv_folds=2)
    out = baselines.run(cfg, tmp_path, budget=1, max_tasks=2)
    res = pd.read_csv(out, index_col="task_id")
    assert len(res) == 2 and res["accuracy"].between(0, 1).all()
