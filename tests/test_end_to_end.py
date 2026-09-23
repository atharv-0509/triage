import pandas as pd

from metapilot import experiment, metadataset, plots
from metapilot.config import Config


def test_tiny_pipeline(tmp_path):
    cfg = Config(sources=["synthetic", "moons_circles"], n_synthetic=4, n_moons_circles=2,
                 subsamples_per_parent=2, max_rows=150, n_jobs=1, outer_folds=3, bootstrap=100)
    meta_dir = metadataset.build(cfg, tmp_path / "meta")
    perf = pd.read_csv(meta_dir / "performance.csv", index_col="task_id")
    assert perf.shape == (12, 18)
    assert (meta_dir / "text_meta.jsonl").read_text().count("\n") == 12

    summary = experiment.run(meta_dir, tmp_path / "run", cfg)
    assert set(summary.index) == {"Random", "Single best", "kNN meta-learner",
                                  "LightGBM regressor", "LightGBM LambdaRank"}
    assert (summary["regret@1"] >= summary["regret@3"]).all()
    assert (meta_dir / "folds.csv").exists()

    ablated = experiment.run(meta_dir, tmp_path / "run_nolm", cfg, drop_groups=("landmarkers",))
    assert len(ablated) == 5

    for path in plots.make_all(tmp_path / "run"):
        assert path.stat().st_size > 0
