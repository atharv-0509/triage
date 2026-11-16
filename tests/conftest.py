import pytest

from metapilot import metadataset
from metapilot.config import Config


@pytest.fixture(scope="session")
def tiny_cfg():
    return Config(sources=["synthetic", "moons_circles", "sklearn"], sklearn_datasets=["iris", "wine"],
                  n_synthetic=3, n_moons_circles=1, subsamples_per_parent=2, max_rows=120, n_jobs=1,
                  outer_folds=3, bootstrap=50)


@pytest.fixture(scope="session")
def tiny_meta(tmp_path_factory, tiny_cfg):
    """12 tasks from 6 parents (iris and wine are the real ones) with the pilot portfolio."""
    return metadataset.build(tiny_cfg, tmp_path_factory.mktemp("tiny") / "meta")
