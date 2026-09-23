"""Experiment configuration.

A config is a small YAML file (see ``configs/``). Every field has a default that
matches the pilot study described in the Milestone 2 report, so a config file
only needs to list what it changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    name: str = "pilot"
    seed: int = 0

    # Base-level data (Section 2)
    sources: list[str] = field(default_factory=lambda: ["sklearn", "synthetic", "moons_circles"])
    n_synthetic: int = 20
    n_moons_circles: int = 6
    openml_suite_id: int = 99
    openml_extra: list[str] = field(default_factory=list)
    openml_cache_dir: str = "openml_cache"

    # Preprocessing and augmentation (Section 3)
    max_rows: int = 800
    subsamples_per_parent: int = 4
    row_frac: tuple[float, float] = (0.3, 1.0)
    col_frac: tuple[float, float] = (0.5, 1.0)
    min_class_count: int = 6
    high_cardinality: int = 20
    landmarker_rows: int = 200

    # Performance matrix (Section 2.2)
    portfolio: str = "pilot"
    cv_folds: int = 3
    fit_timeout: float = 60.0
    n_jobs: int = -1

    # Meta-level evaluation (Section 4.6)
    outer_folds: int = 5
    knn_k: int = 5
    bootstrap: int = 2000
    hit_tolerance: float = 0.5

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        for key in ("row_frac", "col_frac"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["row_frac"] = list(self.row_frac)
        d["col_frac"] = list(self.col_frac)
        return d
