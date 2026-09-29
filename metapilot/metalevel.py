"""Meta-level data handling (report Sections 3.4 and 3.5).

Loads a meta-dataset directory written by :mod:`metapilot.metadataset`,
fills failed evaluations, builds regression/ranking targets and creates the
grouped outer folds that every method (including the Phase 2 LLM) shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class MetaData:
    tasks: pd.DataFrame        # index task_id
    mf: pd.DataFrame           # task x meta-feature
    perf: pd.DataFrame         # task x pipeline, accuracy in [0, 1], no missing values
    perf_raw: pd.DataFrame     # as measured, NaN = failed / timed out
    family: dict[str, str]     # pipeline -> family
    secs: pd.DataFrame | None = None  # task x pipeline, fit+predict seconds summed over CV folds

    @property
    def pipelines(self) -> list[str]:
        return list(self.perf.columns)

    @property
    def groups(self) -> pd.Series:
        return self.tasks["parent"]

    @classmethod
    def load(cls, meta_dir: str | Path) -> "MetaData":
        d = Path(meta_dir)
        tasks = pd.read_csv(d / "tasks.csv", index_col="task_id")
        mf = pd.read_csv(d / "metafeatures.csv", index_col="task_id")
        perf_raw = pd.read_csv(d / "performance.csv", index_col="task_id")
        portfolio = pd.read_csv(d / "portfolio.csv")
        family = dict(zip(portfolio["pipeline"], portfolio["family"]))

        usable = perf_raw.notna().any(axis=1)
        ids = perf_raw.index[usable]
        perf = fill_missing(perf_raw.loc[ids])
        secs = None
        if (d / "fit_seconds.csv").exists():
            secs = pd.read_csv(d / "fit_seconds.csv", index_col="task_id").loc[ids, perf.columns]
        return cls(tasks.loc[ids], mf.loc[ids], perf, perf_raw.loc[ids], family, secs)


def fill_missing(perf: pd.DataFrame) -> pd.DataFrame:
    """Replace failed evaluations with the lowest score observed on that task,
    so a failing pipeline is never ranked highly (report Section 3.1)."""
    return perf.apply(lambda row: row.fillna(row.min()), axis=1)


def gap_to_best(perf: pd.DataFrame) -> pd.DataFrame:
    """Regression target: accuracy minus best accuracy on the task (<= 0)."""
    return perf.sub(perf.max(axis=1), axis=0)


def relevance_labels(perf: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """Ranking target: per-task min-max rescaling to integers 0..levels."""
    lo, hi = perf.min(axis=1), perf.max(axis=1)
    span = (hi - lo).replace(0, np.nan)
    scaled = perf.sub(lo, axis=0).div(span, axis=0).fillna(1.0)
    return (scaled * levels).round().astype(int)


def normalised_scores(perf: pd.DataFrame) -> pd.DataFrame:
    """Per-task min-max scaled accuracy in [0, 1] (used by the kNN meta-learner)."""
    lo, hi = perf.min(axis=1), perf.max(axis=1)
    span = (hi - lo).replace(0, np.nan)
    return perf.sub(lo, axis=0).div(span, axis=0).fillna(1.0)


def long_format(mf: pd.DataFrame, pipelines: list[str]) -> pd.DataFrame:
    """One row per (task, pipeline): meta-features joined with a one-hot
    pipeline identity, so a single model can score every combination."""
    n_t, n_p = len(mf), len(pipelines)
    X = pd.DataFrame(np.repeat(mf.to_numpy(), n_p, axis=0), columns=mf.columns)
    onehot = np.tile(np.eye(n_p), (n_t, 1))
    X[[f"pipe={p}" for p in pipelines]] = onehot
    X.index = pd.MultiIndex.from_product([mf.index, pipelines], names=["task_id", "pipeline"])
    return X


def folds_path(meta_dir: str | Path, seed: int, primary_seed: int) -> Path:
    """``folds.csv`` is the canonical split shared by every method; extra
    seeds used for robustness checks get their own file."""
    name = "folds.csv" if seed == primary_seed else f"folds_seed{seed}.csv"
    return Path(meta_dir) / name


def make_folds(meta: MetaData, n_splits: int, seed: int, path: str | Path | None = None) -> pd.Series:
    """Outer folds grouped by parent dataset. If ``path`` exists the saved
    assignment is reused so that every method sees identical splits."""
    if path is not None and Path(path).exists():
        saved = pd.read_csv(path, index_col="task_id")["fold"]
        missing = set(meta.perf.index) - set(saved.index)
        if missing:
            raise ValueError(f"{path} has no fold for {len(missing)} tasks; delete it to regenerate")
        return saved.loc[meta.perf.index]

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=meta.perf.index, name="fold")
    for k, (_, te) in enumerate(gkf.split(meta.mf, groups=meta.groups)):
        fold.iloc[te] = k
    if path is not None:
        pd.DataFrame({"parent": meta.groups, "fold": fold}).to_csv(path, index_label="task_id")
    return fold


def family_win_share(meta: MetaData) -> pd.Series:
    """Share of tasks on which each family has the best pipeline. Ties are
    split evenly between the tied families."""
    wins: dict[str, float] = {f: 0.0 for f in set(meta.family.values())}
    for _, row in meta.perf.iterrows():
        best = row.index[np.isclose(row.to_numpy(), row.max())]
        fams = {meta.family[p] for p in best}
        for f in fams:
            wins[f] += 1.0 / len(fams)
    return (pd.Series(wins) / len(meta.perf)).sort_values(ascending=False)
