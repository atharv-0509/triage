"""Task augmentation (report Section 3.2).

Each parent dataset is turned into several tasks by keeping a random 30-100%
of its rows and 50-100% of its columns. Tasks where any class has fewer than
``min_class_count`` samples are rejected so stratified CV still works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .preprocess import clean_dataset, stratified_cap
from .sources import Dataset

log = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    parent: str
    source: str
    X: pd.DataFrame
    y: np.ndarray
    description: str = ""

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))


def make_tasks(parent: Dataset, cfg: Config, rng: np.random.Generator, max_tries: int = 50) -> list[Task]:
    X, y = clean_dataset(parent.X, parent.y, max_rows=None, seed=cfg.seed)
    X, y = _drop_rare_classes(X, y, cfg.min_class_count)
    if len(np.unique(y)) < 2:
        log.warning("parent %s has fewer than 2 usable classes; skipped", parent.name)
        return []

    tasks: list[Task] = []
    tries = 0
    while len(tasks) < cfg.subsamples_per_parent and tries < max_tries:
        tries += 1
        row_frac = rng.uniform(*cfg.row_frac)
        col_frac = rng.uniform(*cfg.col_frac)
        n_rows = max(2, int(round(row_frac * len(X))))
        n_cols = max(1, int(np.ceil(col_frac * X.shape[1])))
        rows = np.sort(rng.choice(len(X), size=n_rows, replace=False))
        cols = np.sort(rng.choice(X.shape[1], size=n_cols, replace=False))

        Xt = X.iloc[rows, cols].reset_index(drop=True)
        yt = y[rows]
        Xt = Xt.loc[:, Xt.nunique(dropna=True) > 1]
        if Xt.shape[1] == 0:
            continue
        Xt, yt = stratified_cap(Xt, yt, cfg.max_rows, seed=int(rng.integers(2**31 - 1)))
        _, counts = np.unique(yt, return_counts=True)
        if len(counts) < 2 or counts.min() < cfg.min_class_count:
            continue
        yt = pd.factorize(yt, sort=True)[0].astype(int)  # re-index labels to 0..K-1
        tasks.append(Task(f"{parent.name}__s{len(tasks)}", parent.name, parent.source, Xt, yt, parent.description))

    if len(tasks) < cfg.subsamples_per_parent:
        log.warning("parent %s: only %d/%d valid subsamples after %d tries",
                    parent.name, len(tasks), cfg.subsamples_per_parent, tries)
    return tasks


def _drop_rare_classes(X: pd.DataFrame, y: np.ndarray, min_count: int):
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= min_count]
    if len(keep_classes) == len(classes):
        return X, y
    mask = np.isin(y, keep_classes)
    return X.loc[mask].reset_index(drop=True), pd.factorize(y[mask], sort=True)[0].astype(int)
