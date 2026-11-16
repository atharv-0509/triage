"""Recommend pipelines for a new dataset (the demo in report Table 7).

Uses a meta-dataset built by ``metapilot build``: the meta-model is trained on
all of its tasks, the new dataset's 14 meta-features are extracted, and the
portfolio is ranked. To show *why*, the most similar training tasks (in
standardised meta-feature space) are listed with the pipeline that won on
each. Optionally the top-k recommendations are run on the data with the same
cross-validation as the meta-dataset, so the user sees real accuracies.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import metafeatures
from .augment import _drop_rare_classes
from .config import Config
from .metalevel import MetaData
from .portfolio import build_pipeline, get_portfolio
from .preprocess import clean_dataset
from .selectors import KNNMetaLearner, LGBMGapRegressor, LGBMLambdaRank

SELECTORS = {"lambdarank": LGBMLambdaRank, "regressor": LGBMGapRegressor, "knn": KNNMetaLearner}


@dataclass
class Recommendation:
    ranking: pd.DataFrame        # rank, pipeline, family, score
    metafeatures: pd.Series
    neighbours: pd.DataFrame     # most similar training tasks and their winners
    verified: pd.DataFrame | None
    shape: tuple[int, int, int]  # rows, columns, classes after cleaning
    dropped_classes: int


class Recommender:
    """Fit once on a meta-dataset, then recommend for any number of datasets."""

    def __init__(self, meta_dir: str | Path, selector: str = "lambdarank"):
        if selector not in SELECTORS:
            raise ValueError(f"selector must be one of {sorted(SELECTORS)}")
        self.meta = MetaData.load(meta_dir)
        self.cfg = Config.load(Path(meta_dir) / "config.yaml")
        cls = SELECTORS[selector]
        self.model = (cls(self.cfg.knn_k) if cls is KNNMetaLearner else cls(self.cfg.seed)) \
            .fit(self.meta.mf, self.meta.perf)
        self.scaler = StandardScaler().fit(self.meta.mf)
        self.specs = {s.name: s for s in get_portfolio(self.cfg.portfolio)}

    def recommend(self, X: pd.DataFrame, y, verify_k: int = 0, n_neighbours: int = 5) -> Recommendation:
        cfg = self.cfg
        X, y = clean_dataset(X, y, max_rows=cfg.max_rows, seed=cfg.seed)
        n_before = len(np.unique(y))
        X, y = _drop_rare_classes(X, y, cfg.min_class_count)
        if len(np.unique(y)) < 2:
            raise ValueError(f"need at least two classes with {cfg.min_class_count}+ rows each")

        mf = pd.Series(metafeatures.extract(X, y, cfg.landmarker_rows, cfg.seed))
        row = pd.DataFrame([mf], index=["new"])[self.meta.mf.columns]
        scores = self.model.score(row).iloc[0]
        order = scores.sort_values(ascending=False, kind="stable")
        ranking = pd.DataFrame({
            "rank": range(1, len(order) + 1),
            "pipeline": order.index,
            "family": [self.meta.family[p] for p in order.index],
            "score": order.to_numpy(),
        })

        dist = np.sqrt(((self.scaler.transform(self.meta.mf) - self.scaler.transform(row)) ** 2).sum(axis=1))
        near = self.meta.mf.index[np.argsort(dist)[:n_neighbours]]
        perf = self.meta.perf.loc[near]
        neighbours = pd.DataFrame({
            "task": near,
            "parent": self.meta.tasks.loc[near, "parent"].to_numpy(),
            "distance": np.sort(dist)[:n_neighbours],
            "best_pipeline": perf.idxmax(axis=1).to_numpy(),
            "best_accuracy": perf.max(axis=1).to_numpy(),
        })

        verified = self.verify(X, y, list(order.index[:verify_k])) if verify_k else None
        return Recommendation(ranking, mf, neighbours, verified, (len(X), X.shape[1], len(np.unique(y))),
                              n_before - len(np.unique(y)))

    def verify(self, X: pd.DataFrame, y: np.ndarray, pipelines: list[str]) -> pd.DataFrame:
        """Cross-validated accuracy of the given pipelines on this dataset."""
        cv = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.seed)
        rows = []
        for rank, name in enumerate(pipelines, start=1):
            accs, t0 = [], time.perf_counter()
            for tr, te in cv.split(X, y):
                pipe = build_pipeline(self.specs[name], X.iloc[tr], self.cfg.seed, self.cfg.high_cardinality)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X.iloc[tr], y[tr])
                    accs.append(float(np.mean(pipe.predict(X.iloc[te]) == y[te])))
            rows.append({"rank": rank, "pipeline": name, "cv_accuracy": float(np.mean(accs)),
                         "seconds": time.perf_counter() - t0})
        return pd.DataFrame(rows)


def recommend_csv(meta_dir: str | Path, csv: str | Path, target: str, verify_k: int = 0,
                  selector: str = "lambdarank") -> Recommendation:
    df = pd.read_csv(csv)
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in {list(df.columns)[:20]}")
    return Recommender(meta_dir, selector).recommend(df.drop(columns=[target]), df[target], verify_k)
