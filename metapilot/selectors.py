"""Pipeline selectors: baselines and Phase 1 meta-models (report Sections 4.3, 4.5).

Every selector has the same interface::

    sel.fit(mf_train, perf_train)      # meta-features and accuracies of training tasks
    scores = sel.score(mf_test)        # task x pipeline, higher = recommend earlier
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.preprocessing import StandardScaler

from .metalevel import gap_to_best, long_format, normalised_scores, relevance_labels

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=1)


class Selector:
    name = "selector"

    def fit(self, mf: pd.DataFrame, perf: pd.DataFrame) -> "Selector":
        self.pipelines_ = list(perf.columns)
        return self

    def score(self, mf: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RandomSelector(Selector):
    """Lower bound: a uniformly random order for every task."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, mf):
        rng = np.random.default_rng(self.seed)
        return pd.DataFrame(rng.random((len(mf), len(self.pipelines_))), index=mf.index, columns=self.pipelines_)


class SingleBest(Selector):
    """The same order for every task: pipelines sorted by mean training accuracy."""

    name = "Single best"

    def fit(self, mf, perf):
        super().fit(mf, perf)
        self.mean_ = perf.mean(axis=0)
        return self

    def score(self, mf):
        return pd.DataFrame(np.tile(self.mean_.to_numpy(), (len(mf), 1)), index=mf.index, columns=self.pipelines_)


class KNNMetaLearner(Selector):
    """auto-sklearn style: average the normalised pipeline scores of the k
    nearest training tasks in standardised meta-feature space."""

    name = "kNN meta-learner"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, mf, perf):
        super().fit(mf, perf)
        self.scaler_ = StandardScaler().fit(mf)
        self.Z_ = self.scaler_.transform(mf)
        self.norm_ = normalised_scores(perf).to_numpy()
        return self

    def score(self, mf):
        Z = self.scaler_.transform(mf)
        d = ((Z[:, None, :] - self.Z_[None, :, :]) ** 2).sum(axis=2)
        k = min(self.k, len(self.Z_))
        nn = np.argsort(d, axis=1, kind="stable")[:, :k]
        return pd.DataFrame(self.norm_[nn].mean(axis=1), index=mf.index, columns=self.pipelines_)


class LGBMGapRegressor(Selector):
    """Predict the gap to the best pipeline for each (task, pipeline) pair."""

    name = "LightGBM regressor"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, mf, perf):
        super().fit(mf, perf)
        X = long_format(mf, self.pipelines_)
        y = gap_to_best(perf).stack().loc[X.index].to_numpy()
        self.model_ = LGBMRegressor(**LGBM_PARAMS, random_state=self.seed).fit(X, y)
        return self

    def score(self, mf):
        X = long_format(mf, self.pipelines_)
        pred = self.model_.predict(X)
        return pd.DataFrame(pred.reshape(len(mf), len(self.pipelines_)), index=mf.index, columns=self.pipelines_)

    def feature_gain(self) -> pd.Series:
        booster = self.model_.booster_
        return pd.Series(booster.feature_importance("gain"), index=booster.feature_name())


class LGBMLambdaRank(Selector):
    """Same inputs as the regressor, trained with the LambdaRank objective
    on per-task relevance labels 0..10."""

    name = "LightGBM LambdaRank"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, mf, perf):
        super().fit(mf, perf)
        X = long_format(mf, self.pipelines_)
        y = relevance_labels(perf).stack().loc[X.index].to_numpy()
        group = [len(self.pipelines_)] * len(mf)
        self.model_ = LGBMRanker(**LGBM_PARAMS, objective="lambdarank", random_state=self.seed).fit(X, y, group=group)
        return self

    def score(self, mf):
        X = long_format(mf, self.pipelines_)
        pred = self.model_.predict(X)
        return pd.DataFrame(pred.reshape(len(mf), len(self.pipelines_)), index=mf.index, columns=self.pipelines_)


def default_selectors(seed: int = 0, knn_k: int = 5) -> list[Selector]:
    return [
        RandomSelector(seed),
        SingleBest(),
        KNNMetaLearner(knn_k),
        LGBMGapRegressor(seed),
        LGBMLambdaRank(seed),
    ]
