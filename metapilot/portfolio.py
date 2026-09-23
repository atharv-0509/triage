"""Candidate pipeline portfolio (report Section 2.3, Table 2).

Each entry is a named algorithm configuration. :func:`build_pipeline` wraps it
with the shared preprocessor (Section 3.1); scaling is added only for the
scale-sensitive families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocess import make_preprocessor


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    family: str
    scale: bool
    factory: Callable[[int], BaseEstimator]  # seed -> unfitted estimator


def _lr(C):
    return lambda s: LogisticRegression(C=C, max_iter=2000)


def _knn(k, weights="uniform"):
    return lambda s: KNeighborsClassifier(n_neighbors=k, weights=weights)


def _svm(C):
    return lambda s: SVC(C=C, kernel="rbf", gamma="scale", random_state=s)


def _mlp(hidden, alpha=1e-4):
    return lambda s: MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=300, random_state=s)


PILOT: list[PipelineSpec] = [
    PipelineSpec("logreg_C0.1", "logistic_regression", True, _lr(0.1)),
    PipelineSpec("logreg_C10", "logistic_regression", True, _lr(10.0)),
    PipelineSpec("ridge", "ridge", True, lambda s: RidgeClassifier()),
    PipelineSpec("gaussian_nb", "naive_bayes", False, lambda s: GaussianNB()),
    PipelineSpec("knn_1", "knn", True, _knn(1)),
    PipelineSpec("knn_5", "knn", True, _knn(5)),
    PipelineSpec("knn_15", "knn", True, _knn(15)),
    PipelineSpec("tree_d3", "decision_tree", False, lambda s: DecisionTreeClassifier(max_depth=3, random_state=s)),
    PipelineSpec("tree_full", "decision_tree", False, lambda s: DecisionTreeClassifier(random_state=s)),
    PipelineSpec("rf_100", "random_forest", False, lambda s: RandomForestClassifier(100, random_state=s)),
    PipelineSpec("rf_100_d5", "random_forest", False,
                 lambda s: RandomForestClassifier(100, max_depth=5, random_state=s)),
    PipelineSpec("et_100", "extra_trees", False, lambda s: ExtraTreesClassifier(100, random_state=s)),
    PipelineSpec("hgb_default", "gradient_boosting", False, lambda s: HistGradientBoostingClassifier(random_state=s)),
    PipelineSpec("hgb_shallow_lr0.05", "gradient_boosting", False,
                 lambda s: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=s)),
    PipelineSpec("svm_rbf_C1", "svm_rbf", True, _svm(1.0)),
    PipelineSpec("svm_rbf_C10", "svm_rbf", True, _svm(10.0)),
    PipelineSpec("mlp_32", "mlp", True, _mlp((32,))),
    PipelineSpec("mlp_128", "mlp", True, _mlp((128,))),
]

# Full study: 3-6 configurations per family (~40 pipelines).
FULL: list[PipelineSpec] = [
    *[PipelineSpec(f"logreg_C{C:g}", "logistic_regression", True, _lr(C)) for C in (0.01, 0.1, 1.0, 10.0, 100.0)],
    *[PipelineSpec(f"ridge_a{a:g}", "ridge", True, (lambda a: lambda s: RidgeClassifier(alpha=a))(a))
      for a in (0.1, 1.0, 10.0)],
    *[PipelineSpec(f"gaussian_nb_vs{v:g}", "naive_bayes", False, (lambda v: lambda s: GaussianNB(var_smoothing=v))(v))
      for v in (1e-9, 1e-6, 1e-3)],
    PipelineSpec("knn_1", "knn", True, _knn(1)),
    PipelineSpec("knn_5", "knn", True, _knn(5)),
    PipelineSpec("knn_15", "knn", True, _knn(15)),
    PipelineSpec("knn_30_dist", "knn", True, _knn(30, "distance")),
    PipelineSpec("tree_d3", "decision_tree", False, lambda s: DecisionTreeClassifier(max_depth=3, random_state=s)),
    PipelineSpec("tree_d6", "decision_tree", False, lambda s: DecisionTreeClassifier(max_depth=6, random_state=s)),
    PipelineSpec("tree_full", "decision_tree", False, lambda s: DecisionTreeClassifier(random_state=s)),
    PipelineSpec("tree_leaf5", "decision_tree", False,
                 lambda s: DecisionTreeClassifier(min_samples_leaf=5, random_state=s)),
    PipelineSpec("rf_100", "random_forest", False, lambda s: RandomForestClassifier(100, random_state=s)),
    PipelineSpec("rf_100_d5", "random_forest", False,
                 lambda s: RandomForestClassifier(100, max_depth=5, random_state=s)),
    PipelineSpec("rf_300_mf0.5", "random_forest", False,
                 lambda s: RandomForestClassifier(300, max_features=0.5, random_state=s)),
    PipelineSpec("rf_100_leaf5", "random_forest", False,
                 lambda s: RandomForestClassifier(100, min_samples_leaf=5, random_state=s)),
    PipelineSpec("et_100", "extra_trees", False, lambda s: ExtraTreesClassifier(100, random_state=s)),
    PipelineSpec("et_300", "extra_trees", False, lambda s: ExtraTreesClassifier(300, random_state=s)),
    PipelineSpec("et_100_d8", "extra_trees", False,
                 lambda s: ExtraTreesClassifier(100, max_depth=8, random_state=s)),
    PipelineSpec("hgb_default", "gradient_boosting", False, lambda s: HistGradientBoostingClassifier(random_state=s)),
    PipelineSpec("hgb_shallow_lr0.05", "gradient_boosting", False,
                 lambda s: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=s)),
    PipelineSpec("hgb_leaves63", "gradient_boosting", False,
                 lambda s: HistGradientBoostingClassifier(max_leaf_nodes=63, min_samples_leaf=5, random_state=s)),
    PipelineSpec("hgb_l2_lr0.05", "gradient_boosting", False,
                 lambda s: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                                          random_state=s)),
    *[PipelineSpec(f"svm_rbf_C{C:g}", "svm_rbf", True, _svm(C)) for C in (0.1, 1.0, 10.0, 100.0)],
    PipelineSpec("mlp_32", "mlp", True, _mlp((32,))),
    PipelineSpec("mlp_128", "mlp", True, _mlp((128,))),
    PipelineSpec("mlp_128x128", "mlp", True, _mlp((128, 128))),
    PipelineSpec("mlp_128_a0.01", "mlp", True, _mlp((128,), alpha=1e-2)),
]


def _tabpfn(seed: int) -> BaseEstimator:
    from tabpfn import TabPFNClassifier  # optional dependency

    return TabPFNClassifier(random_state=seed)


PORTFOLIOS: dict[str, list[PipelineSpec]] = {
    "pilot": PILOT,
    "full": FULL,
    "full_tabpfn": FULL + [PipelineSpec("tabpfn", "tabpfn", False, _tabpfn)],
}


def get_portfolio(name: str) -> list[PipelineSpec]:
    if name not in PORTFOLIOS:
        raise ValueError(f"Unknown portfolio {name!r}; choose from {sorted(PORTFOLIOS)}")
    specs = PORTFOLIOS[name]
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), "pipeline names must be unique"
    if name == "full_tabpfn":
        try:
            import tabpfn  # noqa: F401
        except ImportError as e:
            raise ImportError("portfolio 'full_tabpfn' needs `pip install tabpfn`") from e
    return specs


def build_pipeline(spec: PipelineSpec, X: pd.DataFrame, seed: int, high_cardinality: int = 20) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X, scale=spec.scale, high_cardinality=high_cardinality)),
        ("model", spec.factory(seed)),
    ])


def family_of(portfolio: list[PipelineSpec]) -> dict[str, str]:
    return {s.name: s.family for s in portfolio}
