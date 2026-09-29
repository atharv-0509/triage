"""Meta-feature extraction (report Section 3.3, Table 4).

14 meta-features in four groups. Landmarkers are 2-fold CV accuracies of fast
models on at most ``landmarker_rows`` randomly chosen rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .preprocess import numeric_matrix, stratified_cap

GROUPS: dict[str, list[str]] = {
    "simple": ["log_n_rows", "log_n_cols", "cols_rows_ratio", "n_classes"],
    "statistical": ["mean_abs_corr", "mean_abs_skew", "mean_kurtosis", "pca_frac_95"],
    "info_theoretic": ["class_entropy", "minority_majority_ratio"],
    "landmarkers": ["lm_stump", "lm_1nn", "lm_gnb", "lm_lda"],
}
FEATURES: list[str] = [f for g in GROUPS.values() for f in g]
MAX_CORR_COLS = 500


def extract(X: pd.DataFrame, y: np.ndarray, landmarker_rows: int = 200, seed: int = 0) -> dict[str, float]:
    Xn = numeric_matrix(X)
    y = np.asarray(y)
    n, p = Xn.shape
    classes, counts = np.unique(y, return_counts=True)
    k = len(classes)

    mf: dict[str, float] = {
        "log_n_rows": float(np.log(n)),
        "log_n_cols": float(np.log(p)),
        "cols_rows_ratio": p / n,
        "n_classes": float(k),
    }

    std = Xn.std(axis=0)
    Xv = Xn[:, std > 0]  # drop constant columns for the statistical group
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        if Xv.shape[1] >= 2:
            # On very wide data (e.g. arcene, 10k columns) the full correlation
            # matrix is too large; estimate the mean from a random column subset.
            cols = np.random.default_rng(seed).choice(Xv.shape[1], size=min(Xv.shape[1], MAX_CORR_COLS),
                                                      replace=False)
            corr = np.corrcoef(Xv[:, np.sort(cols)], rowvar=False)
            iu = np.triu_indices_from(corr, k=1)
            mf["mean_abs_corr"] = float(np.nanmean(np.abs(corr[iu])))
        else:
            mf["mean_abs_corr"] = 0.0
        if Xv.shape[1] >= 1:
            mf["mean_abs_skew"] = float(np.nanmean(np.abs(stats.skew(Xv, axis=0))))
            mf["mean_kurtosis"] = float(np.nanmean(stats.kurtosis(Xv, axis=0)))
            Z = StandardScaler().fit_transform(Xv)
            ratio = PCA().fit(Z).explained_variance_ratio_
            n95 = int(np.searchsorted(np.cumsum(ratio), 0.95) + 1)
            mf["pca_frac_95"] = min(n95, len(ratio)) / len(ratio)
        else:
            mf.update(mean_abs_skew=0.0, mean_kurtosis=0.0, pca_frac_95=1.0)

    probs = counts / counts.sum()
    mf["class_entropy"] = float(-(probs * np.log(probs)).sum() / np.log(k)) if k > 1 else 0.0
    mf["minority_majority_ratio"] = float(counts.min() / counts.max())

    mf.update(_landmarkers(Xn, y, landmarker_rows, seed))
    return {f: float(np.nan_to_num(mf[f], nan=0.0, posinf=0.0, neginf=0.0)) for f in FEATURES}


def _landmarkers(Xn: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> dict[str, float]:
    Xs, ys = stratified_cap(pd.DataFrame(Xn), y, max_rows, seed)
    Xs = Xs.to_numpy()
    models = {
        "lm_stump": DecisionTreeClassifier(max_depth=1, random_state=seed),
        "lm_1nn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        "lm_gnb": GaussianNB(),
        "lm_lda": LinearDiscriminantAnalysis(),
    }
    majority = float(np.bincount(ys).max() / len(ys))
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    out = {}
    for name, model in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = float(cross_val_score(model, Xs, ys, cv=cv, scoring="accuracy").mean())
        except Exception:  # noqa: BLE001 - e.g. a class with a single sample in the subsample
            out[name] = majority
    return out
