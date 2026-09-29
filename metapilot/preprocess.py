"""Preprocessing (report Section 3.1).

Two levels:

* :func:`clean_dataset` runs once per dataset, before cross-validation. It only
  does label-free or label-only work: drop rows with a missing target, encode
  labels as ``0..K-1``, drop empty/constant columns, normalise column dtypes and
  apply the stratified row cap.
* :func:`make_preprocessor` builds a ``ColumnTransformer`` that sits *inside*
  every candidate pipeline, so imputation, encoding and scaling are all fitted
  on the training folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataset(X: pd.DataFrame, y, max_rows: int | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    X = X.reset_index(drop=True).copy()
    y = pd.Series(np.asarray(y)).reset_index(drop=True)

    keep = y.notna().to_numpy()
    X, y = X.loc[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    X.columns = [str(c) for c in X.columns]
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("string").astype("category")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] == 0:
        raise ValueError("no informative columns left after cleaning")

    y_codes = pd.factorize(y, sort=True)[0].astype(int)
    X, y_codes = stratified_cap(X, y_codes, max_rows, seed)
    return X, y_codes


def stratified_cap(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, seed: int):
    """Reduce to ``max_rows`` rows by stratified random sampling."""
    y = np.asarray(y)
    if max_rows is None or len(X) <= max_rows:
        return X.reset_index(drop=True), y
    # Proportional allocation per class (largest remainder), then sample within
    # each class. Unlike train_test_split this works for any sizes, e.g. 201 -> 200 rows.
    classes, counts = np.unique(y, return_counts=True)
    quota = counts * max_rows / len(y)
    take = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - take), kind="stable")[: max_rows - take.sum()]:
        take[i] += 1
    rng = np.random.default_rng(seed)
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), size=k, replace=False)
                                   for c, k in zip(classes, take)]))
    return X.iloc[rows].reset_index(drop=True), y[rows]


def split_columns(X: pd.DataFrame, high_cardinality: int) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, low-cardinality categorical, high-cardinality categorical)."""
    num, low, high = [], [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num.append(c)
        elif X[c].nunique(dropna=True) > high_cardinality:
            high.append(c)
        else:
            low.append(c)
    return num, low, high


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its relative frequency in the training data."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X.iloc[:, j].value_counts(normalize=True).to_dict() for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = [X.iloc[:, j].map(m).astype(float).fillna(0.0).to_numpy() for j, m in enumerate(self.maps_)]
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


def make_preprocessor(X: pd.DataFrame, scale: bool, high_cardinality: int = 20) -> ColumnTransformer:
    num, low, high = split_columns(X, high_cardinality)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    # Categories are stored as pandas category dtype; convert to object so the
    # imputers see real missing values instead of a NaN category.
    to_object = _ToObject()
    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps), num))
    if low:
        steps = [("obj", to_object), ("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("cat", Pipeline(steps), low))
    if high:
        steps = [("obj", to_object), ("impute", SimpleImputer(strategy="most_frequent")),
                 ("freq", FrequencyEncoder())]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("freq", Pipeline(steps), high))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


class _ToObject(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(object)
        return X.where(pd.notna(X), np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


def numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """Simple numeric view used for meta-feature extraction: categoricals become
    integer codes and missing values are replaced by the column median."""
    cols = []
    for c in X.columns:
        s = X[c]
        if not pd.api.types.is_numeric_dtype(s):
            codes = s.astype("category").cat.codes.astype(float)
            s = codes.where(codes >= 0, np.nan)
        s = s.astype(float)
        cols.append(s.fillna(s.median() if s.notna().any() else 0.0).to_numpy())
    return np.column_stack(cols)
