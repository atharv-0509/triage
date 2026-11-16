"""Base-level dataset sources (report Section 2.1 and 2.4).

Every loader returns a list of :class:`Dataset` objects, each of which is one
*parent* dataset. Parents are later split into several tasks by
:mod:`metapilot.augment`.

* ``sklearn``        - breast cancer, wine, iris and digits (pilot).
* ``synthetic``      - ``make_classification`` with varied generation parameters.
* ``moons_circles``  - ``make_moons`` / ``make_circles`` with extra noise columns.
* ``openml``         - OpenML-CC18 (suite 99) plus the supplementary datasets
                       listed in ``Config.openml_extra``. Needs ``pip install openml``
                       and network access to openml.org.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import datasets as skds

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    source: str
    description: str = ""


def load_sources(cfg: Config) -> list[Dataset]:
    rng = np.random.default_rng(cfg.seed)
    loaders = {
        "sklearn": lambda: load_sklearn_builtins(cfg.sklearn_datasets),
        "synthetic": lambda: make_synthetic(cfg.n_synthetic, rng),
        "moons_circles": lambda: make_moons_circles(cfg.n_moons_circles, rng),
        "openml": lambda: load_openml(cfg),
    }
    parents: list[Dataset] = []
    for src in cfg.sources:
        if src not in loaders:
            raise ValueError(f"Unknown source {src!r}; choose from {sorted(loaders)}")
        loaded = loaders[src]()
        log.info("source %s: %d parent datasets", src, len(loaded))
        parents.extend(loaded)
    names = [p.name for p in parents]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate parent dataset names: {sorted(dupes)}")
    return parents


# --------------------------------------------------------------------------- #
# scikit-learn built-in datasets
# --------------------------------------------------------------------------- #

_SKLEARN = {
    "breast_cancer": (skds.load_breast_cancer, "Features of cell nuclei from breast mass images, predicting malignant or benign."),
    "wine": (skds.load_wine, "Chemical analysis of wines grown in one region of Italy, predicting the cultivar."),
    "iris": (skds.load_iris, "Sepal and petal measurements of iris flowers, predicting the species."),
    "digits": (skds.load_digits, "8x8 images of handwritten digits, predicting the digit."),
}


def load_sklearn_builtins(names: list[str] | None = None) -> list[Dataset]:
    names = list(_SKLEARN) if names is None else names
    unknown = set(names) - set(_SKLEARN)
    if unknown:
        raise ValueError(f"unknown sklearn datasets {sorted(unknown)}; choose from {sorted(_SKLEARN)}")
    out = []
    for name in names:
        loader, desc = _SKLEARN[name]
        bunch = loader(as_frame=True)
        out.append(Dataset(name, bunch.data.copy(), bunch.target.copy(), "sklearn", desc))
    return out


# --------------------------------------------------------------------------- #
# Synthetic datasets
# --------------------------------------------------------------------------- #

def _log_uniform_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))


def make_synthetic(n: int, rng: np.random.Generator) -> list[Dataset]:
    """``make_classification`` datasets spanning size, dimensionality, class
    separation, label noise and non-linearity."""
    out = []
    for i in range(n):
        n_samples = _log_uniform_int(rng, 60, 800)
        n_features = _log_uniform_int(rng, 2, 150)
        n_classes = int(rng.choice([2, 2, 2, 3, 4]))
        n_informative = int(rng.integers(1, min(n_features, 20) + 1))
        n_clusters = int(rng.choice([1, 2]))
        # make_classification requires n_classes * n_clusters <= 2 ** n_informative
        while n_classes * n_clusters > 2 ** n_informative:
            if n_informative < n_features:
                n_informative += 1
            elif n_clusters > 1:
                n_clusters -= 1
            else:
                n_classes -= 1
        n_redundant = int(rng.integers(0, n_features - n_informative + 1)) // 2
        class_sep = float(rng.uniform(0.3, 2.0))
        flip_y = float(rng.uniform(0.0, 0.15))
        minority = float(rng.uniform(0.15, 1.0 / n_classes)) if n_classes == 2 else None
        weights = [minority, 1 - minority] if minority is not None else None
        seed = int(rng.integers(2**31 - 1))

        X, y = skds.make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative,
            n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
            n_clusters_per_class=n_clusters, weights=weights, flip_y=flip_y,
            class_sep=class_sep, random_state=seed,
        )
        nonlinearity = float(rng.choice([0.0, 0.0, 0.3, 0.6, 1.0]))
        X = _apply_nonlinearity(X, nonlinearity, rng)

        desc = (
            f"Synthetic make_classification data: {n_informative} informative of {n_features} features, "
            f"{n_classes} classes, class_sep={class_sep:.2f}, label noise={flip_y:.2f}, "
            f"non-linear fraction={nonlinearity:.1f}."
        )
        cols = [f"x{j}" for j in range(n_features)]
        out.append(Dataset(f"synth_{i:03d}", pd.DataFrame(X, columns=cols), pd.Series(y, name="target"),
                           "synthetic", desc))
    return out


def _apply_nonlinearity(X: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Pass a fraction of the columns through a random non-linear map."""
    if frac <= 0:
        return X
    X = X.copy()
    k = max(1, int(round(frac * X.shape[1])))
    cols = rng.choice(X.shape[1], size=k, replace=False)
    fns = [np.sin, np.abs, np.square, lambda v: np.sign(v) * np.log1p(np.abs(v)), np.tanh]
    for c in cols:
        X[:, c] = fns[int(rng.integers(len(fns)))](X[:, c])
    return X


def make_moons_circles(n: int, rng: np.random.Generator) -> list[Dataset]:
    """Two-dimensional non-linear problems with 0-6 extra pure-noise columns."""
    out = []
    for i in range(n):
        kind = "moons" if i % 2 == 0 else "circles"
        n_samples = int(rng.integers(120, 601))
        noise = float(rng.uniform(0.05, 0.3))
        seed = int(rng.integers(2**31 - 1))
        if kind == "moons":
            X, y = skds.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        else:
            factor = float(rng.uniform(0.3, 0.8))
            X, y = skds.make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
        n_noise = int(rng.integers(0, 7))
        if n_noise:
            X = np.hstack([X, rng.normal(size=(len(X), n_noise))])
        cols = ["u", "v"] + [f"noise{j}" for j in range(n_noise)]
        desc = f"Synthetic two-dimensional {kind} problem, noise={noise:.2f}, with {n_noise} irrelevant columns."
        out.append(Dataset(f"{kind}_{i:03d}", pd.DataFrame(X, columns=cols), pd.Series(y, name="target"),
                           kind, desc))
    return out


# --------------------------------------------------------------------------- #
# OpenML (full study)
# --------------------------------------------------------------------------- #

def load_openml(cfg: Config) -> list[Dataset]:
    """OpenML-CC18 plus supplementary datasets named in ``cfg.openml_extra``.

    Datasets that fail to download or parse are skipped with a warning; the
    report plans for ~165 downloads to end up with ~150 usable parents.
    """
    try:
        import openml
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("The 'openml' source needs `pip install openml`.") from e

    openml.config.set_root_cache_directory(cfg.openml_cache_dir)
    _patch_openml_parquet_download(openml)
    out: list[Dataset] = []
    seen: set[int] = set()

    suite = openml.study.get_suite(cfg.openml_suite_id)
    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            ds = task.get_dataset()
            if ds.dataset_id in seen:
                continue
            out.append(_openml_to_dataset(ds, task.target_name, "openml_cc18"))
            seen.add(ds.dataset_id)
        except Exception as e:  # noqa: BLE001 - OpenML failures are varied
            log.warning("skipping OpenML task %s: %s", task_id, e)

    for ref in cfg.openml_extra:
        try:
            ds = openml.datasets.get_dataset(int(ref) if str(ref).isdigit() else ref, download_data=True)
            if ds.dataset_id in seen:
                continue
            out.append(_openml_to_dataset(ds, ds.default_target_attribute, "openml_extra"))
            seen.add(ds.dataset_id)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping OpenML dataset %s: %s", ref, e)
    return out


def _download_parquet_https(source: str, destination, exists_ok: bool = True, proxy=None) -> None:
    """Drop-in for ``openml._api_calls._download_minio_file`` that uses HTTPS.

    openml 0.15 builds its MinIO client with ``secure=False``, so parquet files
    are requested over plain HTTP. Behind an HTTPS-only proxy that fails (405),
    and openml falls back to the ARFF URL on ``openml.org``, which may not be
    reachable either. The parquet objects are public, so a plain HTTPS GET of
    the same URL is enough.
    """
    import urllib.parse
    from pathlib import Path

    import requests

    parsed = urllib.parse.urlparse(source)
    url = parsed._replace(scheme="https").geturl()
    destination = Path(destination)
    if destination.is_dir():
        destination = destination / parsed.path.rsplit("/", 1)[-1]
    if destination.is_file():
        if not exists_ok:
            raise FileExistsError(f"File already exists in {destination}.")
        return
    resp = requests.get(url, timeout=300)
    if resp.status_code == 404:
        raise FileNotFoundError(f"Object at '{url}' does not exist.")
    if resp.status_code != 200:
        raise FileNotFoundError(f"Could not download {url}: HTTP {resp.status_code}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(destination)


def _patch_openml_parquet_download(openml_module) -> None:
    openml_module._api_calls._download_minio_file = _download_parquet_https


def _openml_to_dataset(ds, target: str, source: str) -> Dataset:
    X, y, _, _ = ds.get_data(target=target)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = pd.Series(y, name=target)
    desc = (ds.description or "").strip()
    name = f"{ds.name}_{ds.dataset_id}"
    return Dataset(name, X, y, source, desc)
