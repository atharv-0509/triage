"""Build the meta-level dataset (report Section 2.2).

For every task: extract meta-features, store text metadata (for Phase 2) and
run every pipeline in the portfolio with stratified k-fold CV. Results are
written to ``out_dir``:

* ``tasks.csv``        - one row per task (parent, source, shape, ...)
* ``metafeatures.csv`` - 14 meta-features per task
* ``performance.csv``  - mean CV accuracy, task x pipeline (blank = failed/timeout)
* ``fit_seconds.csv``  - total fit+predict time, task x pipeline
* ``portfolio.csv``    - pipeline name and family
* ``text_meta.jsonl``  - column names, dtypes and description per task
* ``config.yaml``      - the config used
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from . import metafeatures
from .augment import Task, make_tasks
from .config import Config
from .portfolio import build_pipeline, get_portfolio
from .preprocess import split_columns
from .sources import load_sources

log = logging.getLogger(__name__)


class FitTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Raise :class:`FitTimeout` after ``seconds`` using SIGALRM where possible.

    SIGALRM is delivered only between Python bytecodes, so a long call inside
    compiled code (e.g. libsvm) finishes before it is interrupted; the caller
    also checks elapsed time afterwards so such fits are still marked as timed out.
    """
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable or seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise FitTimeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def evaluate_task(task: Task, cfg: Config) -> dict:
    """Meta-features + CV accuracy of every pipeline on one task."""
    portfolio = get_portfolio(cfg.portfolio)
    seed = cfg.seed
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(task.X, task.y))

    acc, secs = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in portfolio:
            scores, elapsed = [], 0.0
            try:
                for tr, te in folds:
                    Xtr, Xte = task.X.iloc[tr], task.X.iloc[te]
                    pipe = build_pipeline(spec, Xtr, seed, cfg.high_cardinality)
                    t0 = time.perf_counter()
                    with time_limit(cfg.fit_timeout):
                        pipe.fit(Xtr, task.y[tr])
                        pred = pipe.predict(Xte)
                    dt = time.perf_counter() - t0
                    elapsed += dt
                    if dt > cfg.fit_timeout:
                        raise FitTimeout()
                    scores.append(float(np.mean(pred == task.y[te])))
                acc[spec.name] = float(np.mean(scores))
            except FitTimeout:
                log.info("%s / %s: timeout", task.task_id, spec.name)
                acc[spec.name] = np.nan
            except Exception as e:  # noqa: BLE001 - any failing fit is recorded as missing
                log.info("%s / %s: failed (%s)", task.task_id, spec.name, e)
                acc[spec.name] = np.nan
            secs[spec.name] = elapsed

    mf = metafeatures.extract(task.X, task.y, cfg.landmarker_rows, seed)
    return {"task": task_info(task, cfg), "mf": mf, "acc": acc, "secs": secs, "text": text_meta(task)}


def task_info(task: Task, cfg: Config) -> dict:
    num, low, high = split_columns(task.X, cfg.high_cardinality)
    return {
        "task_id": task.task_id,
        "parent": task.parent,
        "source": task.source,
        "n_rows": len(task.X),
        "n_cols": task.X.shape[1],
        "n_classes": task.n_classes,
        "n_numeric": len(num),
        "n_categorical": len(low) + len(high),
        "missing_frac": float(task.X.isna().to_numpy().mean()),
    }


def text_meta(task: Task) -> dict:
    """Raw material for the Phase 2 text prompts (report Section 3.6)."""
    counts = np.bincount(task.y)
    return {
        "task_id": task.task_id,
        "parent": task.parent,
        "n_rows": len(task.X),
        "n_classes": task.n_classes,
        "minority_frac": float(counts.min() / counts.sum()),
        "missing_frac": float(task.X.isna().to_numpy().mean()),
        "columns": [{"name": c, "dtype": "numeric" if pd.api.types.is_numeric_dtype(task.X[c]) else "categorical"}
                    for c in task.X.columns],
        "description": task.description,
    }


def build(cfg: Config, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    portfolio = get_portfolio(cfg.portfolio)
    pd.DataFrame([{"pipeline": s.name, "family": s.family, "scaled": s.scale} for s in portfolio]) \
        .to_csv(out / "portfolio.csv", index=False)

    parents = load_sources(cfg)
    rng = np.random.default_rng(cfg.seed)
    tasks: list[Task] = []
    for p in parents:
        try:
            tasks.extend(make_tasks(p, cfg, rng))
        except Exception as e:  # noqa: BLE001
            log.warning("parent %s skipped: %s", p.name, e)
    log.info("%d parents -> %d tasks, %d pipelines, %d CV folds = %d model fits",
             len(parents), len(tasks), len(portfolio), cfg.cv_folds, len(tasks) * len(portfolio) * cfg.cv_folds)

    t0 = time.perf_counter()
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(delayed(evaluate_task)(t, cfg) for t in tasks)
    log.info("portfolio evaluation took %.1f s", time.perf_counter() - t0)

    ids = [r["task"]["task_id"] for r in results]
    pd.DataFrame([r["task"] for r in results]).to_csv(out / "tasks.csv", index=False)
    pd.DataFrame([r["mf"] for r in results], index=pd.Index(ids, name="task_id")).to_csv(out / "metafeatures.csv")
    names = [s.name for s in portfolio]
    pd.DataFrame([r["acc"] for r in results], index=pd.Index(ids, name="task_id"))[names] \
        .to_csv(out / "performance.csv")
    pd.DataFrame([r["secs"] for r in results], index=pd.Index(ids, name="task_id"))[names] \
        .to_csv(out / "fit_seconds.csv")
    with open(out / "text_meta.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r["text"]) + "\n")
    return out
