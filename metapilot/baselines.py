"""FLAML AutoML baseline (report Section 4.5, full study).

FLAML runs on the *base-level* tasks, not the meta-level: for every task it
searches for a model on each training fold within a fixed time budget and is
scored on the same held-out folds as the portfolio. Its regret is measured
against the best portfolio pipeline, so it can be negative when FLAML finds a
better model than anything in the portfolio.

Results go to ``<meta_dir>/flaml/parts/`` (one checkpoint per task, so runs can
be resumed or sharded like ``build``) and are combined into
``<meta_dir>/flaml.csv``, which ``evaluate`` picks up automatically.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from .augment import Task
from .config import Config
from .metadataset import generate_tasks, in_shard, part_name

log = logging.getLogger(__name__)


def evaluate_flaml(task: Task, cfg: Config, budget: float) -> dict:
    from flaml import AutoML  # optional dependency: pip install "flaml[automl]"

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    scores, seconds, chosen = [], 0.0, []
    for tr, te in cv.split(task.X, task.y):
        automl = AutoML()
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                automl.fit(task.X.iloc[tr], task.y[tr], task="classification", metric="accuracy",
                           time_budget=budget, seed=cfg.seed, n_jobs=1, verbose=0)
                pred = np.asarray(automl.predict(task.X.iloc[te]))
            scores.append(float(np.mean(pred == task.y[te])))
            chosen.append(str(automl.best_estimator))
        except Exception as e:  # noqa: BLE001 - a failed search counts as a missing result
            log.info("%s: FLAML failed (%s)", task.task_id, e)
            scores.append(np.nan)
        seconds += time.perf_counter() - t0
    acc = float(np.mean(scores)) if not np.isnan(scores).any() else None
    return {"task_id": task.task_id, "accuracy": acc, "seconds": seconds, "budget_per_fold": budget,
            "best_estimators": ",".join(chosen)}


def run(cfg: Config, meta_dir: str | Path, budget: float, shard: tuple[int, int] = (0, 1),
        max_tasks: int | None = None) -> Path:
    meta = Path(meta_dir)
    parts = meta / "flaml" / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks(cfg)
    built = meta / "performance.csv"
    if built.exists():  # only tasks that are in the meta-dataset
        ids = set(pd.read_csv(built, index_col="task_id").index)
        tasks = [t for t in tasks if t.task_id in ids]
    if max_tasks is not None:  # evenly spread subset, e.g. for a quick first run
        tasks = sorted(tasks, key=lambda t: part_name(t.task_id))
        tasks = [tasks[i] for i in np.linspace(0, len(tasks) - 1, min(max_tasks, len(tasks))).astype(int)]
    mine = [t for t in tasks if in_shard(t.task_id, shard)]
    todo = [t for t in mine if not _done(parts / part_name(t.task_id), budget)]
    log.info("FLAML (%.0fs per fold): %d tasks in shard, %d to run, about %.1f CPU-hours",
             budget, len(mine), len(todo), len(todo) * cfg.cv_folds * budget / 3600)

    Parallel(n_jobs=cfg.n_jobs, verbose=5)(delayed(_run_and_save)(t, cfg, budget, parts) for t in todo)
    return assemble(meta)


def _done(path: Path, budget: float) -> bool:
    try:
        return json.loads(path.read_text())["budget_per_fold"] == budget
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def _run_and_save(task: Task, cfg: Config, budget: float, parts: Path) -> None:
    result = evaluate_flaml(task, cfg, budget)
    path = parts / part_name(task.task_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)


def assemble(meta_dir: str | Path) -> Path:
    meta = Path(meta_dir)
    rows = [json.loads(p.read_text()) for p in sorted((meta / "flaml" / "parts").glob("*.json"))]
    budgets = {r["budget_per_fold"] for r in rows}
    if len(budgets) > 1:
        raise ValueError(f"FLAML parts were run with different budgets {sorted(budgets)}; keep one")
    out = meta / "flaml.csv"
    pd.DataFrame(rows).set_index("task_id").to_csv(out)
    log.info("FLAML results for %d tasks written to %s", len(rows), out)
    return out
