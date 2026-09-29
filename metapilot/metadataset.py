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
* ``parts/``           - one JSON checkpoint per task (used to resume and to
                         combine shards run on different machines)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import signal
import threading
import time
import warnings
import zlib
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
from .sources import Dataset, load_sources

log = logging.getLogger(__name__)

# Bump when a code change alters task results (sampling, meta-features,
# pipelines), so parts computed by older code are recomputed or rejected.
RESULTS_VERSION = 2


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


def build(cfg: Config, out_dir: str | Path, shard: tuple[int, int] = (0, 1)) -> Path:
    """Evaluate every task of this shard, one checkpoint file per task.

    Re-running skips tasks that already have a result, so an interrupted build
    resumes where it stopped. With ``shard=(i, n)`` only tasks whose id hashes
    to ``i`` are run; each group member runs one shard and :func:`assemble`
    combines the ``parts/`` folders afterwards.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    portfolio = get_portfolio(cfg.portfolio)
    pd.DataFrame([{"pipeline": s.name, "family": s.family, "scaled": s.scale} for s in portfolio]) \
        .to_csv(out / "portfolio.csv", index=False)

    tasks = generate_tasks(cfg)
    log.info("%d tasks x %d pipelines x %d CV folds = %d model fits",
             len(tasks), len(portfolio), cfg.cv_folds, len(tasks) * len(portfolio) * cfg.cv_folds)

    mine = [t for t in tasks if in_shard(t.task_id, shard)]
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    key = config_key(cfg)
    todo = [t for t in mine if not _part_done(parts / part_name(t.task_id), key)]
    log.info("shard %d/%d: %d tasks, %d already done, %d to run", shard[0], shard[1], len(mine),
             len(mine) - len(todo), len(todo))

    t0 = time.perf_counter()
    Parallel(n_jobs=cfg.n_jobs, verbose=5)(delayed(_evaluate_and_save)(t, cfg, parts, key) for t in todo)
    log.info("portfolio evaluation took %.1f s", time.perf_counter() - t0)

    if shard == (0, 1):
        assemble(out, cfg, expected=[t.task_id for t in tasks])
    else:
        log.info("shard finished; copy %s from every machine into one directory and run `metapilot assemble`", parts)
    return out


# --------------------------------------------------------------------------- #
# Task generation, sharding and checkpointing
# --------------------------------------------------------------------------- #

def generate_tasks(cfg: Config) -> list[Task]:
    """Load all parents and augment them into tasks.

    Each parent gets its own RNG derived from its name, so the task list is
    identical on every machine even if a different set of parents loads.
    """
    parents = dedupe_parents(load_sources(cfg))
    tasks: list[Task] = []
    for p in parents:
        try:
            tasks.extend(make_tasks(p, cfg, np.random.default_rng([cfg.seed, _stable_hash(p.name)])))
        except Exception as e:  # noqa: BLE001
            log.warning("parent %s skipped: %s", p.name, e)
    log.info("%d parents -> %d tasks", len(parents), len(tasks))
    return tasks


def dedupe_parents(parents: list[Dataset]) -> list[Dataset]:
    """Drop parents that look like copies of an earlier one (same shape and
    class counts). Copies in different outer folds would leak across the split."""
    seen: dict[tuple, str] = {}
    out = []
    for p in parents:
        fp = (p.X.shape, tuple(sorted(pd.Series(np.asarray(p.y)).value_counts().tolist())))
        if fp in seen:
            log.warning("parent %s looks like a duplicate of %s; skipped", p.name, seen[fp])
            continue
        seen[fp] = p.name
        out.append(p)
    return out


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def in_shard(task_id: str, shard: tuple[int, int]) -> bool:
    i, n = shard
    return _stable_hash(task_id) % n == i


def parse_shard(text: str) -> tuple[int, int]:
    """``"2/4"`` -> ``(2, 4)``; shards are numbered from 0."""
    i, n = (int(v) for v in text.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard index must be in 0..{n - 1}, got {text}")
    return i, n


def part_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", task_id) + ".json"


def config_key(cfg: Config) -> str:
    """Hash of every setting that changes a task's result. Parts written with
    a different key are recomputed (build) or rejected (assemble)."""
    fields = ("seed", "max_rows", "subsamples_per_parent", "row_frac", "col_frac", "min_class_count",
              "high_cardinality", "landmarker_rows", "portfolio", "cv_folds", "fit_timeout")
    d = cfg.to_dict()
    payload = {f: d[f] for f in fields} | {"results_version": RESULTS_VERSION}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _part_done(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("config_key") == key
    except json.JSONDecodeError:
        return False  # a partially written file from an interrupted run


def _evaluate_and_save(task: Task, cfg: Config, parts: Path, key: str) -> None:
    try:
        result = evaluate_task(task, cfg)
    except Exception:  # noqa: BLE001 - one bad task must not stop a long build; it is retried next run
        log.exception("task %s failed; it will be retried on the next build", task.task_id)
        return
    result["config_key"] = key
    path = parts / part_name(task.task_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, default=_json_default))
    tmp.replace(path)  # atomic, so an interrupted run never leaves half a result


def _json_default(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    raise TypeError(type(v))


def assemble(out_dir: str | Path, cfg: Config | None = None, expected: list[str] | None = None) -> Path:
    """Combine ``out_dir/parts/*.json`` into the meta-dataset CSVs."""
    out = Path(out_dir)
    if cfg is None:
        cfg = Config.load(out / "config.yaml")
    key = config_key(cfg)
    results = []
    for path in sorted((out / "parts").glob("*.json")):
        r = json.loads(path.read_text())
        if r.get("config_key") != key:
            raise ValueError(f"{path.name} was built with a different config ({r.get('config_key')} != {key})")
        results.append(r)
    if not results:
        raise ValueError(f"no results in {out / 'parts'}")
    if expected is not None:
        missing = set(expected) - {r["task"]["task_id"] for r in results}
        if missing:
            log.warning("%d expected tasks have no result yet, e.g. %s", len(missing), sorted(missing)[:3])

    names = [s.name for s in get_portfolio(cfg.portfolio)]
    ids = pd.Index([r["task"]["task_id"] for r in results], name="task_id")
    pd.DataFrame([r["task"] for r in results]).to_csv(out / "tasks.csv", index=False)
    pd.DataFrame([r["mf"] for r in results], index=ids).to_csv(out / "metafeatures.csv")
    pd.DataFrame([r["acc"] for r in results], index=ids)[names].to_csv(out / "performance.csv")
    pd.DataFrame([r["secs"] for r in results], index=ids)[names].to_csv(out / "fit_seconds.csv")
    with open(out / "text_meta.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r["text"]) + "\n")
    log.info("assembled %d tasks from %d parents into %s", len(results),
             len({r["task"]["parent"] for r in results}), out)
    return out
