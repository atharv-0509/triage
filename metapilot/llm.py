"""Phase 2: text prompts for the LLM selector and scoring of its outputs
(report Sections 3.6 and 4.4).

This module needs no GPU or deep-learning packages. Fine-tuning and
generation live in :mod:`metapilot.llm_train`, which writes raw model outputs
to ``<meta_dir>/llm/predictions/<variant>.jsonl``. Here those outputs are
parsed into rankings and scored like any other selector.

Two prompt versions are written for every task:

* **numeric** (Prompt A) - the same information LightGBM gets: the task's
  shape plus all 14 meta-features. Used for research question 2.
* **semantic** (Prompt B) - Prompt A plus column names and the dataset
  description. Compared with A on real (non-synthetic) tasks for research
  question 3.

The target is the full list of pipelines ordered by measured accuracy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metalevel import MetaData, folds_path, make_folds
from .selectors import SingleBest

SYSTEM_PROMPT = (
    "You are an AutoML assistant. Given a profile of a tabular classification dataset, rank every candidate "
    "pipeline from the one expected to reach the highest accuracy to the lowest. Answer with all candidate "
    "pipeline names, each exactly once, separated by commas, and nothing else."
)

# Parents whose data was generated, not collected; semantic text carries no
# real-world meaning for them (research question 3 uses the other tasks).
SYNTHETIC_SOURCES = {"synthetic", "moons", "circles"}

MAX_COLUMNS_SHOWN = 30
MAX_DESCRIPTION_CHARS = 600


def numeric_prompt(task: dict, mf: dict, pipelines: list[str]) -> str:
    target = "binary" if task["n_classes"] == 2 else f"{task['n_classes']} classes"
    lines = [
        "Dataset profile:",
        f"- Rows: {task['n_rows']}",
        f"- Columns: {task['n_cols']} ({task['n_numeric']} numeric, {task['n_categorical']} categorical)",
        f"- Columns-to-rows ratio: {mf['cols_rows_ratio']:.3f}",
        f"- Target: {target}; minority-to-majority class ratio {mf['minority_majority_ratio']:.3f}; "
        f"normalised class entropy {mf['class_entropy']:.3f}",
        f"- Missing values: {100 * task['missing_frac']:.1f}%",
        f"- Mean absolute feature correlation: {mf['mean_abs_corr']:.3f}",
        f"- Mean absolute skewness: {mf['mean_abs_skew']:.3f}; mean kurtosis: {mf['mean_kurtosis']:.3f}",
        f"- Fraction of principal components for 95% variance: {mf['pca_frac_95']:.3f}",
        f"- Landmarker accuracies (2-fold CV): decision stump {mf['lm_stump']:.3f}, 1-NN {mf['lm_1nn']:.3f}, "
        f"Gaussian naive Bayes {mf['lm_gnb']:.3f}, LDA {mf['lm_lda']:.3f}",
    ]
    return "\n".join(lines + _candidates(pipelines))


def semantic_prompt(task: dict, mf: dict, pipelines: list[str], text: dict) -> str:
    names = [c["name"] for c in text.get("columns", [])]
    shown = ", ".join(names[:MAX_COLUMNS_SHOWN])
    if len(names) > MAX_COLUMNS_SHOWN:
        shown += f", ... ({len(names) - MAX_COLUMNS_SHOWN} more)"
    desc = _clean_description(text.get("description", "")) or "No description available."
    profile = numeric_prompt(task, mf, pipelines).split("\nCandidate pipelines:")[0]
    lines = [profile, f"- Column names: {shown}", f"- Description: {desc}"]
    return "\n".join(lines + _candidates(pipelines))


def _candidates(pipelines: list[str]) -> list[str]:
    return ["", "Candidate pipelines: " + ", ".join(pipelines), "",
            "Rank all candidate pipelines from best to worst."]


def _clean_description(desc: str) -> str:
    """OpenML descriptions are long markdown; keep a short plain-text start."""
    desc = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", desc)    # [text](link) -> text
    desc = re.sub(r"[#*_`>]+", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    if len(desc) > MAX_DESCRIPTION_CHARS:
        desc = desc[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + " ..."
    return desc


def target_ranking(perf_row: pd.Series) -> list[str]:
    """Pipelines by measured accuracy, best first; ties keep portfolio order."""
    order = np.argsort(-perf_row.to_numpy(), kind="stable")
    return [perf_row.index[i] for i in order]


def write_prompts(meta_dir: str | Path, cfg: Config, out: str | Path | None = None) -> Path:
    """Write ``<meta_dir>/llm/prompts.jsonl``: one record per task with both
    prompts, the target ranking and the task's outer fold (``folds.csv``, the
    split every method shares)."""
    meta = MetaData.load(meta_dir)
    folds = make_folds(meta, cfg.outer_folds, cfg.seed, folds_path(meta_dir, cfg.seed, cfg.seed))
    text = {}
    with open(Path(meta_dir) / "text_meta.jsonl") as f:
        for line in f:
            r = json.loads(line)
            text[r["task_id"]] = r

    out = Path(out) if out else Path(meta_dir) / "llm" / "prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pipelines = meta.pipelines
    with open(out, "w") as f:
        for task_id in meta.perf.index:
            task = meta.tasks.loc[task_id].to_dict()
            mf = meta.mf.loc[task_id].to_dict()
            f.write(json.dumps({
                "task_id": task_id,
                "parent": task["parent"],
                "source": task["source"],
                "fold": int(folds.loc[task_id]),
                "system": SYSTEM_PROMPT,
                "numeric": numeric_prompt(task, mf, pipelines),
                "semantic": semantic_prompt(task, mf, pipelines, text.get(task_id, {})),
                "target": ", ".join(target_ranking(meta.perf.loc[task_id])),
            }) + "\n")
    return out


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Parsing model output
# --------------------------------------------------------------------------- #

def parse_ranking(text: str, pipelines: list[str]) -> list[str]:
    """Pipeline names in the order the model first mentions them.

    Matching is case-insensitive and exact: ``rf_100`` does not match inside
    ``rf_100_d5``, nor ``svm_rbf_C1`` inside ``svm_rbf_C10``. Names the model
    leaves out are *not* filled in here; see :func:`llm_scores`.
    """
    hits = []
    for p in pipelines:
        m = re.search(r"(?<![\w.])" + re.escape(p) + r"(?![\w.])", text, flags=re.IGNORECASE)
        if m:
            hits.append((m.start(), p))
    return [p for _, p in sorted(hits)]


def llm_scores(preds: list[dict], meta: MetaData, folds: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Turn raw outputs into a score matrix (task x pipeline, higher = earlier).

    Pipelines the model did not name are appended in the single-best order of
    that task's training folds, so every ranking is complete. Because a model
    that names nothing would then score exactly like single best, the
    per-task share of pipelines the model named itself is returned as well
    (``coverage``) and must be reported next to the metrics.
    """
    pipelines = meta.pipelines
    by_task = {}
    for r in preds:
        if r["task_id"] in by_task:
            raise ValueError(f"duplicate prediction for {r['task_id']}")
        if r["task_id"] not in folds.index:
            continue
        if "fold" in r and int(r["fold"]) != int(folds.loc[r["task_id"]]):
            raise ValueError(f"{r['task_id']}: prediction made for fold {r['fold']}, but folds.csv says "
                             f"{folds.loc[r['task_id']]}; the LLM must use the shared split")
        by_task[r["task_id"]] = r["output"]

    fallback = {}
    for k in sorted(folds.unique()):
        train = folds.index[folds != k]
        sb = SingleBest().fit(meta.mf.loc[train], meta.perf.loc[train])
        fallback[k] = list(sb.mean_.sort_values(ascending=False, kind="stable").index)

    rows, coverage = {}, {}
    for task_id, output in by_task.items():
        named = parse_ranking(output, pipelines)
        order = named + [p for p in fallback[folds.loc[task_id]] if p not in named]
        rows[task_id] = {p: float(len(order) - i) for i, p in enumerate(order)}
        coverage[task_id] = len(named) / len(pipelines)
    order = [t for t in folds.index if t in rows]  # meta-dataset order, as for every other selector
    scores = pd.DataFrame.from_dict(rows, orient="index").loc[order, pipelines]
    scores.index.name = "task_id"
    return scores, pd.Series(coverage, name="coverage").loc[order]


def prediction_files(meta_dir: str | Path) -> dict[str, Path]:
    d = Path(meta_dir) / "llm" / "predictions"
    return {p.stem: p for p in sorted(d.glob("*.jsonl"))} if d.exists() else {}
