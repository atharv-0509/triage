"""Selection-quality metrics (report Sections 4.2 and 4.6).

Accuracies are in [0, 1]; regrets are returned in accuracy *percentage points*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ranking(scores: pd.DataFrame) -> np.ndarray:
    """Pipeline column indices per task, best-scored first (stable on ties)."""
    return np.argsort(-scores.to_numpy(), axis=1, kind="stable")


def regret_at_k(scores: pd.DataFrame, perf: pd.DataFrame, k: int) -> pd.Series:
    """Best accuracy on the task minus the best accuracy among the top-k
    recommended pipelines, per task."""
    perf = perf.loc[scores.index, scores.columns]
    acc = perf.to_numpy()
    top = ranking(scores)[:, :k]
    best_chosen = np.take_along_axis(acc, top, axis=1).max(axis=1)
    return pd.Series(100 * (acc.max(axis=1) - best_chosen), index=scores.index)


def regret_curve(scores: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """Per-task regret after evaluating 1..P pipelines in recommended order."""
    P = scores.shape[1]
    return pd.DataFrame({k: regret_at_k(scores, perf, k) for k in range(1, P + 1)})


def cost_at_k(scores: pd.DataFrame, secs: pd.DataFrame, k: int) -> pd.Series:
    """Seconds spent running the top-k recommended pipelines, per task."""
    s = secs.loc[scores.index, scores.columns].to_numpy()
    top = ranking(scores)[:, :k]
    return pd.Series(np.take_along_axis(s, top, axis=1).sum(axis=1), index=scores.index)


def spearman_per_task(scores: pd.DataFrame, perf: pd.DataFrame) -> pd.Series:
    perf = perf.loc[scores.index, scores.columns]
    out = []
    for s, a in zip(scores.to_numpy(), perf.to_numpy()):
        if np.ptp(s) == 0 or np.ptp(a) == 0:
            out.append(np.nan)
        else:
            out.append(spearmanr(s, a).statistic)
    return pd.Series(out, index=scores.index)


def bootstrap_ci(values: pd.Series | np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarise(scores: pd.DataFrame, perf: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
              hit_tolerance: float = 0.5) -> dict[str, float]:
    r1 = regret_at_k(scores, perf, 1)
    r3 = regret_at_k(scores, perf, 3)
    rho = spearman_per_task(scores, perf)
    r1_lo, r1_hi = bootstrap_ci(r1, n_boot, seed)
    r3_lo, r3_hi = bootstrap_ci(r3, n_boot, seed)
    return {
        "regret@1": r1.mean(), "regret@1_lo": r1_lo, "regret@1_hi": r1_hi,
        "regret@3": r3.mean(), "regret@3_lo": r3_lo, "regret@3_hi": r3_hi,
        "top1_hit": 100 * float((r1 <= hit_tolerance + 1e-9).mean()),
        "spearman": float(np.nanmean(rho)) if rho.notna().any() else np.nan,
        "n_tasks": len(scores),
    }
