"""Phase 1 experiment: grouped outer CV over all selectors (report Section 5).

Writes to ``out_dir``:

* ``summary.csv``            - Table 6 (regret@1/@3 with 95% CIs, top-1 hit, Spearman)
* ``per_task_regret1.csv``   - Regret@1 per task and selector
* ``regret_curve.csv``       - mean regret vs. number of pipelines evaluated
* ``family_wins.csv``        - share of tasks won by each family (Figure 2)
* ``feature_gain.csv``       - LightGBM regressor gain per meta-feature (Figure 5)
* ``scores/<selector>.csv``  - out-of-fold scores (task x pipeline)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import metrics
from .config import Config
from .metafeatures import GROUPS
from .metalevel import MetaData, family_win_share, make_folds
from .selectors import LGBMGapRegressor, default_selectors

log = logging.getLogger(__name__)

DOMINANCE_LIMIT = 0.5


def run(meta_dir: str | Path, out_dir: str | Path, cfg: Config, drop_groups: tuple[str, ...] = ()) -> pd.DataFrame:
    meta = MetaData.load(meta_dir)
    out = Path(out_dir)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    mf = meta.mf
    if drop_groups:
        unknown = set(drop_groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown meta-feature groups {sorted(unknown)}; choose from {sorted(GROUPS)}")
        mf = mf.drop(columns=[f for g in drop_groups for f in GROUPS[g]])
        log.info("ablation: dropped %s, %d meta-features left", list(drop_groups), mf.shape[1])

    n_failed = int(meta.perf_raw.isna().sum().sum())
    log.info("%d tasks from %d parents, %d pipelines, %d failed evaluations filled",
             len(meta.perf), meta.groups.nunique(), len(meta.pipelines), n_failed)

    # Sanity check (Section 4.6): no family should win on more than about half the tasks.
    wins = family_win_share(meta)
    wins.rename("win_share").to_csv(out / "family_wins.csv", index_label="family")
    if wins.iloc[0] > DOMINANCE_LIMIT:
        log.warning("family %s is best on %.0f%% of tasks; the dataset pool may need more diversity",
                    wins.index[0], 100 * wins.iloc[0])

    folds = make_folds(meta, cfg.outer_folds, cfg.seed, Path(meta_dir) / "folds.csv")

    selectors = default_selectors(cfg.seed, cfg.knn_k)
    oof = {s.name: [] for s in selectors}
    gains = []
    for k in sorted(folds.unique()):
        tr, te = folds.index[folds != k], folds.index[folds == k]
        for sel in selectors:
            sel.fit(mf.loc[tr], meta.perf.loc[tr])
            oof[sel.name].append(sel.score(mf.loc[te]))
            if isinstance(sel, LGBMGapRegressor):
                gains.append(sel.feature_gain())

    rows, per_task, curves = [], {}, {}
    for name, parts in oof.items():
        scores = pd.concat(parts).loc[meta.perf.index]
        scores.to_csv(out / "scores" / f"{_slug(name)}.csv")
        s = metrics.summarise(scores, meta.perf, cfg.bootstrap, cfg.seed, cfg.hit_tolerance)
        rows.append({"selector": name, **s})
        per_task[name] = metrics.regret_at_k(scores, meta.perf, 1)
        curves[name] = metrics.regret_curve(scores, meta.perf).mean()

    summary = pd.DataFrame(rows).set_index("selector")
    summary.to_csv(out / "summary.csv", float_format="%.4f")
    pd.DataFrame(per_task).to_csv(out / "per_task_regret1.csv", float_format="%.4f")
    pd.DataFrame(curves).rename_axis("n_evaluated").to_csv(out / "regret_curve.csv", float_format="%.4f")
    if gains:
        gain = pd.concat(gains, axis=1).mean(axis=1)
        gain.sort_values(ascending=False).rename("gain").to_csv(out / "feature_gain.csv", index_label="feature")
    return summary


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def format_summary(summary: pd.DataFrame) -> str:
    lines = [f"{'Selector':<22} {'Regret@1 (95% CI)':<22} {'Regret@3 (95% CI)':<22} {'Top-1 hit':>9} {'Spearman':>9}"]
    for name, r in summary.iterrows():
        lines.append(
            f"{name:<22} {r['regret@1']:5.2f} ({r['regret@1_lo']:.2f} to {r['regret@1_hi']:.2f})  "
            f"{r['regret@3']:5.2f} ({r['regret@3_lo']:.2f} to {r['regret@3_hi']:.2f})  "
            f"{r['top1_hit']:8.1f}% {r['spearman']:9.2f}"
        )
    return "\n".join(lines)
