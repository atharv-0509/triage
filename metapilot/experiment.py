"""Phase 1 experiment: grouped outer CV over all selectors (report Section 5).

Writes to ``out_dir``:

* ``summary.csv``            - Table 6 (regret@1/@3 with 95% CIs, top-1 hit, Spearman,
                               mean seconds to run the top-1/top-3 recommendations,
                               and a FLAML row when ``<meta_dir>/flaml.csv`` exists)
                               LLM rows (``LLM: <variant>``) come from
                               ``<meta_dir>/llm/predictions/*.jsonl``, with
                               ``llm_coverage``: the share of pipelines the model
                               named itself (the rest follow single best)
* ``summary_real.csv``       - the same metrics on real (non-synthetic) tasks only
* ``summary_seeds.csv``      - mean and std of each metric over seeds (``--seeds``)
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

from . import llm, metrics
from .config import Config
from .metafeatures import GROUPS
from .metalevel import MetaData, family_win_share, folds_path, make_folds
from .selectors import LGBMGapRegressor, default_selectors

log = logging.getLogger(__name__)

DOMINANCE_LIMIT = 0.5


def run(meta_dir: str | Path, out_dir: str | Path, cfg: Config, drop_groups: tuple[str, ...] = (),
        extra_seeds: tuple[int, ...] = ()) -> pd.DataFrame:
    """Evaluate all selectors. ``extra_seeds`` repeats the grouped outer CV
    with other fold assignments and selector seeds (robustness check)."""
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

    seeds = list(dict.fromkeys([cfg.seed, *extra_seeds]))  # primary seed first, no repeats
    per_seed = []
    for seed in seeds:
        folds = make_folds(meta, cfg.outer_folds, seed, folds_path(meta_dir, seed, cfg.seed))
        oof, gains = cross_validate(meta, mf, folds, seed, cfg.knn_k)
        rows = [{"selector": name, "seed": seed,
                 **metrics.summarise(scores, meta.perf, cfg.bootstrap, seed, cfg.hit_tolerance)}
                for name, scores in oof.items()]
        per_seed.append(pd.DataFrame(rows))
        if seed == cfg.seed:
            primary, primary_gains, primary_folds = oof, gains, folds

    # Phase 2: LLM outputs written by `metapilot llm-predict`, scored on the same split.
    coverage = {}
    for variant, path in llm.prediction_files(meta_dir).items():
        scores, cov = llm.llm_scores(llm.load_jsonl(path), meta, primary_folds)
        if scores.empty:
            continue
        primary[f"LLM: {variant}"] = scores
        coverage[f"LLM: {variant}"] = cov
        if cov.mean() < 0.9:
            log.warning("LLM %s named only %.0f%% of pipelines on average; the rest come from the single-best "
                        "order, so its metrics are partly single best's", variant, 100 * cov.mean())

    # Detailed outputs use the primary seed (the shared folds.csv split).
    seed_rows = per_seed[0].set_index("selector")
    rows, per_task, curves = [], {}, {}
    for name, scores in primary.items():
        scores.to_csv(out / "scores" / f"{_slug(name)}.csv")
        if name in seed_rows.index:
            s = seed_rows.loc[name].drop("seed").to_dict()
        else:
            s = metrics.summarise(scores, meta.perf, cfg.bootstrap, cfg.seed, cfg.hit_tolerance)
        if meta.secs is not None:
            s["cost@1_s"] = metrics.cost_at_k(scores, meta.secs, 1).mean()
            s["cost@3_s"] = metrics.cost_at_k(scores, meta.secs, 3).mean()
        if name in coverage:
            s["llm_coverage"] = coverage[name].mean()
            s["llm_fully_parsed"] = float((coverage[name] == 1).mean())
        s.update(metrics.diversity(scores))
        rows.append({"selector": name, **s})
        per_task[name] = metrics.regret_at_k(scores, meta.perf, 1)
        curves[name] = metrics.regret_curve(scores, meta.perf).mean()

    flaml = flaml_row(meta, Path(meta_dir) / "flaml.csv", cfg)
    if flaml is not None:
        rows.append(flaml)

    summary = pd.DataFrame(rows).set_index("selector")
    summary.to_csv(out / "summary.csv", float_format="%.4f")

    # Research question 3 compares methods on real data, where column names and
    # descriptions mean something.
    real = meta.tasks.index[~meta.tasks["source"].isin(llm.SYNTHETIC_SOURCES)]
    if len(real):
        real_rows = []
        for name, scores in primary.items():
            sub = scores.loc[scores.index.intersection(real)]
            if len(sub):
                s = metrics.summarise(sub, meta.perf, cfg.bootstrap, cfg.seed, cfg.hit_tolerance)
                if name in coverage:
                    s["llm_coverage"] = coverage[name].loc[sub.index].mean()
                real_rows.append({"selector": name, **s})
        flaml_real = flaml_row(meta, Path(meta_dir) / "flaml.csv", cfg, tasks=real)
        if flaml_real is not None:
            real_rows.append(flaml_real)
        pd.DataFrame(real_rows).set_index("selector").to_csv(out / "summary_real.csv", float_format="%.4f")
        log.info("real-data subset: %d tasks from %d parents -> summary_real.csv",
                 len(real), meta.tasks.loc[real, "parent"].nunique())
    pd.DataFrame(per_task).to_csv(out / "per_task_regret1.csv", float_format="%.4f")
    pd.DataFrame(curves).rename_axis("n_evaluated").to_csv(out / "regret_curve.csv", float_format="%.4f")
    gain = pd.concat(primary_gains, axis=1).mean(axis=1)
    gain.sort_values(ascending=False).rename("gain").to_csv(out / "feature_gain.csv", index_label="feature")

    if len(seeds) > 1:
        all_seeds = pd.concat(per_seed)
        all_seeds.to_csv(out / "summary_by_seed.csv", index=False, float_format="%.4f")
        cols = ["regret@1", "regret@3", "top1_hit", "spearman"]
        agg = all_seeds.groupby("selector", sort=False)[cols].agg(["mean", "std"])
        agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
        agg.to_csv(out / "summary_seeds.csv", float_format="%.4f")
        log.info("summary over seeds %s written to summary_seeds.csv", seeds)
    return summary


def cross_validate(meta: MetaData, mf: pd.DataFrame, folds: pd.Series, seed: int, knn_k: int):
    """Out-of-fold scores of every selector, plus LightGBM regressor gains."""
    selectors = default_selectors(seed, knn_k)
    oof = {s.name: [] for s in selectors}
    gains = []
    for k in sorted(folds.unique()):
        tr, te = folds.index[folds != k], folds.index[folds == k]
        for sel in selectors:
            sel.fit(mf.loc[tr], meta.perf.loc[tr])
            oof[sel.name].append(sel.score(mf.loc[te]))
            if isinstance(sel, LGBMGapRegressor):
                gains.append(sel.feature_gain())
    return {name: pd.concat(parts).loc[meta.perf.index] for name, parts in oof.items()}, gains


def flaml_row(meta: MetaData, path: Path, cfg: Config, tasks: pd.Index | None = None) -> dict | None:
    """Summary row for the FLAML baseline, if ``metapilot flaml`` has been run.
    Regret is against the best portfolio pipeline and can be negative."""
    if not path.exists():
        return None
    f = pd.read_csv(path, index_col="task_id")
    f = f[f["accuracy"].notna() & f.index.isin(meta.perf.index if tasks is None else tasks)]
    if f.empty:
        return None
    regret = 100 * (meta.perf.loc[f.index].max(axis=1) - f["accuracy"])
    lo, hi = metrics.bootstrap_ci(regret, cfg.bootstrap, cfg.seed)
    budget = f["budget_per_fold"].iloc[0]
    return {"selector": f"FLAML ({budget:g}s/fold)", "regret@1": regret.mean(), "regret@1_lo": lo, "regret@1_hi": hi,
            "top1_hit": 100 * float((regret <= cfg.hit_tolerance + 1e-9).mean()), "n_tasks": len(f),
            "cost@1_s": f["seconds"].mean()}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def format_summary(summary: pd.DataFrame) -> str:
    def ci(r, m):
        if pd.isna(r.get(m)):
            return f"{'-':<22}"
        return f"{r[m]:5.2f} ({r[m + '_lo']:.2f} to {r[m + '_hi']:.2f})".ljust(22)

    def num(v, fmt):
        return format(v, fmt) if pd.notna(v) else "-"

    has_cost = "cost@3_s" in summary
    lines = [f"{'Selector':<24} {'Regret@1 (95% CI)':<22} {'Regret@3 (95% CI)':<22} {'Top-1 hit':>9} {'Spearman':>9}"
             + (f" {'Cost@1 s':>9} {'Cost@3 s':>9}" if has_cost else "")]
    for name, r in summary.iterrows():
        line = (f"{name:<24} {ci(r, 'regret@1')} {ci(r, 'regret@3')} {num(r['top1_hit'], '8.1f')}% "
                f"{num(r.get('spearman'), '9.2f'):>9}")
        if has_cost:
            line += f" {num(r.get('cost@1_s'), '9.2f'):>9} {num(r.get('cost@3_s'), '9.2f'):>9}"
        if pd.notna(r.get("llm_coverage")):
            line += f"   (LLM named {100 * r['llm_coverage']:.0f}% of pipelines)"
        lines.append(line)
    return "\n".join(lines)
