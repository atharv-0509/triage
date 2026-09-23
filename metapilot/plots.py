"""Figures 2-5 of the report, drawn from the files written by
:mod:`metapilot.experiment`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metafeatures import GROUPS  # noqa: E402

COLOR = "#3b6ea5"
MUTED = "#9aa5b1"


def make_all(results_dir: str | Path) -> list[Path]:
    d = Path(results_dir)
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    return [
        family_wins(d, fig_dir / "fig2_family_wins.png"),
        regret_ci(d, fig_dir / "fig3_regret1_ci.png"),
        regret_curve(d, fig_dir / "fig4_regret_curve.png"),
        feature_gain(d, fig_dir / "fig5_feature_gain.png"),
    ]


def family_wins(d: Path, path: Path) -> Path:
    wins = pd.read_csv(d / "family_wins.csv", index_col="family")["win_share"].sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(wins.index, 100 * wins.to_numpy(), color=COLOR)
    for y, v in enumerate(wins.to_numpy()):
        ax.text(100 * v + 0.5, y, f"{100 * v:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("Tasks where the family is best (%)")
    _finish(fig, ax, path)
    return path


def regret_ci(d: Path, path: Path) -> Path:
    s = pd.read_csv(d / "summary.csv", index_col="selector")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    y = range(len(s))[::-1]
    err = [s["regret@1"] - s["regret@1_lo"], s["regret@1_hi"] - s["regret@1"]]
    ax.errorbar(s["regret@1"], list(y), xerr=err, fmt="o", color=COLOR, ecolor=MUTED, capsize=3)
    ax.set_yticks(list(y), s.index)
    ax.set_xlabel("Regret@1 (accuracy points, 95% CI)")
    _finish(fig, ax, path)
    return path


def regret_curve(d: Path, path: Path) -> Path:
    c = pd.read_csv(d / "regret_curve.csv", index_col="n_evaluated")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name in c.columns:
        ax.plot(c.index, c[name], marker="o", markersize=3, label=name)
    ax.set_xlabel("Pipelines evaluated in recommended order")
    ax.set_ylabel("Mean regret (accuracy points)")
    ax.set_xlim(1, min(10, c.index.max()))
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, path)
    return path


def feature_gain(d: Path, path: Path, top: int = 10) -> Path:
    g = pd.read_csv(d / "feature_gain.csv", index_col="feature")["gain"]
    g = g[~g.index.str.startswith("pipe")]  # meta-features only
    share = (g / g.sum()).sort_values(ascending=False).head(top)[::-1]
    group_of = {f: grp for grp, fs in GROUPS.items() for f in fs}
    palette = {"simple": "#8fb3d9", "statistical": "#5b8fc7", "info_theoretic": "#9aa5b1", "landmarkers": COLOR}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(share.index, 100 * share.to_numpy(), color=[palette[group_of.get(f, "simple")] for f in share.index])
    ax.set_xlabel("Share of LightGBM gain over meta-features (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), frameon=False, fontsize=8, loc="lower right")
    _finish(fig, ax, path)
    return path


def _finish(fig, ax, path: Path) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
