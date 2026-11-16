# Phase 2 pilot run (Colab T4, 16 November 2026)

Qwen2.5-1.5B-Instruct, 4-bit QLoRA (r = 16), 3 epochs, one adapter per outer fold and prompt kind, run with
`notebooks/phase2_qlora.ipynb` on the pilot meta-dataset (120 tasks, 18 pipelines). About 40 minutes on one T4,
3.5–4 minutes per fold.

* `summary.csv`, `summary_real.csv`, `per_task_regret1.csv`, `regret_curve.csv`, `family_wins.csv`,
  `feature_gain.csv`: written by `metapilot evaluate` in Colab.
* `predictions/*.jsonl`: raw model outputs (`base_numeric`, `ft_numeric`, `ft_semantic`).
* `diversity.csv`: how many different rankings each variant produced (computed afterwards from the predictions).

**Provenance:** Colab built its own copy of the pilot meta-dataset (Python 3.13, newer library versions), so its
performance matrix differs slightly from this repository's `results/pilot/`. For example, the LightGBM regressor's
Regret@1 is 3.06 here against 2.80 there. All rows in these tables were scored on the same Colab build and are
comparable with each other, but not digit for digit with `results/pilot/`.

**Main caveat:** the fine-tuned numeric model collapsed to an almost fixed ranking: 11 distinct rankings for 120
tasks, and one ranking for all 24 tasks of fold 0. See `diversity.csv`.
