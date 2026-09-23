# MetaPilot

Meta-learned and LLM-based pipeline selection for tabular machine learning.
This is the code for the Intelligent Model Design Lab project. The design is described in the Milestone 2 report.

We treat pipeline selection as a learning problem. First, a fixed portfolio of pipelines is run on many
classification tasks. Then a meta-model learns, from dataset meta-features, which pipeline will do best on a
dataset it has not seen.

* **Phase 1** (implemented here): kNN and LightGBM meta-models trained on 14 numeric meta-features.
* **Phase 2** (planned): QLoRA fine-tuned Qwen2.5-1.5B-Instruct that ranks pipelines from a text description of
  the dataset. The text metadata it needs is already collected by the Phase 1 build step (`text_meta.jsonl`).

## Quick start

```bash
pip install -e ".[dev]"              # add ,openml for the full study
python -m metapilot pilot --config configs/pilot.yaml
```

This builds the pilot meta-dataset in `data/pilot/`, evaluates all selectors in `runs/pilot/` and draws the
report figures in `runs/pilot/figures/`. It takes about 2 minutes on 4 cores.

You can also run each step on its own:

```bash
python -m metapilot build    --config configs/pilot.yaml --out data/pilot
python -m metapilot evaluate --config configs/pilot.yaml --meta data/pilot --out runs/pilot
python -m metapilot evaluate --config configs/pilot.yaml --meta data/pilot --out runs/pilot_nolm \
                             --drop-group landmarkers          # landmarker ablation (Section 5.4)
python -m metapilot plots    --results runs/pilot
python -m pytest
```

## Code layout

| Module | Report section | What it does |
|---|---|---|
| `metapilot/config.py` | – | YAML config. The defaults are the pilot settings |
| `metapilot/sources.py` | 2.1, 2.4 | Parent datasets: scikit-learn built-ins, `make_classification`, moons/circles, OpenML-CC18 plus supplementary datasets |
| `metapilot/preprocess.py` | 3.1 | Dataset cleaning, stratified row cap, and an in-pipeline `ColumnTransformer` (imputation, one-hot/frequency encoding, scaling) |
| `metapilot/augment.py` | 3.2 | Subsamples 30–100% of rows and 50–100% of columns, and rejects tasks with a class under 6 samples |
| `metapilot/portfolio.py` | 2.3 | The 18-pipeline pilot portfolio (Table 2), the ~38-pipeline full portfolio and optional TabPFN |
| `metapilot/metafeatures.py` | 3.3 | The 14 meta-features (simple, statistical, information-theoretic, landmarkers) |
| `metapilot/metadataset.py` | 2.2 | Runs every pipeline on every task with stratified CV and a per-fit time limit, then writes the meta-dataset |
| `metapilot/metalevel.py` | 3.4, 3.5 | Missing-value fill, gap/relevance targets, long format, grouped folds, family win share |
| `metapilot/selectors.py` | 4.3, 4.5 | Random, single best, kNN meta-learner, LightGBM regressor, LightGBM LambdaRank |
| `metapilot/metrics.py` | 4.2, 4.6 | Regret@k, top-1 hit, Spearman, bootstrap CIs, regret curve |
| `metapilot/experiment.py` | 5 | Grouped outer CV over all selectors, dominance check, result tables |
| `metapilot/plots.py` | 5 | Figures 2–5 |

## Outputs

`build` writes the **meta-dataset**, which every method uses, including the Phase 2 LLM:

| File | Content |
|---|---|
| `performance.csv` | Mean CV accuracy, one row per task and one column per pipeline. A blank cell means the fit failed or timed out |
| `metafeatures.csv` | 14 meta-features per task |
| `tasks.csv` | Parent, source, shape, class count and missing rate per task |
| `text_meta.jsonl` | Column names, column types and description per task, for the Phase 2 prompts |
| `fit_seconds.csv`, `portfolio.csv`, `config.yaml` | Timing, pipeline families and the exact config used |
| `folds.csv` | Outer fold per task, grouped by parent. It is written on the first `evaluate` and reused afterwards so all methods share the same splits |

`evaluate` writes `summary.csv` (Table 6), `per_task_regret1.csv`, `regret_curve.csv`, `family_wins.csv`,
`feature_gain.csv` and out-of-fold scores for each selector in `scores/`.

## Current pilot results

Setup: 30 parents, 120 tasks, 18 pipelines, 3-fold CV (6,480 fits) and 5-fold grouped outer CV. Regret is measured
in accuracy points.

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman |
|---|---|---|---|---|
| Random | 6.41 (5.23–7.82) | 2.04 (1.59–2.49) | 15.0% | −0.03 |
| Single best | 2.87 (2.40–3.42) | 1.39 (1.02–1.84) | 27.5% | 0.38 |
| kNN meta-learner | 3.16 (2.65–3.74) | 1.54 (1.17–1.97) | 22.5% | 0.33 |
| LightGBM regressor | 3.07 (2.48–3.71) | 1.29 (0.99–1.64) | 21.7% | 0.43 |
| LightGBM LambdaRank | 3.40 (2.84–3.99) | 1.23 (0.92–1.58) | 18.3% | 0.39 |

These numbers come from this codebase. They are **not** the numbers in the Milestone 2 report, because the
synthetic parents are generated differently. In this run the single-best default is harder to beat at Regret@1
than it was in the report. Only 30 parents are used, 80% of them synthetic, so the confidence intervals overlap
heavily. Section 5.5 of the report already names this as a pilot limitation, and scaling up to OpenML is the next
step. The other findings match the report: the SVM is the most frequent winner (18%), no family dominates, and
landmarkers plus dataset size carry most of the LightGBM gain.

## Implementation notes

* **No leakage from preprocessing.** Imputation, encoding and scaling live inside each sklearn pipeline, so they
  are fitted on the training folds only. At the meta level, the kNN scaler and LightGBM are fitted on the training
  tasks of each outer fold only.
* **Failed fits.** A failing or timed-out fit is stored as blank in `performance.csv` and replaced by the task's
  lowest score at load time. The time limit uses `SIGALRM`, and fits running inside compiled code (for example
  libsvm) are also marked as timed out once they return late.
* **Family win share.** When several families tie for the best accuracy on a task, the win is split between them.
* **`pca_frac_95`** is the number of principal components needed for 95% of the variance, divided by the number of
  components (min(rows, columns)), computed on standardised features.
* **OpenML** loading (`sources: [openml, ...]`, see `configs/full.yaml`) could not be tested in the development
  sandbox because openml.org was unreachable. Please run it once on a machine with network access before the full
  build.

## Roadmap (report Table 7)

- [x] Phase 1 framework: data sources, augmentation, portfolio, meta-features, performance matrix, meta-models,
      evaluation, figures
- [ ] Scale up: OpenML-CC18 plus supplementary datasets (~150 parents), 10 subsamples each, full portfolio plus
      TabPFN
- [ ] Phase 1 completion: FLAML baseline, landmarker ablation at full scale, multiple seeds
- [ ] Phase 2: prompt serialisation (numeric-only and semantic versions), base-LLM evaluation, QLoRA fine-tuning
- [ ] Phase 2 evaluation on the same folds and metrics
- [ ] Gradio demo and final report
