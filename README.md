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
python -m metapilot evaluate --config configs/pilot.yaml --meta data/pilot --out runs/pilot_seeds \
                             --seeds 1 2 3 4                   # repeated outer CV, see summary_seeds.csv
python -m metapilot plots    --results runs/pilot
python -m pytest
```

## Full study

The full study (`configs/full.yaml`) uses OpenML-CC18, the supplementary OpenML datasets, wine, iris, 40
synthetic parents and 10 moons/circles parents. Each parent gets 10 subsamples, and the 38-pipeline portfolio is
run on tasks capped at 2,000 rows. That comes to roughly 170,000 model fits. Timing the portfolio on 2,000-row
tasks gives 30–60 s for a typical task and a few minutes for wide ones, so budget **about 25–40 CPU-hours**.
Split the work across group members:

```bash
pip install -e ".[openml,flaml]"

# On machine i of 4 (i = 0, 1, 2, 3). Everyone uses the same config.
python -m metapilot build --config configs/full.yaml --out data/full --shard i/4
```

Every finished task is saved straight away to `data/full/parts/<task>.json`. If a build is interrupted, run the
same command again and it continues where it stopped. When all shards are done, copy every machine's `parts/`
folder into one `data/full/parts/` and run:

```bash
python -m metapilot assemble --out data/full
python -m metapilot evaluate --config configs/full.yaml --meta data/full --out runs/full --seeds 1 2 3 4
python -m metapilot evaluate --config configs/full.yaml --meta data/full --out runs/full_nolm --drop-group landmarkers
```

Every machine derives the task list from the config alone. Each parent gets its own random seed based on its
name, so the shards line up without any coordination. A part built with different settings is rejected by
`assemble`. Parents that look like copies of each other (same shape and class counts) are dropped, because copies
in different outer folds would leak. This is also why `full.yaml` leaves out scikit-learn's breast cancer and
digits datasets: CC18 already contains them as wdbc and optdigits.

**FLAML baseline.** FLAML runs on the same tasks and CV folds with a fixed time budget per fold. `evaluate` then
adds a FLAML row to `summary.csv`. For an equal-budget comparison, set `--budget` to about `cost@3_s / 3` of the
meta-models, which `evaluate` reports. `--shard` and resume work the same way as for `build`, and `--max-tasks`
runs FLAML on an evenly spread subset.

```bash
python -m metapilot flaml --config configs/full.yaml --meta data/full --budget 10 --shard i/4
python -m metapilot flaml --config configs/full.yaml --meta data/full --budget 10 --max-tasks 300   # cheaper
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
| `metapilot/metadataset.py` | 2.2 | Runs every pipeline on every task with stratified CV and a per-fit time limit, with per-task checkpoints, sharding and assembly, then writes the meta-dataset |
| `metapilot/baselines.py` | 4.5 | FLAML AutoML baseline on the same tasks and folds |
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

## Results so far

Snapshots of the result tables and figures are kept in `results/`. Regret is measured in accuracy points (lower is
better), using 5-fold grouped outer CV.

### Scale test: full-study settings on local sources (`configs/local_full.yaml`)

This run uses 54 parents (4 scikit-learn, 40 synthetic, 10 moons/circles) with 10 subsamples each, giving 540 tasks.
The 38-pipeline portfolio and the 2,000-row cap make 61,560 fits, which took 21 minutes on 4 cores. The table uses
the shared `folds.csv` split. The ± values are the mean and standard deviation over 5 outer-CV seeds.

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman | Regret@1 over 5 seeds | Regret@3 over 5 seeds |
|---|---|---|---|---|---|---|
| Random | 6.88 (6.23–7.62) | 3.05 (2.71–3.40) | 13.3% | 0.00 | 7.20 ± 0.45 | 3.02 ± 0.10 |
| Single best | 2.94 (2.66–3.21) | 2.24 (1.99–2.49) | 25.2% | 0.36 | 2.94 ± 0.00 | 2.24 ± 0.00 |
| kNN meta-learner | 3.15 (2.85–3.48) | 1.92 (1.70–2.15) | 23.9% | 0.47 | 3.21 ± 0.15 | 1.93 ± 0.07 |
| LightGBM regressor | 2.91 (2.62–3.20) | 1.90 (1.66–2.14) | 26.9% | 0.48 | 3.02 ± 0.11 | 1.96 ± 0.05 |
| LightGBM LambdaRank | **2.76** (2.49–3.06) | **1.57** (1.38–1.78) | 25.6% | 0.44 | **2.72 ± 0.08** | **1.58 ± 0.04** |

What changed compared with the 120-task pilot:

* **LambdaRank now performs best.** In the pilot it was the weakest meta-model, and the report put this down to too
  few training tasks. With about 430 training tasks per fold it has the lowest regret at both cutoffs.
* **The meta-models now clearly beat single best at Regret@3** (1.57–1.92 against 2.24), where in the pilot they
  did not. At Regret@1 the gain is small (2.72 against 2.94 averaged over seeds) and the CIs overlap.
* **Landmarker ablation.** Removing the landmarkers makes every meta-model worse: LambdaRank goes from 2.76 to 3.16
  at Regret@1 and from 1.57 to 1.73 at Regret@3, and kNN goes from 3.15 to 3.61. Landmarkers carry 56% of the
  LightGBM gain (`summary_no_landmarkers.csv`).
* **No single family dominates.** The most frequent winners are extra trees (17%), MLP (15%) and SVM (15%).
* **FLAML** (5 s per fold, 60 evenly spread tasks, about 15 s per task) has a Regret@1 of 5.75 (4.41–7.16). On the
  same 60 tasks LambdaRank scores 2.46 at Regret@1 and 1.65 at Regret@3, using about 1.8 s to run its top 3. There
  are two caveats. FLAML's search space has no SVM, kNN or MLP, which often win on this synthetic-heavy pool. And
  regret against the best of 38 cross-validated scores is optimistic about the oracle, which penalises any single
  method.

About 85% of these parents are synthetic, so treat this run as a test that the full-scale pipeline works and as a
first sign of trends. It is not a finding about real data yet. The OpenML build is what will settle it.

### Pilot (`configs/pilot.yaml`)

This run uses 30 parents and 120 tasks, with 18 pipelines and 3-fold CV (6,480 fits).

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman |
|---|---|---|---|---|
| Random | 5.28 (4.29–6.54) | 2.19 (1.72–2.72) | 13.3% | −0.05 |
| Single best | 2.54 (2.07–3.06) | 1.72 (1.35–2.10) | 29.2% | 0.36 |
| kNN meta-learner | 3.39 (2.69–4.23) | 1.48 (1.10–1.91) | 21.7% | 0.36 |
| LightGBM regressor | 2.80 (2.28–3.40) | 1.33 (1.01–1.72) | 25.0% | 0.39 |
| LightGBM LambdaRank | 2.93 (2.37–3.56) | 1.43 (1.06–1.86) | 24.2% | 0.34 |

These pilot numbers differ from the Milestone 2 report because the synthetic parents are generated differently. At
this size the CIs overlap too much to separate the selectors.

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
- [x] Scale-up tooling: sharded and resumable build, full portfolio, 10 subsamples, duplicate check; tested at 540
      tasks on local sources
- [x] FLAML baseline, landmarker ablation and multi-seed evaluation (implemented and run on the local scale test)
- [ ] Full build on OpenML-CC18 plus supplementary datasets (~150 parents); needs openml.org network access
- [ ] TabPFN in the portfolio (`portfolio: full_tabpfn`)
- [ ] Phase 2: prompt serialisation (numeric-only and semantic versions), base-LLM evaluation, QLoRA fine-tuning
- [ ] Phase 2 evaluation on the same folds and metrics
- [ ] Gradio demo and final report
