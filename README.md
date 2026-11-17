# MetaPilot

Meta-learned and LLM-based pipeline selection for tabular machine learning.
This is the code for the Intelligent Model Design Lab project. The design is described in the Milestone 2 report.

We treat pipeline selection as a learning problem. First, a fixed portfolio of pipelines is run on many
classification tasks. Then a meta-model learns, from dataset meta-features, which pipeline will do best on a
dataset it has not seen.

* **Phase 1**: kNN and LightGBM meta-models trained on 14 numeric meta-features. Implemented and run.
* **Phase 2**: QLoRA fine-tuned Qwen2.5-1.5B-Instruct that ranks pipelines from a text profile of the dataset.
  Prompts, training, generation and scoring are implemented and tested end to end with a tiny offline model; the
  real fine-tuning run needs a GPU (`notebooks/phase2_qlora.ipynb`) and **has not been run yet**.
* **Demo**: a Gradio app that ranks the portfolio for an uploaded CSV (`app/demo.py`).

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
run on tasks capped at 2,000 rows. That comes to 1,310 tasks and 149,340 model fits. Timing the portfolio on 2,000-row
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

## Phase 2: LLM selector

```bash
python -m metapilot prompts --config configs/pilot.yaml --meta data/pilot     # writes data/pilot/llm/prompts.jsonl
```

Every task gets two prompts. **Prompt A (numeric)** holds the task's shape and all 14 meta-features, the same
information LightGBM gets (research question 2). **Prompt B (semantic)** adds the column names and the dataset
description (research question 3). The target is the full portfolio ordered by measured accuracy. Each record
carries its outer fold from `folds.csv`, the split every method shares.

Fine-tuning and generation run on a GPU. The notebook `notebooks/phase2_qlora.ipynb` runs the whole loop on a
free T4, on Colab or Kaggle (it detects which), or on any machine with a GPU.

**Colab**
1. File > Open notebook > GitHub, enter `atharv-0509/triage` (branch `main`), then pick
   `notebooks/phase2_qlora.ipynb`.
2. Runtime > Change runtime type > **T4 GPU**, then Runtime > Run all, and allow Google Drive access.
3. Work is saved to `MyDrive/metapilot_phase2/<RUN_NAME>/` after every fold. After a disconnect, run all again;
   finished folds are skipped.

**Kaggle** (30 GPU-hours a week, runs of up to 12 hours)
1. Create a notebook and use File > Import Notebook with the notebook's GitHub URL, or upload the `.ipynb`.
2. In the settings panel set Accelerator to **GPU T4 x2** (the notebook uses one of the two) and turn **Internet**
   on. Internet access needs a phone-verified Kaggle account.
3. Save Version > **Save & Run All (Commit)**. The run continues without the browser open, and
   `/kaggle/working/metapilot_phase2/<RUN_NAME>/` becomes the version's output.
4. To resume an unfinished run, add that version's output as an input (Add Input > Your Work) and commit again.
   Finished folds are copied in and skipped.

On either platform, check the coverage the notebook prints after the base model and after the first fine-tuned
fold. If it is far below 100%, stop: the model is not producing rankings. By default the notebook builds the pilot
meta-dataset itself. For the full OpenML study, set `CONFIG = "configs/full.yaml"`, `META = "metadata/full"` and
`RUN_NAME = "full"` in the first cell; `metadata/full/` (1,310 tasks, 12 MB) is in the repository. These are now the notebook's defaults,
with `EPOCHS = 1` to keep the full run to roughly 5–6 hours on a T4 (an estimate from the pilot's timings); the
pilot settings are given in a comment in that cell. The notebook was run end to end on CPU
with a tiny stand-in model; it has not yet run on a T4 with Qwen.

The same steps from the command line:

```bash
pip install -e ".[llm]" bitsandbytes
python -m metapilot llm-predict --prompts P --kind numeric --out data/pilot/llm/predictions/base_numeric.jsonl
for k in 0 1 2 3 4; do
  python -m metapilot llm-train   --prompts P --fold $k --kind numeric --out adapters/numeric_f$k
  python -m metapilot llm-predict --prompts P --fold $k --kind numeric --adapter adapters/numeric_f$k \
                                  --out data/pilot/llm/predictions/ft_numeric.jsonl
done                                   # and the same with --kind semantic -> ft_semantic.jsonl
python -m metapilot evaluate --config configs/pilot.yaml --meta data/pilot --out runs/pilot
```

(`P` is `data/pilot/llm/prompts.jsonl`.) With `metapilot prompts --shuffle-candidates`, every task lists the
candidates in its own random order (stable per task, and the same in Prompts A and B), so a model cannot learn to
echo the listed order. The setting is stored in each prompt record and in each adapter's `train_info.json`, and
`llm-predict` refuses to mix the two. The notebook turns it on with `SHUFFLE_CANDIDATES = True` (the default since
the pilot run) and keeps shuffled runs apart: Drive folder `<RUN_NAME>_shuf`, predictions `ft_numeric_shuf`, and
so on. Shuffling removes one shortcut; it does not by itself make the model use the profile. Check
`distinct_rankings` in the results. Each adapter trains only on tasks outside its fold and refuses to predict
any other fold. `evaluate` adds one `LLM: <variant>` row per predictions file. A model's output is matched to
pipeline names exactly; pipelines it leaves out are filled in from the training folds' single-best order so the
ranking is complete. Because a model that names nothing would then score exactly like single best, every LLM row
reports **`llm_coverage`**, the share of pipelines the model named itself, and `evaluate` warns below 90%.
`summary_real.csv` repeats all metrics on real (non-synthetic) tasks only, which is the comparison research
question 3 needs. Synthetic descriptions state their generator settings, so Prompt B should not be compared with
Prompt A on synthetic tasks.

## Demo

```bash
pip install -e ".[demo]"
python app/demo.py --meta data/pilot          # http://127.0.0.1:7860
python -m metapilot recommend --meta data/pilot --csv my.csv --target label --verify 3   # same, in the terminal
```

Upload a CSV and choose the target column. The app ranks the portfolio with LightGBM LambdaRank trained on every
task in the meta-dataset. It also lists the most similar training tasks and the pipeline that won on each, and can
run the top three pipelines on the data with cross-validation so the ranking can be checked against real
accuracies.

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
| `metapilot/llm.py` | 3.6, 4.4 | Prompt A/B serialisation, ranking parser, scoring of LLM outputs with coverage |
| `metapilot/llm_train.py` | 4.4 | QLoRA fine-tuning per outer fold and batched generation (GPU; `llm` extra) |
| `metapilot/recommend.py` | 6 | Recommendations for a new dataset, with similar tasks and optional verification |
| `app/demo.py` | 6 | Gradio demo |
| `notebooks/phase2_qlora.ipynb` | 4.4 | Notebook for the Phase 2 runs (Colab, Kaggle or any GPU) |

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

### Full study (OpenML) (`configs/full.yaml`, `results/full/`)

131 parents: all 72 OpenML-CC18 datasets, 7 of the 8 supplementary OpenML datasets (madelon is already in CC18),
wine, iris, 40 synthetic and 10 moons/circles parents. Each parent gets 10 subsamples, which gives 1,310 tasks
(810 of them from the 81 real parents). The 38-pipeline portfolio is run with 3-fold CV on tasks capped at 2,000
rows, 149,340 fits in all, taking about 5 hours of wall time on 4 cores (about 20 CPU-hours). No task failed. 75 of the 49,780 task-pipeline cells (0.15%) have no result and are filled with the task's lowest score: `knn_30_dist` cannot run on 21 tiny synthetic tasks with fewer than 30 training rows per fold, and the HistGradientBoosting variants and `rf_300_mf0.5` hit the 120 s per-fit limit on the two widest image datasets (CIFAR-10 and Devnagari-Script, about 1,000–2,800 columns). The
meta-dataset is exported to `metadata/full/` (12 MB) for Phase 2.

The table uses the shared `folds.csv` split, with 95% bootstrap CIs. The ± columns are the mean and standard
deviation over 5 outer-CV seeds.

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman | Regret@1 over 5 seeds | Regret@3 over 5 seeds |
|---|---|---|---|---|---|---|
| Random | 6.62 (6.15–7.07) | 2.69 (2.50–2.89) | 11.4% | 0.00 | 6.65 ± 0.51 | 2.62 ± 0.08 |
| Single best | 2.64 (2.48–2.80) | 1.96 (1.82–2.09) | 26.5% | 0.45 | 2.64 ± 0.00 | 1.95 ± 0.01 |
| kNN meta-learner | 2.87 (2.66–3.07) | 1.64 (1.51–1.76) | 26.4% | 0.50 | 3.11 ± 0.20 | 1.72 ± 0.12 |
| LightGBM regressor | 2.57 (2.41–2.72) | 1.55 (1.44–1.67) | 26.8% | 0.52 | 2.64 ± 0.14 | 1.61 ± 0.09 |
| LightGBM LambdaRank | **2.37** (2.22–2.53) | **1.23** (1.13–1.34) | 26.6% | 0.52 | **2.41 ± 0.07** | **1.28 ± 0.06** |
| FLAML (5 s/fold, 300 tasks) | 4.49 (3.91–5.19) | – | 9.7% | – | – | – |

How to read it:

* **Regret@3 is where the meta-models clearly win.** All three beat single best (1.96), and their CIs do not
  overlap with it. LambdaRank (1.23) is also clearly ahead of the regressor (1.55) and kNN (1.64). Running its top 3
  costs about 6 s per task, against 4 s for single best's top 3.
* **At Regret@1 the gain is small.** LambdaRank is lowest (2.37 against 2.64 for single best, and 2.41 against
  2.64 over seeds), but the CIs overlap. A paired bootstrap that resamples parents (so the 10 subsamples of one
  dataset move together) gives a difference of −0.26 points, 95% interval −0.56 to −0.01: only just significant.
  The regressor (−0.07, −0.22 to +0.09) and kNN (+0.23, −0.16 to +0.66) are not distinguishable from single best,
  and LambdaRank against the regressor is not significant either (−0.19, −0.44 to +0.03). Top-1 hit rates are all
  about 26–27%, so no method picks the exact winner more often than single best; the gains come from better near
  misses.
* **Real tasks only** (`summary_real.csv`, 810 tasks from 81 parents, the subset research question 3 needs): the
  same ordering. LambdaRank has Regret@1 2.06 (1.89–2.22) against 2.37 (2.18–2.56) for single best (paired
  difference −0.32, −0.73 to +0.05, not significant), and Regret@3 0.95 (0.85–1.06) against 1.80 (1.64–1.97),
  clearly better.
* **No single family dominates.** Winners are spread over SVM-RBF (18%), extra trees (16%), gradient boosting
  (16%), MLP (13%), random forest (12%) and logistic regression (11%).
* **Landmarkers are not needed here** (`summary_no_landmarkers.csv`). Without them every meta-model is level or
  slightly better: LambdaRank 2.27 / 1.20, the regressor 2.45 / 1.57, kNN 2.76 / 1.53 (Regret@1 / Regret@3), all
  inside the CIs above. This differs from the local scale test, where kNN got clearly worse without them. The LDA
  landmarker still has the second-highest gain in the regressor (`feature_gain.csv`), so its information seems to
  be available from the other meta-features too.
* **FLAML** (5 s per fold, 300 evenly spread tasks, about 16 s per task) has Regret@1 4.49 (3.91–5.19). On the same
  300 tasks LambdaRank's Regret@1 is 2.43 (2.09–2.75) and single best's 2.53, and FLAML beats LambdaRank's first
  pick on 30% of them. Caveats: the budget rule (`max(5, cost@3 / 3)`) gave FLAML about 16 s per task against
  about 1.6 s for LambdaRank's top 1 and 6 s for its top 3, so FLAML had more compute, not less. FLAML's search
  space has no SVM, kNN or MLP, which win about 35% of tasks here. It reported no best estimator on 2 of its 900
  folds. And regret against the best of 38 cross-validated scores is optimistic about the oracle, which penalises
  any single method.

Compared with the local scale test, LambdaRank's lead holds up with real data in the pool, and the meta-models'
Regret@3 advantage is larger. The Regret@1 advantage over single best stays small.

### Phase 2 pilot (`results/pilot_llm/`)

Qwen2.5-1.5B-Instruct with QLoRA was run on the pilot (120 tasks) in Colab: about 40 minutes on a T4. Zero-shot, the
base model is worse than random (Regret@1 8.64). The fine-tuned numeric model names every pipeline (100% coverage)
and reaches Regret@1 3.42, against 2.55 for single best. But it **collapsed to a nearly fixed ranking**: 11
distinct rankings for 120 tasks, and one ranking for all 24 tasks of fold 0. So it learned an average order, not to
read the profile. The semantic model varies more (35 rankings) and is lowest on the 16 real tasks (1.34 against
1.66 for single best). That subset is 4 famous datasets, the intervals overlap, and the model may recognise them
from pretraining. Details in `docs/REPORT_DRAFT.md` Section 5. `evaluate` now reports `distinct_rankings` and
`distinct_top1` for every method, so a collapse like this shows up in the main table.

### Scale test: full-study settings on local sources (`configs/local_full.yaml`)

This run uses 54 parents (4 scikit-learn, 40 synthetic, 10 moons/circles) with 10 subsamples each, giving 540 tasks.
The 38-pipeline portfolio and the 2,000-row cap make 61,560 fits, which took 21 minutes on 4 cores. The table uses
the shared `folds.csv` split. The ± values are the mean and standard deviation over 5 outer-CV seeds.

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman | Regret@1 over 5 seeds | Regret@3 over 5 seeds |
|---|---|---|---|---|---|---|
| Random | 6.88 (6.23–7.62) | 3.05 (2.71–3.40) | 13.3% | 0.00 | 7.20 ± 0.45 | 3.02 ± 0.10 |
| Single best | 2.94 (2.66–3.21) | 2.24 (1.99–2.49) | 25.2% | 0.36 | 2.94 ± 0.00 | 2.24 ± 0.00 |
| kNN meta-learner | 3.15 (2.85–3.48) | 1.92 (1.70–2.15) | 23.9% | 0.47 | 3.21 ± 0.15 | 1.93 ± 0.07 |
| LightGBM regressor | 3.08 (2.78–3.40) | 1.94 (1.71–2.17) | 24.8% | 0.48 | 3.05 ± 0.10 | 1.98 ± 0.04 |
| LightGBM LambdaRank | **2.72** (2.46–2.99) | **1.53** (1.35–1.74) | 26.3% | 0.44 | **2.77 ± 0.10** | **1.57 ± 0.04** |

What changed compared with the 120-task pilot:

* **LambdaRank now performs best.** In the pilot it was the weakest and least stable meta-model, and the report
  put this down to too few training tasks. With about 430 training tasks per fold it has the lowest regret at both
  cutoffs.
* **The meta-models beat single best at Regret@3** (1.53–1.94 against 2.24). Only LambdaRank's CI (1.35–1.74)
  clears single best's (1.99–2.49). At Regret@1 the gain is small (2.77 against 2.94 averaged over seeds), and the
  CIs overlap.
* **Landmarker ablation** (`summary_no_landmarkers.csv`). Without the landmarkers kNN gets clearly worse (3.15 →
  3.61 at Regret@1, 1.92 → 2.21 at Regret@3). LambdaRank changes only a little (2.72 → 2.85 and 1.53 → 1.62), well
  inside its CIs. Landmarkers still carry 56% of the LightGBM regressor's gain.
* **No single family dominates.** The most frequent winners are extra trees (17%), MLP (15%) and SVM (15%).
* **FLAML** (5 s per fold, 60 evenly spread tasks, about 15 s per task) has a Regret@1 of 5.78 (4.43–7.16). On the
  same 60 tasks LambdaRank scores 2.71 at Regret@1 and 1.41 at Regret@3, using about 3 s to run its top 3. There are
  two caveats. FLAML's search space has no SVM, kNN or MLP, which often win on this synthetic-heavy pool. And regret
  against the best of 38 cross-validated scores is optimistic about the oracle, which penalises any single method.

**On real tasks only** (`summary_real.csv`: 40 tasks from the 4 scikit-learn parents), single best (Regret@1 1.27)
and the LightGBM regressor (1.29) are level, and LambdaRank is lowest at Regret@3 (0.81 against 1.04). With 4
parents this subset is far too small to conclude anything.

About 85% of these parents are synthetic, so treat this run as a test that the full-scale pipeline works and as a
first sign of trends. It is not a finding about real data yet.

### Pilot (`configs/pilot.yaml`)

This run uses 30 parents and 120 tasks, with 18 pipelines and 3-fold CV (6,480 fits).

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman | Regret@1 over 5 seeds |
|---|---|---|---|---|---|
| Random | 5.28 (4.29–6.54) | 2.19 (1.72–2.72) | 13.3% | −0.05 | 4.98 ± 0.39 |
| Single best | 2.54 (2.07–3.06) | 1.72 (1.35–2.10) | 29.2% | 0.36 | 2.78 ± 0.33 |
| kNN meta-learner | 3.39 (2.69–4.23) | 1.48 (1.10–1.91) | 21.7% | 0.36 | 3.44 ± 0.32 |
| LightGBM regressor | 2.80 (2.28–3.40) | 1.33 (1.01–1.72) | 25.0% | 0.39 | 3.03 ± 0.29 |
| LightGBM LambdaRank | 3.55 (2.67–4.63) | 1.43 (1.08–1.84) | 24.2% | 0.33 | 3.48 ± 0.22 |

These pilot numbers differ from the Milestone 2 report because the synthetic parents are generated differently. At
this size the CIs overlap too much to separate the selectors, and LambdaRank is unstable. Its pilot Regret@1 moved
between 2.93 and 3.55 from nothing more than 11 of 2,160 relevance labels flipping on float noise; that noise is
fixed now (see the implementation notes).

## Implementation notes

* **No leakage from preprocessing.** Imputation, encoding and scaling live inside each sklearn pipeline, so they
  are fitted on the training folds only. At the meta level, the kNN scaler and LightGBM are fitted on the training
  tasks of each outer fold only.
* **Failed fits.** A failing or timed-out fit is stored as blank in `performance.csv` and replaced by the task's
  lowest score at load time. The time limit uses `SIGALRM`, and fits running inside compiled code (for example
  libsvm) are also marked as timed out once they return late.
* **Reproducible across library versions.** Meta-level CSV files are read with exact (`round_trip`) float
  parsing, and LambdaRank's 0–10 relevance labels are rounded so that 1e-15 float noise cannot push a value across
  a .5 boundary. Accuracies on small tasks are multiples of 1/n, so such boundaries are common. Before this fix,
  moving from pandas 3.0 to 3.1 changed pilot LambdaRank results.
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
- [x] Full build on OpenML-CC18 plus supplementary datasets: 131 parents, 1,310 tasks (`results/full/`,
      meta-dataset in `metadata/full/`)
- [ ] TabPFN in the portfolio (`portfolio: full_tabpfn`)
- [x] Phase 2 code: prompts A/B, per-fold QLoRA training, generation, scoring with coverage, Colab notebook
      (tested end to end with a tiny offline model)
- [x] Phase 2 pilot run on a Colab T4 (base, fine-tuned A, fine-tuned B): fine-tuned A collapsed to a fixed ranking
- [ ] Phase 2 run on the full study (`META = "metadata/full"`)
- [x] Gradio demo and `recommend` command
- [ ] Final report (draft with Phase 1 numbers in `docs/REPORT_DRAFT.md`)
