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
1. File > Open notebook > GitHub, enter `atharv-0509/triage`, choose branch `claude/starting-from-beginning-joh2ym`,
   then pick `notebooks/phase2_qlora.ipynb`.
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
`RUN_NAME = "full"` in the first cell once `metadata/full/` has been pushed. The notebook was run end to end on CPU
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

(`P` is `data/pilot/llm/prompts.jsonl`.) Each adapter trains only on tasks outside its fold and refuses to predict
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
better), using 5-fold grouped outer CV. The full OpenML study is running in a separate session and will be added
here when it finishes.

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
- [ ] Full build on OpenML-CC18 plus supplementary datasets (~150 parents): running in a separate cloud session
- [ ] TabPFN in the portfolio (`portfolio: full_tabpfn`)
- [x] Phase 2 code: prompts A/B, per-fold QLoRA training, generation, scoring with coverage, Colab notebook
      (tested end to end with a tiny offline model)
- [ ] Phase 2 runs on a GPU: base model, fine-tuned A, fine-tuned B; then evaluation on the shared folds
- [x] Gradio demo and `recommend` command
- [ ] Final report (draft with Phase 1 numbers in `docs/REPORT_DRAFT.md`)
