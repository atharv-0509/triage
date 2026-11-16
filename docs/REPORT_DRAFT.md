# MetaPilot: Meta-Learned and LLM-Based Pipeline Selection for Tabular Machine Learning

**Final report: working draft**

Atharv Arora (S24CSEU2439), Ansh Dixit (S24CSEU1018), Anvi Tyagi (S24CSEU1026), Fairy Tyagi (S24CSEU0794)
Guide: Dr. Neha Kumari. Intelligent Model Design Lab.

> **Status of this draft.** Phase 1 is complete on the full OpenML study (Section 4.1, `results/full/`), with the
> pilot and a 540-task scale test kept as earlier stages (Section 4.4).
> **Phase 2 (LLM) has been run on the pilot only** (Section 5). The run on the full study is still to do.
> An earlier draft (IMD 1.0) reported LLM tables that came from a simulation that returned the single-best
> ranking; they have been removed (see `docs/IMD_1.0_REVIEW.md`).

## 1. Research questions

1. **RQ1.** Can a meta-model trained on measured pipeline performance select better pipelines than a strong fixed
   default, and at what evaluation budget does this advantage disappear?
2. **RQ2.** Can a fine-tuned LLM perform as well as a tree-based meta-model when both get the same numeric
   information?
3. **RQ3.** Does semantic information (column names, dataset description) improve pipeline selection over numeric
   meta-features alone?

## 2. Data and meta-dataset

**Parent datasets.**

* Pilot: 30 parents. These are 4 scikit-learn datasets (breast cancer, wine, iris, digits), 20
  `make_classification` datasets with varied size, dimensionality, class separation, label noise and
  non-linearity, and 6 moons/circles datasets with extra noise columns.
* Scale test: 54 parents with the same sources (4 + 40 + 10).
* Full study: 131 parents. These are all 72 OpenML-CC18 datasets, 7 supplementary OpenML datasets (ionosphere,
  sonar, heart-statlog, hepatitis, haberman, arcene, gina; madelon is already in CC18), wine and iris from
  scikit-learn, 40 `make_classification` and 10 moons/circles parents. 81 of the parents are real. Parents that look
  like copies of each other are dropped, which is why scikit-learn's breast cancer and digits are not used: CC18 has
  them as wdbc and optdigits.

**Tasks.** Each parent is subsampled several times: 4 subsamples in the pilot and 10 in the scale test and full
study. A subsample keeps a random 30–100% of rows and 50–100% of columns, and is capped at 800 rows (pilot) or
2,000 rows (scale test and full study). Subsamples with a class under 6 rows are rejected.

**Portfolio.**

* Pilot: 18 pipelines from 10 families (Table 2 of the Milestone 2 report).
* Scale test and full study: 38 pipelines, with 3–6 configurations per family.

Imputation, encoding and scaling are fitted inside each CV training fold. Each pipeline's score on a task is its
mean 3-fold stratified CV accuracy. A failed or timed-out fit takes the task's lowest score. In the full study,
which ran 149,340 fits (about 20 CPU-hours), this affected 75 of the 49,780 task-pipeline cells (0.15%).
`knn_30_dist` cannot run on 21 tiny synthetic tasks with fewer than 30 training rows per fold, and three
HistGradientBoosting variants and `rf_300_mf0.5` hit the 120 s per-fit limit on CIFAR-10 and Devnagari-Script.

**Meta-features.** There are 14 in four groups: simple (4), statistical (4), information-theoretic (2) and
landmarkers (4: decision stump, 1-NN, Gaussian naive Bayes and LDA, each scored by 2-fold CV on at most 200 rows).

**Evaluation protocol.** All methods use one shared 5-fold GroupKFold split grouped by parent dataset
(`folds.csv`), so subsamples of one parent never appear on both sides. The metrics are:

* Regret@k: the best accuracy on the task minus the best accuracy among the top k recommended pipelines, in
  accuracy points;
* top-1 hit: Regret@1 ≤ 0.5;
* Spearman correlation between predicted scores and measured accuracies;
* 95% bootstrap CIs over tasks (2,000 resamples).

Robustness is checked by repeating the grouped split with 5 seeds.

## 3. Methods

| Method | Description |
|---|---|
| Random | Random order (lower bound) |
| Single best | Pipelines ordered by mean training accuracy; the same order for every task |
| kNN meta-learner | Averages the normalised scores of the 5 nearest training tasks in standardised meta-feature space (auto-sklearn style) |
| LightGBM regressor | Predicts the gap to the best pipeline for each (task, pipeline) pair; 300 trees, learning rate 0.05, 15 leaves |
| LightGBM LambdaRank | Same inputs, ranking objective on per-task relevance labels 0–10 |
| FLAML | AutoML on the base-level task with a fixed time budget per CV fold, on the same folds |
| LLM (Phase 2) | Qwen2.5-1.5B-Instruct with 4-bit QLoRA, one adapter per outer fold, trained only on the other folds; Prompt A = all 14 meta-features, Prompt B = A plus column names and description |

## 4. Phase 1 results (RQ1)

### 4.1 Full study: 1,310 tasks from 131 parents, 38 pipelines

All rows use the shared `folds.csv` split. The ± columns are the mean and standard deviation over 5 outer-CV seeds.
Every number below was reproduced exactly from the committed meta-dataset (`metadata/full/`).

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman | Regret@1, 5 seeds | Regret@3, 5 seeds |
|---|---|---|---|---|---|---|
| Random | 6.62 (6.15–7.07) | 2.69 (2.50–2.89) | 11.4% | 0.00 | 6.65 ± 0.51 | 2.62 ± 0.08 |
| Single best | 2.64 (2.48–2.80) | 1.96 (1.82–2.09) | 26.5% | 0.45 | 2.64 ± 0.00 | 1.95 ± 0.01 |
| kNN meta-learner | 2.87 (2.66–3.07) | 1.64 (1.51–1.76) | 26.4% | 0.50 | 3.11 ± 0.20 | 1.72 ± 0.12 |
| LightGBM regressor | 2.57 (2.41–2.72) | 1.55 (1.44–1.67) | 26.8% | 0.52 | 2.64 ± 0.14 | 1.61 ± 0.09 |
| LightGBM LambdaRank | **2.37** (2.22–2.53) | **1.23** (1.13–1.34) | 26.6% | 0.52 | **2.41 ± 0.07** | **1.28 ± 0.06** |

Subsamples of one parent are not independent, so method differences are tested with a **paired bootstrap that
resamples parents** (all 10 subsamples of a dataset move together):

| Difference in Regret@1 | Mean | 95% interval |
|---|---|---|
| LambdaRank − single best | −0.26 | −0.56 to −0.01 |
| Regressor − single best | −0.07 | −0.22 to +0.09 |
| kNN − single best | +0.23 | −0.15 to +0.67 |
| LambdaRank − regressor | −0.19 | −0.43 to +0.03 |

**Where the advantage is.** The table below gives mean regret after evaluating the top k recommendations, and
LambdaRank's paired difference from single best (same parent-level bootstrap):

| k | 1 | 2 | 3 | 5 | 8 | 10 |
|---|---|---|---|---|---|---|
| Single best | 2.64 | 2.44 | 1.96 | 1.44 | 0.80 | 0.62 |
| LambdaRank | 2.37 | 1.59 | 1.23 | 0.88 | 0.61 | 0.49 |
| Difference (95% interval) | −0.26 (−0.55 to −0.01) | −0.84 (−1.12 to −0.61) | −0.72 (−0.99 to −0.51) | −0.56 (−0.80 to −0.36) | −0.19 (−0.27 to −0.13) | −0.14 (−0.20 to −0.08) |

**Real tasks only** (`summary_real.csv`, 810 tasks from 81 parents) shows the same ordering. LambdaRank's Regret@1
is 2.06 (1.89–2.22) against 2.37 (2.18–2.56) for single best; the paired difference is −0.32 (−0.73 to +0.05), not
significant. Its Regret@3 is 0.95 (0.85–1.06) against 1.80 (1.64–1.97).

Winners are spread over many families: SVM-RBF 18%, extra trees 16%, gradient boosting 16%, MLP 13%, random forest
12%, logistic regression 11%. Top-1 hit rates are about 26–27% for every method except random, so the meta-models
do not pick the exact winner more often than single best does. Their gain comes from better near misses and better
second and third choices.

### 4.2 Landmarker ablation

| Selector | Regret@1, 14 features | Regret@1, no landmarkers | Regret@3, 14 features | Regret@3, no landmarkers |
|---|---|---|---|---|
| kNN meta-learner | 2.87 | 2.76 | 1.64 | 1.53 |
| LightGBM regressor | 2.57 | 2.45 | 1.55 | 1.57 |
| LightGBM LambdaRank | 2.37 | 2.27 | 1.23 | 1.20 |

On the full study, removing the four landmarkers leaves every meta-model level or slightly better, and all the
differences fall inside the CIs. In the synthetic-heavy scale test, kNN was clearly worse without them (3.15 → 3.61).
The LDA landmarker still has the second-highest gain in the regressor, so the information it carries seems to be
available from the other meta-features as well. This also rules out the concern that the naive Bayes landmarker
leaks the naive Bayes pipeline's score: the meta-models do not need landmarkers at all.

### 4.3 FLAML

FLAML ran on 300 evenly spread tasks with 5 s per CV fold, about 16 s per task. Its Regret@1 was 4.49 (3.91–5.19).
On the same 300 tasks LambdaRank scored 2.43 and single best 2.53, and FLAML beat LambdaRank's first pick on 30% of
them. So FLAML had more compute than LambdaRank's top 1 (about 1.6 s) or top 3 (about 6 s) and still did worse on
average. Three caveats apply:

* FLAML's default search space has no SVM, kNN or MLP, which win about 35% of tasks here.
* Regret is measured against the best of 38 CV estimates, which is optimistic about the oracle and penalises any
  single method.
* A different budget could change the picture.

### 4.4 Earlier stages: pilot and scale test

| Selector | Pilot (120 tasks, 18 pipelines) Regret@1 / @3 | Scale test (540 tasks, 85% synthetic) Regret@1 / @3 |
|---|---|---|
| Single best | 2.54 / 1.72 | 2.94 / 2.24 |
| kNN meta-learner | 3.39 / 1.48 | 3.15 / 1.92 |
| LightGBM regressor | 2.80 / 1.33 | 3.08 / 1.94 |
| LightGBM LambdaRank | 3.55 / 1.43 | 2.72 / 1.53 |

At 120 tasks nothing beat single best at Regret@1. LambdaRank was also unstable there: 11 of 2,160 training labels
flipping on float noise (since fixed) moved its Regret@1 by 0.6 points. LambdaRank's lead appeared at 540 tasks
and held on the full study. Details: `results/pilot/`, `results/local_full/`.

## 5. Phase 2: LLM selector (RQ2, RQ3)

**Setup.** Qwen2.5-1.5B-Instruct with 4-bit QLoRA (r = 16, 3 epochs), one adapter per outer fold and prompt kind,
trained only on the other folds (`metapilot/llm_train.py`, `notebooks/phase2_qlora.ipynb`). There are three
variants: the base model zero-shot with Prompt A, fine-tuned on Prompt A (the 14 meta-features, the same
information LightGBM gets), and fine-tuned on Prompt B (Prompt A plus column names and description).

**Reporting rules.**

* Every LLM row reports **coverage**, the share of pipelines the model named itself. Missing pipelines are filled
  in from the single-best order, so a model with low coverage scores close to single best.
* Every row also reports **diversity**: how many different rankings and first picks the method produced. A model
  that gives every task the same ranking is a fixed default, whatever its regret.
* RQ3 is compared on real (non-synthetic) tasks only (`summary_real.csv`). Synthetic descriptions list their
  generator settings and would give Prompt B extra numeric information.

### 5.1 Pilot run (120 tasks, 18 pipelines; Colab T4, about 40 minutes)

Colab built its own copy of the pilot meta-dataset, so the Phase 1 rows differ slightly from Section 4.1. All rows
below were scored on that one build (`results/pilot_llm/`).

| Selector | Regret@1 (95% CI) | Regret@3 | Spearman | Coverage | Distinct rankings | Regret@1, 16 real tasks |
|---|---|---|---|---|---|---|
| Single best | 2.55 (2.07–3.07) | 1.72 | 0.36 | – | 1 per fold | 1.66 |
| kNN meta-learner | 3.40 (2.70–4.23) | 1.51 | 0.36 | – | – | 2.77 |
| LightGBM regressor | 3.06 (2.49–3.72) | 1.41 | 0.39 | – | – | 2.45 |
| LightGBM LambdaRank | 3.49 (2.66–4.51) | 1.64 | 0.33 | – | – | 3.26 |
| LLM, base (zero-shot, A) | 8.64 (7.75–9.67) | 3.01 | −0.11 | 81% | 33 (always `tree_full` first) | 9.49 |
| LLM, fine-tuned, A | 3.42 (2.83–4.07) | 1.80 | 0.30 | 100% | **11** (1–4 per fold) | 1.98 |
| LLM, fine-tuned, B | 4.08 (3.29–4.97) | 2.05 | 0.22 | 99% | 35 (4–11 per fold) | 1.34 (0.73–2.00) |

What the pilot shows:

* **Fine-tuning is needed.** Zero-shot, the base model does not follow the format reliably (81% coverage) and is
  worse than random (Regret@1 8.64 against 5.27).
* **The fine-tuned numeric model collapsed to a fixed ranking.** It produced 11 different rankings for 120 tasks;
  the fold 0 adapter gave the same ranking to all 24 of its test tasks. With 96 training tasks per fold it learned
  roughly the average ranking of its training data and ignores the dataset profile. Its regret, close to kNN's, is
  therefore the regret of a fixed default, and a worse one than single best. It is not evidence of selection.
* **Prompt B varies more** (35 rankings, 7 different first picks) and has the lowest Regret@1 on the 16 real tasks
  (1.34 against 1.66 for single best and 1.98 for Prompt A). Those 16 tasks come from only 4 parents (iris, wine,
  breast cancer, digits), the intervals overlap, and these datasets are famous enough that the model may know
  them from pretraining. This is not evidence for RQ3 yet.

### 5.2 Full study

To do: the same notebook with `META = "metadata/full"`. Two things change at once compared with the pilot. There
are about 10 times more training tasks per fold, and the candidates are listed in a different random order for
each task (`SHUFFLE_CANDIDATES = True`), so the model cannot echo the prompt's order. The diversity columns show
whether the collapse goes away, but not which of the two changes caused it. Separating them needs a second
full-scale run with `SHUFFLE_CANDIDATES = False`, which costs about the same GPU time again.

## 6. Answers to the research questions

* **RQ1: yes, modestly for the first pick and clearly for a small budget.** Trained on 1,310 tasks, LightGBM
  LambdaRank has lower mean regret than the single-best default at every budget up to k = 30; beyond that both are
  within 0.01 points of zero. At k = 1 the gain is small and only just significant (−0.26 accuracy points, 95%
  interval −0.56 to −0.01, resampling datasets). It is largest at k = 2–4 (about −0.7 to −0.8 points). It then
  shrinks: under 0.2 points from k = 7, under 0.1 from k = 16, and practically gone (under 0.02) from k = 23. The kNN
  meta-learner and the regressor are not distinguishable from single best at k = 1, and help only from k = 2 on.
  A meta-model is therefore most useful when a user can afford to try two to four pipelines, adds little beyond
  about eight, and nothing beyond about twenty. The pilot's reading, that the advantage disappears at k = 3, came
  from too few training tasks.
* **RQ2:** pilot only. The fine-tuned LLM did not learn to use the numeric information; it reproduced a fixed
  ranking. To be tested at full scale.
* **RQ3:** pilot only. There is a weak, non-significant hint on 16 real tasks that is confounded by pretraining
  knowledge of famous datasets. To be tested at full scale.

## 7. Limitations

* The full study has 81 real parents, 72 of them from OpenML-CC18, so the results describe medium-sized benchmark
  classification data. Tasks are capped at 2,000 rows, and very large or very wide datasets are underrepresented.
* The 10 subsamples of a parent are correlated. All significance statements use a bootstrap over parents, but the
  per-task CIs in the tables are narrower than they would be with independent tasks.
* Each performance value comes from 3-fold CV on at most 2,000 rows, so it is noisy. Regret against the best of
  38 such estimates favours the oracle.
* 75 task-pipeline cells (0.15%) failed or timed out and take the task's lowest score (Section 2).
* Accuracy is the only metric, and TabPFN is not yet in the portfolio.
* The pilot results in this draft differ from the Milestone 2 report because the synthetic parents are generated
  differently.

## 8. Reproduction

```bash
pip install -e ".[dev,flaml]"
python -m metapilot pilot --config configs/pilot.yaml
python -m metapilot build --config configs/local_full.yaml --out data/local_full
python -m metapilot evaluate --config configs/local_full.yaml --meta data/local_full --out runs/local_full --seeds 1 2 3 4
python -m metapilot evaluate --config configs/local_full.yaml --meta data/local_full --out runs/local_full_nolm --drop-group landmarkers
python -m metapilot flaml --config configs/local_full.yaml --meta data/local_full --budget 5 --max-tasks 60

# Full study: build (about 20 CPU-hours; shard it with --shard i/n), or reuse the committed meta-dataset
python -m metapilot build --config configs/full.yaml --out data/full
python -m metapilot evaluate --config configs/full.yaml --meta metadata/full --out runs/full --seeds 1 2 3 4
```
