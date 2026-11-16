# MetaPilot: Meta-Learned and LLM-Based Pipeline Selection for Tabular Machine Learning

**Final report: working draft**

Atharv Arora (S24CSEU2439), Ansh Dixit (S24CSEU1018), Anvi Tyagi (S24CSEU1026), Fairy Tyagi (S24CSEU0794)
Guide: Dr. Neha Kumari. Intelligent Model Design Lab.

> **Status of this draft.** Phase 1 numbers come from this repository's pilot and from a 540-task scale test.
> The full OpenML-CC18 study is running; its tables replace Section 4.2 when it finishes (`results/full/`).
> **Phase 2 (LLM) has not been run.** The code and Colab notebook are ready, but no fine-tuned model exists yet,
> so Sections 5 and 6 (RQ2, RQ3) have no numbers. An earlier draft (IMD 1.0) reported LLM tables that came from
> a simulation that returned the single-best ranking; they have been removed (see `docs/IMD_1.0_REVIEW.md`).

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
* Full study: OpenML-CC18 (72 datasets) and supplementary small and high-dimensional OpenML datasets, plus
  synthetic parents. Parents that look like copies of each other are dropped, for example scikit-learn's breast
  cancer, which is wdbc in CC18.

**Tasks.** Each parent is subsampled several times: 4 subsamples in the pilot and 10 in the scale test and full
study. A subsample keeps a random 30–100% of rows and 50–100% of columns, and is capped at 800 rows (pilot) or
2,000 rows (scale test and full study). Subsamples with a class under 6 rows are rejected.

**Portfolio.**

* Pilot: 18 pipelines from 10 families (Table 2 of the Milestone 2 report).
* Scale test and full study: 38 pipelines, with 3–6 configurations per family.

Imputation, encoding and scaling are fitted inside each CV training fold. Each pipeline's score on a task is its
mean 3-fold stratified CV accuracy. A failed or timed-out fit takes the task's lowest score.

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

### 4.1 Pilot: 120 tasks, 18 pipelines

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman |
|---|---|---|---|---|
| Random | 5.28 (4.29–6.54) | 2.19 (1.72–2.72) | 13.3% | −0.05 |
| Single best | 2.54 (2.07–3.06) | 1.72 (1.35–2.10) | 29.2% | 0.36 |
| kNN meta-learner | 3.39 (2.69–4.23) | 1.48 (1.10–1.91) | 21.7% | 0.36 |
| LightGBM regressor | 2.80 (2.28–3.40) | 1.33 (1.01–1.72) | 25.0% | 0.39 |
| LightGBM LambdaRank | 3.55 (2.67–4.63) | 1.43 (1.08–1.84) | 24.2% | 0.33 |

At 120 tasks, none of the meta-models beats single best at Regret@1. At Regret@3 the meta-models are slightly
better, but all the intervals overlap. LambdaRank is unstable at this size: changing 11 of its 2,160 training
labels (float noise, since fixed) moved its Regret@1 by 0.6 points.

### 4.2 Scale test: 540 tasks, 38 pipelines (85% synthetic parents)

| Selector | Regret@1 (95% CI) | Regret@3 (95% CI) | Top-1 hit | Spearman | Regret@1, 5 seeds | Regret@3, 5 seeds |
|---|---|---|---|---|---|---|
| Random | 6.88 (6.23–7.62) | 3.05 (2.71–3.40) | 13.3% | 0.00 | 7.20 ± 0.45 | 3.02 ± 0.10 |
| Single best | 2.94 (2.66–3.21) | 2.24 (1.99–2.49) | 25.2% | 0.36 | 2.94 ± 0.00 | 2.24 ± 0.00 |
| kNN meta-learner | 3.15 (2.85–3.48) | 1.92 (1.70–2.15) | 23.9% | 0.47 | 3.21 ± 0.15 | 1.93 ± 0.07 |
| LightGBM regressor | 3.08 (2.78–3.40) | 1.94 (1.71–2.17) | 24.8% | 0.48 | 3.05 ± 0.10 | 1.98 ± 0.04 |
| LightGBM LambdaRank | 2.72 (2.46–2.99) | 1.53 (1.35–1.74) | 26.3% | 0.44 | 2.77 ± 0.10 | 1.57 ± 0.04 |

With 4.5 times as many tasks, LambdaRank has the lowest regret at both cutoffs. At Regret@3 all three
meta-models are ahead of single best (2.24, CI 1.99–2.49). LambdaRank's interval (1.35–1.74) does not overlap
single best's, while kNN's and the regressor's overlap it slightly. At Regret@1 only LambdaRank is ahead, by 0.2
points, and its CI overlaps single best's. No family wins more than 17% of tasks
(extra trees 17%, MLP 15%, SVM 15%), so the selection problem is not trivial. Figures:
`results/local_full/figures/`.

**Provisional answer to RQ1.** This holds only on synthetic-heavy data and still has to be confirmed on OpenML.
A meta-model's advantage over a strong default is small for the very first recommendation and becomes clear once
the top three are tried. That is the opposite of the pilot's reading ("the advantage disappears at k = 3"), which
did not survive the larger meta-dataset.

### 4.3 Landmarker ablation

| Selector | Regret@1, 14 features | Regret@1, without landmarkers | Regret@3, 14 features | Regret@3, without landmarkers |
|---|---|---|---|---|
| kNN meta-learner | 3.15 | 3.61 | 1.92 | 2.21 |
| LightGBM regressor | 3.08 | 3.08 | 1.94 | 2.02 |
| LightGBM LambdaRank | 2.72 | 2.85 | 1.53 | 1.62 |

Landmarkers carry 56% of the LightGBM regressor's gain, and the kNN meta-learner depends on them clearly. The
LightGBM models lose little without them, so their advantage is not just a shortcut through the naive Bayes
landmarker matching the naive Bayes pipeline.

### 4.4 FLAML

On 60 evenly spread tasks of the scale test, FLAML had 5 s per CV fold (about 15 s per task). Its Regret@1 was
5.78 (4.43–7.16). On the same tasks LambdaRank scored 2.71 at Regret@1 and 1.41 at Regret@3, and running its top 3
took about 3 s. FLAML's default search space has no SVM, kNN or MLP, which win often on this pool, and regret is
measured against the best of 38 CV estimates. The full-study run uses a budget matched to the meta-models' cost.

## 5. Phase 2: LLM selector (RQ2, RQ3)

**Not yet run.** The protocol is implemented (`metapilot/llm.py`, `metapilot/llm_train.py`,
`notebooks/phase2_qlora.ipynb`) and tested end to end with a tiny offline model. The three variants to run
are: base model zero-shot with Prompt A, fine-tuned on Prompt A, and fine-tuned on Prompt B. Each fine-tuned
variant needs one adapter per outer fold.

Reporting rules for this section:

* Every LLM row reports **coverage**, the share of pipelines the model named itself. Missing pipelines are filled
  in from the single-best order, so a model with low coverage scores close to single best. Such a row is not
  evidence that the LLM selects anything.
* RQ3 is answered on real (non-synthetic) tasks only (`summary_real.csv`). Synthetic descriptions list their
  generator settings and would give Prompt B extra numeric information.

## 6. Answers to the research questions

* **RQ1:** provisional answer in Section 4.2; to be confirmed with the OpenML study.
* **RQ2:** pending the Phase 2 run.
* **RQ3:** pending the Phase 2 run.

## 7. Limitations

* The scale test is about 85% synthetic. The pilot's real-data subset is 4 parents, and conclusions about real
  data wait for OpenML.
* Each performance value comes from 3-fold CV on at most 2,000 rows, so it is noisy. Regret against the best of
  38 such estimates favours the oracle.
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
```
