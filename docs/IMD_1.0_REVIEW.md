# Review of the IMD 1.0 upload

IMD 1.0 is a second implementation of MetaPilot (a `src/` package with Phase 1, Phase 2, a Gradio demo and a
report draft). This note records what was found in it and what was carried into this repository. The upload
itself is not committed.

## Must be fixed before anything from IMD 1.0 is reported

**The Phase 2 (LLM) results are not LLM results.** `src/llm_infer.py` never loads a model during evaluation:
`evaluate_llm_variants` calls `simulate_or_run_llm_inference` without a `checkpoint_dir`, so it always takes the
"deterministic simulation" branch. That branch returns the single-best ranking unchanged for numeric prompts. For
semantic prompts on real datasets it re-sorts pipelines if the prompt contains the words "cancer", "linear" or
"digits". As a result:

* `results/llm_evaluation_all.csv` reports Regret@1 = 2.838950079293988 for all three "LLM" variants, which is
  the single-best baseline to the last digit, and the same Regret@3 too;
* the small "improvement" of Prompt B on real datasets (Spearman 0.398 → 0.416) comes from that keyword rule;
* `docs/REPORT_DRAFT.md` presents these numbers as findings for research questions 2 and 3 and explains them with
  mechanisms that were never tested (tokenised digits, semantic priors on clinical data).

These tables and the RQ2/RQ3 answers must not be submitted. `docs/REPORT_DRAFT.md` in this repository replaces
them with "not yet run" until the fine-tuning has actually been done.

**The multi-seed evaluation never varied anything.** Every seed reloads the same saved `folds.json`. kNN and
single best have no randomness, and LightGBM without bagging gives the same model for any seed. So the standard
deviations are 0.00 for every method except Random. This repository's `evaluate --seeds` draws a new grouped
split for each seed (`folds_seed<k>.csv`).

**Other issues in the Phase 2 code:**

* The Colab notebook fine-tunes on *all* prompts with no fold split. Any evaluation afterwards would test on
  training tasks.
* If `all_prompts.parquet` is missing, the notebook silently trains on a made-up demo sample.
* The notebook never generates predictions for held-out tasks, so nothing connects training to evaluation.
* `max_new_tokens=150` is about enough for 18 pipeline names and too short for the ~38 in the full portfolio.
  Truncated outputs would fall back to single best without any warning.
* Prompt A leaves out 8 of the 14 meta-features (correlation, skewness, kurtosis, PCA fraction, entropy and
  others), so the LLM did not get "the same numeric information" as LightGBM, which RQ2 requires.
* Prompt B lists the candidate pipelines twice.
* Both `get_peft_model` and `SFTTrainer(peft_config=...)` are applied, which wraps the model twice.

**Smaller points:**

* The demo's "Primary decision factors" come from fixed if-rules (for example "LDA landmarker is highest →
  linear separability"), not from the meta-model. When the meta-dataset is missing, the demo silently returns an
  unranked list.
* The report draft's author line ("Senior ML Research Assistant & DeepMind Pair Programming Engineer") should
  be the group members.
* `docs/PROGRESS.md` marks M5–M9 as completed and then lists M7–M9 again as pending. M5's OpenML scale-up and
  M6's TabPFN/FLAML baselines were implemented but never run.

## What was carried over, and how

| IMD 1.0 | In this repository |
|---|---|
| `src/serialize_text.py` (Prompts A/B) | `metapilot/llm.py`. Prompt A now has all 14 meta-features, Prompt B lists the candidates once, and each record carries its fold from the shared `folds.csv`. |
| `src/llm_finetune.py`, notebook | `metapilot/llm_train.py` and `notebooks/phase2_qlora.ipynb`. One adapter per outer fold, trained only on the other folds, with the loss on the ranking only. An adapter refuses to predict any fold but its own. Output length scales with the portfolio size. Work is saved to Drive after each fold. There is no simulation path: no GPU means an error. |
| `src/llm_infer.py` parser and evaluation | `metapilot.llm.parse_ranking` and `llm_scores`, scored inside `metapilot evaluate`. Names are matched exactly (`rf_100` is not matched inside `rf_100_d5`). Every LLM row reports `llm_coverage`, and `evaluate` warns when the model names fewer than 90% of the pipelines. `summary_real.csv` gives the real-data-only comparison for all methods. |
| `app/gradio_demo.py` | `app/demo.py` and `metapilot/recommend.py`. The ranking comes from LambdaRank trained on the meta-dataset. "Why" lists the most similar training tasks and their winners. Optional verification runs the top 3 pipelines on the uploaded data. The run-history log was left out. |
| Phase 1 (`src/` pipeline, cached results) | Not carried over: this repository already covers it, including sharding, FLAML and the ablation. The IMD 1.0 Phase 1 pilot numbers (for example kNN 1.91 against single best 2.84 Regret@1) do look genuinely computed: 2,160 cached evaluations, 120 tasks × 18 pipelines. They come from a different run (seed 42, different synthetic data) and are not comparable to this repository's tables. |

All of the Phase 2 code here was exercised end to end in tests: prompts, training a LoRA adapter, generation,
resuming, the fold guard and scoring. The tests use a tiny randomly initialised Qwen2-architecture model built
offline. They show that the code path works, not how well Qwen2.5-1.5B ranks pipelines; that needs the GPU run.
