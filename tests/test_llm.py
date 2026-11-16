import json
import shutil

import pandas as pd
import pytest

from metapilot import experiment, llm
from metapilot.metafeatures import FEATURES
from metapilot.metalevel import MetaData


def test_parse_ranking_matches_exact_names_in_order():
    pipes = ["rf_100", "rf_100_d5", "svm_rbf_C1", "svm_rbf_C10", "logreg_C0.1", "logreg_C10"]
    text = "1. SVM_RBF_C10\n2. rf_100_d5, logreg_C0.1; then rf_100 and svm_rbf_C10 again"
    assert llm.parse_ranking(text, pipes) == ["svm_rbf_C10", "rf_100_d5", "logreg_C0.1", "rf_100"]
    assert llm.parse_ranking("", pipes) == []
    assert llm.parse_ranking("logreg_C0.15", pipes) == []  # not logreg_C0.1


@pytest.fixture
def meta_copy(tiny_meta, tiny_cfg, tmp_path):
    d = tmp_path / "meta"
    shutil.copytree(tiny_meta, d, ignore=shutil.ignore_patterns("parts", "llm", "folds*.csv"))
    llm.write_prompts(d, tiny_cfg)
    return d


def test_prompts(meta_copy):
    meta = MetaData.load(meta_copy)
    folds = pd.read_csv(meta_copy / "folds.csv", index_col="task_id")["fold"]
    rows = llm.load_jsonl(meta_copy / "llm" / "prompts.jsonl")
    assert [r["task_id"] for r in rows] == list(meta.perf.index)
    text = {json.loads(l)["task_id"]: json.loads(l) for l in open(meta_copy / "text_meta.jsonl")}
    for r in rows:
        assert r["fold"] == folds[r["task_id"]]
        target = r["target"].split(", ")
        assert sorted(target) == sorted(meta.pipelines)
        accs = meta.perf.loc[r["task_id"], target].to_numpy()
        assert (accs[:-1] >= accs[1:]).all()  # best first
        mf = meta.mf.loc[r["task_id"]]
        for f in ("cols_rows_ratio", "mean_abs_corr", "pca_frac_95", "lm_lda", "class_entropy"):
            assert f"{mf[f]:.3f}" in r["numeric"]
        assert r["numeric"].count("Candidate pipelines:") == 1 and r["semantic"].count("Candidate pipelines:") == 1
        assert r["semantic"].startswith(r["numeric"].split("\nCandidate pipelines:")[0])
        first_col = text[r["task_id"]]["columns"][0]["name"]
        assert first_col in r["semantic"] and "Column names" not in r["numeric"]
    assert len(FEATURES) == 14


def _write_preds(meta_dir, variant, outputs):
    d = meta_dir / "llm" / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    rows = llm.load_jsonl(meta_dir / "llm" / "prompts.jsonl")
    with open(d / f"{variant}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"task_id": r["task_id"], "fold": r["fold"], "output": outputs(r)}) + "\n")


def test_llm_rows_in_evaluate(meta_copy, tiny_cfg):
    _write_preds(meta_copy, "oracle", lambda r: r["target"])          # perfect rankings
    _write_preds(meta_copy, "silent", lambda r: "I cannot help with that.")
    summary = experiment.run(meta_copy, meta_copy.parent / "run", tiny_cfg)
    oracle, silent = summary.loc["LLM: oracle"], summary.loc["LLM: silent"]
    assert oracle["regret@1"] == pytest.approx(0) and oracle["llm_coverage"] == 1
    assert silent["llm_coverage"] == 0 and silent["llm_fully_parsed"] == 0
    # With nothing parsed, the ranking is exactly the training-fold single best.
    sb = pd.read_csv(meta_copy.parent / "run" / "scores" / "single_best.csv", index_col="task_id")
    sc = pd.read_csv(meta_copy.parent / "run" / "scores" / "llm_silent.csv", index_col="task_id")
    assert (sb.rank(axis=1, method="first") == sc.rank(axis=1, method="first")).all().all()
    for col in ("regret@1", "regret@1_lo", "regret@1_hi", "regret@3"):  # same tasks, same order, same CIs
        assert silent[col] == pytest.approx(summary.loc["Single best", col])
    real = pd.read_csv(meta_copy.parent / "run" / "summary_real.csv", index_col="selector")
    assert real.loc["LLM: oracle", "n_tasks"] == 4  # iris and wine, 2 subsamples each
    assert "LLM named 0%" in experiment.format_summary(summary)


def test_predictions_must_use_shared_folds(meta_copy, tiny_cfg):
    _write_preds(meta_copy, "bad", lambda r: r["target"])
    path = meta_copy / "llm" / "predictions" / "bad.jsonl"
    rows = llm.load_jsonl(path)
    rows[0]["fold"] = (rows[0]["fold"] + 1) % tiny_cfg.outer_folds
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="shared split"):
        experiment.run(meta_copy, meta_copy.parent / "run", tiny_cfg)


def test_train_and_predict_tiny_model(meta_copy, tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from llm_fixtures import make_tiny_model

    from metapilot import llm_train

    prompts = meta_copy / "llm" / "prompts.jsonl"
    rows = llm.load_jsonl(prompts)
    model = make_tiny_model([r["numeric"] + r["semantic"] + r["target"] for r in rows], tmp_path / "tiny")
    adapter = llm_train.train(prompts, 0, "semantic", tmp_path / "ad", str(model), max_steps=1, batch_size=2,
                              grad_accum=1, max_length=1024, load_in_4bit=False)
    info = json.loads((adapter / "train_info.json").read_text())
    test_ids = {r["task_id"] for r in rows if r["fold"] == 0}
    assert info["n_train_tasks"] == len(rows) - len(test_ids) and not test_ids & set(info["train_task_ids"])

    out = meta_copy / "llm" / "predictions" / "ft_semantic.jsonl"
    llm_train.predict(prompts, "semantic", out, str(model), adapter, fold=0, max_new_tokens=8, load_in_4bit=False)
    llm_train.predict(prompts, "semantic", out, str(model), adapter, fold=0, max_new_tokens=8, load_in_4bit=False)
    preds = llm.load_jsonl(out)
    assert sorted(p["task_id"] for p in preds) == sorted(test_ids)  # resumed, no duplicates
    with pytest.raises(ValueError, match="trained for fold 0"):
        llm_train.predict(prompts, "semantic", tmp_path / "x.jsonl", str(model), adapter, fold=1)


def test_t4_uses_fp16(monkeypatch):
    torch = pytest.importorskip("torch")
    from metapilot import llm_train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))  # Tesla T4
    assert llm_train.native_bf16() is False
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))  # A100
    assert llm_train.native_bf16() is True


def test_adapters_float32_for_fp16_training():
    torch = pytest.importorskip("torch")
    from metapilot.llm_train import adapter_dtype_for_mixed_precision

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model[0].requires_grad_(False)
    model.to(torch.bfloat16)  # what TRL does to QLoRA adapters
    adapter_dtype_for_mixed_precision(model, bf16=True)
    assert model[1].weight.dtype == torch.bfloat16  # bf16 training: leave as is
    adapter_dtype_for_mixed_precision(model, bf16=False)
    assert model[1].weight.dtype == torch.float32 and model[1].bias.dtype == torch.float32
    assert model[0].weight.dtype == torch.bfloat16  # frozen weights untouched
