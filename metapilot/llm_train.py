"""Phase 2: QLoRA fine-tuning and generation for the LLM selector
(report Section 4.4).

Needs the ``llm`` extra (``pip install -e ".[llm]"``) and, for the real
model, a GPU: Qwen2.5-1.5B-Instruct in 4-bit fits a free Colab T4. See
``notebooks/phase2_qlora_colab.ipynb`` for the full loop.

The protocol matches Phase 1: for outer fold k, the adapter is trained only
on tasks whose fold is not k, and predicts only tasks whose fold is k. Raw
outputs go to ``<meta_dir>/llm/predictions/<variant>.jsonl``, which
``metapilot evaluate`` parses and scores. Nothing here fills in or simulates
a ranking; a missing GPU or package is an error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .llm import load_jsonl

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KINDS = ("numeric", "semantic")


def _messages(row: dict, kind: str) -> list[dict]:
    return [{"role": "system", "content": row["system"]}, {"role": "user", "content": row[kind]}]


def _load_model(model_id: str, load_in_4bit: bool | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if load_in_4bit is None:
        load_in_4bit = cuda
    kwargs = {}
    if load_in_4bit:
        if not cuda:
            raise RuntimeError("4-bit loading needs a CUDA GPU; pass load_in_4bit=False to run on CPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        kwargs["device_map"] = "auto"
    elif cuda:
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tok, load_in_4bit


def train(prompts_path: str | Path, fold: int, kind: str, out_dir: str | Path, model_id: str = DEFAULT_MODEL,
          epochs: float = 3, learning_rate: float = 2e-4, lora_r: int = 16, lora_alpha: int = 32,
          batch_size: int = 2, grad_accum: int = 4, max_length: int = 2048, seed: int = 0,
          load_in_4bit: bool | None = None, max_steps: int = -1) -> Path:
    """Fine-tune a LoRA adapter on the tasks outside outer fold ``fold``."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rows = load_jsonl(prompts_path)
    train_rows = [r for r in rows if int(r["fold"]) != fold]
    if not train_rows or len(train_rows) == len(rows):
        raise ValueError(f"fold {fold} does not split the prompts into train and test")
    assert all(int(r["fold"]) != fold for r in train_rows)  # no test task in training

    # Prompt-completion format: the loss is computed on the ranking only.
    ds = Dataset.from_list([{
        "prompt": _messages(r, kind),
        "completion": [{"role": "assistant", "content": r["target"]}],
    } for r in train_rows])

    model, tok, four_bit = _load_model(model_id, load_in_4bit)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    out = Path(out_dir)
    args = SFTConfig(
        output_dir=str(out / "checkpoints"), num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate, lr_scheduler_type="cosine", warmup_steps=0.03,  # <1 = ratio
        logging_steps=10, save_strategy="no", report_to="none", seed=seed, max_length=max_length,
        bf16=bf16, fp16=cuda and not bf16, gradient_checkpointing=cuda,
        optim="paged_adamw_8bit" if four_bit else "adamw_torch",
    )
    peft = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft)
    result = trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    (out / "train_info.json").write_text(json.dumps({
        "base_model": model_id, "fold": fold, "kind": kind, "n_train_tasks": len(train_rows),
        "train_task_ids": [r["task_id"] for r in train_rows], "epochs": epochs, "max_steps": max_steps,
        "learning_rate": learning_rate, "lora_r": lora_r, "lora_alpha": lora_alpha, "load_in_4bit": four_bit,
        "train_loss": result.training_loss,
    }, indent=2))
    log.info("fold %d %s: trained on %d tasks, loss %.4f -> %s", fold, kind, len(train_rows), result.training_loss, out)
    return out


def predict(prompts_path: str | Path, kind: str, out_path: str | Path, model_id: str = DEFAULT_MODEL,
            adapter: str | Path | None = None, fold: int | None = None, max_new_tokens: int | None = None,
            batch_size: int = 8, load_in_4bit: bool | None = None) -> Path:
    """Generate rankings and append them to ``out_path`` (JSONL).

    With an adapter, ``fold`` must be the fold it was *not* trained on, and
    only that fold's tasks are predicted. Without an adapter (zero-shot base
    model) all tasks may be predicted. Tasks already in ``out_path`` are
    skipped, so one file collects all folds and interrupted runs resume.
    """
    import torch

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rows = load_jsonl(prompts_path)
    if adapter is not None:
        info = json.loads((Path(adapter) / "train_info.json").read_text())
        if fold is None or int(info["fold"]) != fold or info["kind"] != kind:
            raise ValueError(f"adapter {adapter} was trained for fold {info['fold']} ({info['kind']}); "
                             f"predict with --fold {info['fold']} --kind {info['kind']}")
        overlap = {r["task_id"] for r in rows if int(r["fold"]) == fold} & set(info["train_task_ids"])
        if overlap:
            raise ValueError(f"{len(overlap)} test tasks were in this adapter's training data")
    if fold is not None:
        rows = [r for r in rows if int(r["fold"]) == fold]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in load_jsonl(out)} if out.exists() else set()
    rows = [r for r in rows if r["task_id"] not in done]
    if not rows:
        log.info("nothing to predict for %s", out)
        return out

    model, tok, _ = _load_model(model_id, load_in_4bit)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    tok.padding_side = "left"  # decoder-only batched generation
    n_pipelines = rows[0]["target"].count(",") + 1
    max_new_tokens = max_new_tokens or 12 * n_pipelines + 32  # room for every name

    with open(out, "a") as f, torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [tok.apply_chat_template(_messages(r, kind), tokenize=False, add_generation_prompt=True)
                     for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for r, seq in zip(batch, gen):
                text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"task_id": r["task_id"], "fold": int(r["fold"]), "kind": kind,
                                    "model": model_id, "adapter": str(adapter) if adapter else None,
                                    "output": text}) + "\n")
            f.flush()
            log.info("predicted %d/%d", min(i + batch_size, len(rows)), len(rows))
    return out
