"""Command-line entry point.

    metapilot build    --config configs/pilot.yaml --out data/pilot
    metapilot build    --config configs/full.yaml --out data/full --shard 0/4   # one of 4 machines
    metapilot assemble --out data/full                                          # after merging parts/
    metapilot flaml    --config configs/full.yaml --meta data/full --budget 10
    metapilot evaluate --config configs/pilot.yaml --meta data/pilot --out runs/pilot --seeds 1 2
    metapilot evaluate ... --drop-group landmarkers      # landmarker ablation
    metapilot plots    --results runs/pilot
    metapilot pilot    --config configs/pilot.yaml        # build + evaluate + plots

    Phase 2 (LLM selector; training and prediction need the `llm` extra and a GPU):
    metapilot prompts     --config configs/pilot.yaml --meta data/pilot
    metapilot llm-train   --prompts data/pilot/llm/prompts.jsonl --fold 0 --kind numeric --out adapters/numeric_f0
    metapilot llm-predict --prompts data/pilot/llm/prompts.jsonl --kind numeric --fold 0 \\
                          --adapter adapters/numeric_f0 --out data/pilot/llm/predictions/ft_numeric.jsonl

    metapilot recommend --meta data/pilot --csv my.csv --target label --verify 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="metapilot")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the meta-level dataset (performance matrix + meta-features)")
    b.add_argument("--config", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--shard", default="0/1", help="run only shard i of n, e.g. 0/4 (default: everything)")

    m = sub.add_parser("assemble", help="combine per-task results in <out>/parts into the meta-dataset CSVs")
    m.add_argument("--out", type=Path, required=True)

    f = sub.add_parser("flaml", help="run the FLAML baseline on the tasks of a meta-dataset")
    f.add_argument("--config", type=Path)
    f.add_argument("--meta", type=Path, required=True)
    f.add_argument("--budget", type=float, required=True, help="FLAML time budget in seconds per CV fold")
    f.add_argument("--shard", default="0/1")
    f.add_argument("--max-tasks", type=int, help="run on an evenly spread subset of tasks")

    e = sub.add_parser("evaluate", help="run the Phase 1 selectors with grouped outer CV")
    e.add_argument("--config", type=Path)
    e.add_argument("--meta", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--drop-group", action="append", default=[],
                   help="meta-feature group to leave out (repeatable), e.g. landmarkers")
    e.add_argument("--seeds", type=int, nargs="*", default=[],
                   help="extra seeds for repeated outer CV; results in summary_seeds.csv")

    p = sub.add_parser("plots", help="draw the report figures from evaluation results")
    p.add_argument("--results", type=Path, required=True)

    q = sub.add_parser("prompts", help="write Phase 2 LLM prompts (<meta>/llm/prompts.jsonl)")
    q.add_argument("--config", type=Path)
    q.add_argument("--meta", type=Path, required=True)

    t = sub.add_parser("llm-train", help="QLoRA fine-tune on the training folds of one outer fold (GPU)")
    t.add_argument("--prompts", type=Path, required=True)
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--kind", choices=["numeric", "semantic"], required=True)
    t.add_argument("--out", type=Path, required=True, help="adapter directory")
    t.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--max-steps", type=int, default=-1)

    g = sub.add_parser("llm-predict", help="generate rankings with the base model or a fold's adapter")
    g.add_argument("--prompts", type=Path, required=True)
    g.add_argument("--kind", choices=["numeric", "semantic"], required=True)
    g.add_argument("--out", type=Path, required=True,
                   help="<meta>/llm/predictions/<variant>.jsonl (one file collects all folds)")
    g.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    g.add_argument("--adapter", type=Path, help="adapter from llm-train; omit for the zero-shot base model")
    g.add_argument("--fold", type=int, help="outer fold to predict (required with --adapter)")
    g.add_argument("--batch-size", type=int, default=8)

    r = sub.add_parser("recommend", help="rank the portfolio for a CSV file")
    r.add_argument("--meta", type=Path, required=True)
    r.add_argument("--csv", type=Path, required=True)
    r.add_argument("--target", required=True)
    r.add_argument("--verify", type=int, default=0, help="also cross-validate the top-k pipelines")
    r.add_argument("--selector", default="lambdarank", choices=["lambdarank", "regressor", "knn"])

    a = sub.add_parser("pilot", help="build + evaluate + plots")
    a.add_argument("--config", type=Path)
    a.add_argument("--data", type=Path, default=Path("data/pilot"))
    a.add_argument("--out", type=Path, default=Path("runs/pilot"))

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("matplotlib", "flaml"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from . import experiment, metadataset, plots

    if args.cmd == "build":
        metadataset.build(Config.load(args.config), args.out, metadataset.parse_shard(args.shard))
    elif args.cmd == "assemble":
        metadataset.assemble(args.out)
    elif args.cmd == "flaml":
        from . import baselines

        baselines.run(Config.load(args.config), args.meta, args.budget,
                      metadataset.parse_shard(args.shard), args.max_tasks)
    elif args.cmd == "evaluate":
        summary = experiment.run(args.meta, args.out, Config.load(args.config), tuple(args.drop_group),
                                 tuple(args.seeds))
        print(experiment.format_summary(summary))
    elif args.cmd == "plots":
        for path in plots.make_all(args.results):
            print(path)
    elif args.cmd == "prompts":
        from . import llm

        print(llm.write_prompts(args.meta, Config.load(args.config)))
    elif args.cmd == "llm-train":
        from . import llm_train

        llm_train.train(args.prompts, args.fold, args.kind, args.out, args.model, epochs=args.epochs,
                        max_steps=args.max_steps)
    elif args.cmd == "llm-predict":
        from . import llm_train

        llm_train.predict(args.prompts, args.kind, args.out, args.model, args.adapter, args.fold,
                          batch_size=args.batch_size)
    elif args.cmd == "recommend":
        from .recommend import recommend_csv

        rec = recommend_csv(args.meta, args.csv, args.target, args.verify, args.selector)
        rows, cols, k = rec.shape
        print(f"{rows} rows, {cols} columns, {k} classes" +
              (f" ({rec.dropped_classes} classes with too few rows dropped)" if rec.dropped_classes else ""))
        print("\nRecommended pipelines:\n" + rec.ranking.head(10).to_string(index=False))
        print("\nMost similar tasks in the meta-dataset:\n" + rec.neighbours.to_string(index=False))
        if rec.verified is not None:
            print("\nMeasured on this dataset:\n" + rec.verified.to_string(index=False))
    elif args.cmd == "pilot":
        cfg = Config.load(args.config)
        metadataset.build(cfg, args.data)
        summary = experiment.run(args.data, args.out, cfg)
        print(experiment.format_summary(summary))
        plots.make_all(args.out)


if __name__ == "__main__":
    main()
