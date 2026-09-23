"""Command-line entry point.

    metapilot build   --config configs/pilot.yaml --out data/pilot
    metapilot evaluate --config configs/pilot.yaml --meta data/pilot --out runs/pilot
    metapilot evaluate ... --drop-group landmarkers      # landmarker ablation
    metapilot plots   --results runs/pilot
    metapilot pilot   --config configs/pilot.yaml        # all three steps
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

    e = sub.add_parser("evaluate", help="run the Phase 1 selectors with grouped outer CV")
    e.add_argument("--config", type=Path)
    e.add_argument("--meta", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--drop-group", action="append", default=[],
                   help="meta-feature group to leave out (repeatable), e.g. landmarkers")

    p = sub.add_parser("plots", help="draw the report figures from evaluation results")
    p.add_argument("--results", type=Path, required=True)

    a = sub.add_parser("pilot", help="build + evaluate + plots")
    a.add_argument("--config", type=Path)
    a.add_argument("--data", type=Path, default=Path("data/pilot"))
    a.add_argument("--out", type=Path, default=Path("runs/pilot"))

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    from . import experiment, metadataset, plots

    if args.cmd == "build":
        metadataset.build(Config.load(args.config), args.out)
    elif args.cmd == "evaluate":
        summary = experiment.run(args.meta, args.out, Config.load(args.config), tuple(args.drop_group))
        print(experiment.format_summary(summary))
    elif args.cmd == "plots":
        for path in plots.make_all(args.results):
            print(path)
    elif args.cmd == "pilot":
        cfg = Config.load(args.config)
        metadataset.build(cfg, args.data)
        summary = experiment.run(args.data, args.out, cfg)
        print(experiment.format_summary(summary))
        plots.make_all(args.out)


if __name__ == "__main__":
    main()
