"""MetaPilot demo: upload a CSV, pick the target column, get a ranked list of
pipelines with the evidence behind it.

    pip install -e ".[demo]"
    python app/demo.py --meta data/pilot          # or data/full once built

The ranking comes from the LightGBM LambdaRank meta-model trained on every
task in the meta-dataset. "Why" shows the most similar tasks in that
meta-dataset and what won on them. Ticking "verify" runs the top three
pipelines on the uploaded data with cross-validation, so the recommendation
can be checked against real accuracies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import pandas as pd
from sklearn import datasets as skds

from metapilot.recommend import Recommender
from metapilot.metafeatures import GROUPS

SAMPLES = {
    # These four are also parents in the pilot meta-dataset, so the model has
    # seen subsamples of them; use an uploaded CSV for an unseen dataset.
    "breast cancer (scikit-learn; in pilot training data)": skds.load_breast_cancer,
    "wine (scikit-learn; in pilot training data)": skds.load_wine,
}


def load_sample(name: str) -> pd.DataFrame:
    return SAMPLES[name](as_frame=True).frame


def read_table(file, sample: str | None) -> pd.DataFrame:
    if file is not None:
        return pd.read_csv(file if isinstance(file, str) else file.name)
    if sample:
        return load_sample(sample)
    raise gr.Error("Upload a CSV file or choose a sample dataset.")


def build(meta_dir: str) -> gr.Blocks:
    rec = Recommender(meta_dir)
    n_tasks, n_parents = len(rec.meta.perf), rec.meta.groups.nunique()

    def on_input(file, sample):
        try:
            df = read_table(file, sample)
        except gr.Error:
            return gr.update(choices=[], value=None), None
        return gr.update(choices=list(df.columns), value=df.columns[-1]), df.head(8)

    def on_run(file, sample, target, verify):
        df = read_table(file, sample)
        if target not in df.columns:
            raise gr.Error("Choose the target column.")
        try:
            r = rec.recommend(df.drop(columns=[target]), df[target], verify_k=3 if verify else 0)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        rows, cols, k = r.shape
        top = r.ranking.iloc[0]
        note = f" {r.dropped_classes} class(es) with fewer than {rec.cfg.min_class_count} rows were dropped." \
            if r.dropped_classes else ""
        summary = (f"**Top recommendation: `{top.pipeline}`** ({top.family})\n\n"
                   f"Dataset after cleaning: {rows} rows, {cols} columns, {k} classes.{note} "
                   f"Ranked by LightGBM LambdaRank trained on {n_tasks} tasks from {n_parents} parent datasets.")
        group_of = {f: g for g, fs in GROUPS.items() for f in fs}
        mf = pd.DataFrame({"meta-feature": r.metafeatures.index,
                           "group": [group_of[f] for f in r.metafeatures.index],
                           "value": r.metafeatures.round(4).to_numpy()})
        ranking = r.ranking.assign(score=r.ranking["score"].round(3))
        neighbours = r.neighbours.assign(distance=r.neighbours["distance"].round(2),
                                         best_accuracy=r.neighbours["best_accuracy"].round(3))
        verified = r.verified.round(3) if r.verified is not None else pd.DataFrame()
        return summary, ranking, neighbours, verified, mf

    with gr.Blocks(title="MetaPilot") as demo:
        gr.Markdown("# MetaPilot\nRanks candidate pipelines for a tabular classification dataset "
                    "from its meta-features.")
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="CSV file", file_types=[".csv"])
                sample = gr.Dropdown(list(SAMPLES), label="...or a sample dataset", value=None)
                target = gr.Dropdown([], label="Target column")
                verify = gr.Checkbox(label="Verify: run the top 3 pipelines on this data (slower)")
                run = gr.Button("Recommend", variant="primary")
                preview = gr.Dataframe(label="Preview", interactive=False)
            with gr.Column(scale=2):
                summary = gr.Markdown()
                with gr.Tab("Ranking"):
                    ranking = gr.Dataframe(interactive=False)
                with gr.Tab("Why: similar tasks"):
                    gr.Markdown("The training tasks closest to this dataset in standardised meta-feature space, "
                                "and the pipeline that won on each.")
                    neighbours = gr.Dataframe(interactive=False)
                with gr.Tab("Verified accuracy"):
                    verified = gr.Dataframe(interactive=False)
                with gr.Tab("Meta-features"):
                    mf = gr.Dataframe(interactive=False)

        file.change(on_input, [file, sample], [target, preview])
        sample.change(on_input, [file, sample], [target, preview])
        run.click(on_run, [file, sample, target, verify], [summary, ranking, neighbours, verified, mf])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("data/pilot"), help="meta-dataset built by `metapilot build`")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(str(args.meta)).launch(server_port=args.port)


if __name__ == "__main__":
    main()
