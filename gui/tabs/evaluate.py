"""Evaluate tab: SpectroVQ vs a TopN-peaks baseline (utils.evaluate), per-spectrum compression
ratio vs cosine."""

import os
from types import SimpleNamespace

import gradio as gr

from gui import config
from gui.components.file_picker import SPECTRA_GLOB, host_file_picker
from gui.errors import shows_errors
from gui.model import load_model
from gui.sessions import SessionView, file_listing, with_session_list
from utils import config as defaults
from utils import evaluate
from utils.config import MZ_RANGE
from utils.evaluate import MATCHED, STANDARD
from utils.jobs import new_job, run_logged
from utils.paths import export_dir, host_path, refuse_existing

NOTES = {
    MATCHED: (
        "Ratios are per spectrum against the peaks inside "
        f"{MZ_RANGE[0]}–{MZ_RANGE[1]} m/z (8 bytes per peak), before the entropy "
        "coder; TopN keeps as many binned peaks as that ratio allows; cosine is "
        "computed on square-rooted binned intensities."
    ),
    STANDARD: (
        "Ratios are per spectrum against its raw peak list (8 bytes per peak), "
        "before the entropy coder; cosine is computed on the binned "
        f"{MZ_RANGE[0]}–{MZ_RANGE[1]} m/z spectrum."
    ),
}


def summary_table(stats):
    lines = [
        "| Method | Setting | Size / spectrum | Median ratio | Median cosine "
        f"| Cosine ≥ {evaluate.GOOD_COSINE} |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s['method']} | {s['name']} | {s['size']} | {s['ratio']:.1f}× | "
        f"{s['cosine']:.3f} | {s['good']:.0%} |"
        for s in stats
    ]
    return "\n".join(lines)


def _ints(values):
    return ", ".join(map(str, values))


@shows_errors
def evaluate_run(
    spectra_file,
    out_dir,
    n_q_text,
    top_n_text,
    limit,
    layout,
    batch_size,
    ckpt,
    yaml_path,
    device,
    compile_mode,
    metric=STANDARD,
):
    in_path = host_path(spectra_file, "MGF or mzML file")
    model = load_model(ckpt, yaml_path, device, compile_mode)
    n_qs = evaluate.parse_ints(n_q_text, "Quantizers", 1, int(model.n_q))
    matched = metric == MATCHED
    top_ns = [] if matched else evaluate.parse_ints(top_n_text, "TopN peaks")
    limit = max(int(limit or 0), 0)
    out_dir = export_dir(out_dir)
    stem = evaluate.evaluation_stem(in_path)
    refuse_existing(out_dir, stem, evaluate.OUTPUT_SUFFIXES)
    job = new_job(
        config.OUTPUT_ROOT,
        "evaluate",
        compile_mode,
        input=in_path,
        quantizers=_ints(n_qs),
        top_n="matched" if matched else _ints(top_ns),
        max_spectra=limit,
        layout=layout,
        metric=metric,
    )
    base = str((out_dir or job) / stem)
    state = {}

    def work():
        baseline = "ratio-matched TopN" if matched else f"top {top_ns} peaks"
        print(
            f"Evaluating {in_path}: SpectroVQ with {n_qs} quantizer(s) vs {baseline} ({metric})"
            + (f", first {limit} spectra" if limit else "")
        )
        rows, fig, files = evaluate.run_evaluation(
            in_path, model, n_qs, top_ns, limit, batch_size, metric, layout, base
        )
        state.update(rows=rows, fig=fig)
        return files

    yield "", None, ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, gr.skip(), ""
            continue
        if err is not None:
            yield log + f"\n❌ {err}", None, ""
            return
        files = "\n".join(f"- `{p}`" for p in result if os.path.exists(p))
        summary = (
            summary_table(evaluate.summary(state["rows"], model, matched))
            + f"\n\n{NOTES[metric]}\n\n**Saved on the server:**\n"
            + files
        )
        yield log + "\n✅ Done.", state["fig"], summary


def build(model_inputs, sidebar_outputs):
    with gr.Tab("Evaluate", id="evaluate"):
        gr.Markdown(
            "Benchmark SpectroVQ against a simple baseline that keeps only the N most intense "
            "peaks: per-spectrum compression ratio vs cosine similarity to the input."
        )
        with gr.Row():
            with gr.Column(scale=1):
                file, _ = host_file_picker(
                    "MGF or mzML path on the server", SPECTRA_GLOB
                )
                n_q = gr.Textbox(
                    _ints(defaults.EVAL_QUANTIZERS),
                    label="SpectroVQ quantizers",
                    info="Comma-separated; each is one setting",
                )
                metric = gr.Radio(
                    list(evaluate.METRICS),
                    value=STANDARD,
                    label="Metric",
                    info="The second matches the lab's plot_similarity_vs_compression.py",
                )
                top_n = gr.Textbox(
                    _ints(defaults.EVAL_TOP_N),
                    label="TopN peaks",
                    info="Comma-separated; each is one setting",
                )
                metric.change(
                    lambda m: gr.update(interactive=m != MATCHED),
                    metric,
                    top_n,
                    queue=False,
                )
                limit = gr.Number(
                    defaults.EVAL_MAX_SPECTRA,
                    precision=0,
                    label="Max spectra",
                    info="0 = whole file",
                )
                layout = gr.Radio(
                    ["Side by side", "Overlay"], value="Side by side", label="Plot"
                )
                bs = gr.Number(
                    defaults.EVAL_BATCH_SIZE, precision=0, label="Batch size"
                )
                outdir, _ = host_file_picker(
                    "Export folder on the server",
                    placeholder="empty = the session folder",
                    folders=True,
                )
                btn = gr.Button("Evaluate", variant="primary")
            with gr.Column(scale=3):
                fig = gr.Plot(label="Compression ratio vs similarity")
                summary = gr.Markdown()
                log = gr.Textbox(label="Log", lines=6, max_lines=6, autoscroll=True)
        btn.click(
            with_session_list(evaluate_run),
            [file, outdir, n_q, top_n, limit, layout, bs, *model_inputs, metric],
            [log, fig, summary, *sidebar_outputs],
            concurrency_id="pipeline",
        )
    return SimpleNamespace(
        session=SessionView(
            log,
            {
                file: "input",
                n_q: "quantizers",
                top_n: "top_n",
                limit: "max_spectra",
                layout: "layout",
                metric: "metric",
            },
            summary,
            file_listing("Evaluation files"),
        ),
    )
