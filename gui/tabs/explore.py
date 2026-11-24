"""Explore tab: reconstruct one spectrum with a chosen number of quantizers (utils.explore)."""

import os
from types import SimpleNamespace

import gradio as gr

from gui.components.file_picker import SPECTRA_GLOB, host_file_picker
from gui.errors import shows_errors
from gui.model import load_model
from utils import config as defaults
from utils.config import CODE_BITS, MZ_RANGE
from utils.errors import UserError
from utils.explore import explore_spectrum
from utils.paths import host_path
from utils.plots import mirror_plot, similarity_curve
from utils.spectra import count_spectra, is_mzml


@shows_errors
def explore_load(spectra_file):
    if not (spectra_file or "").strip() or os.path.isdir(
        os.path.expanduser(spectra_file.strip())
    ):
        return gr.update(maximum=1, value=0), "Choose an MGF or mzML file.", None
    path = host_path(spectra_file, "spectra file")
    n = count_spectra(path)
    if n == 0:
        raise UserError("No MS2 spectra found in this file.")
    return (
        gr.update(minimum=0, maximum=max(n - 1, 1), value=0, step=1),
        f"{n} MS2 spectra loaded." if is_mzml(path) else f"{n} spectra loaded.",
        path,
    )


@shows_errors
def explore_run(spectra_path, idx, n_q, ckpt, yaml_path, device, compile_mode):
    if not spectra_path:
        raise UserError("Choose an MGF or mzML file first.")
    model = load_model(ckpt, yaml_path, device, compile_mode)
    ex = explore_spectrum(model, spectra_path, idx, n_q)
    fig = mirror_plot(
        *ex.input_peaks,
        *ex.reconstruction,
        "input (binned)",
        f"reconstruction, {ex.quantizers} quantizer(s)",
        title=ex.title,
    )
    curve = similarity_curve(ex.similarities, ex.quantizers)
    size = int(ex.code_bytes)
    summary = (
        f"**{ex.title}**\n\n"
        f"| | |\n|---|---|\n"
        f"| Peaks in input | {ex.peaks} ({ex.peaks_in_range} within {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z) |\n"
        f"| Peaks in reconstruction | {len(ex.reconstruction[0])} |\n"
        f"| Cosine similarity (binned) | {ex.similarity:.4f} |\n"
        f"| Code size | {ex.quantizers} × {model.quantizedlen} codes × {CODE_BITS} bit = {size} bytes |\n"
        f"| Raw peak list (float32 pairs) | {ex.raw_bytes} bytes (≈{ex.raw_bytes / max(ex.code_bytes, 1):.1f}× larger) |\n"
    )
    return fig, curve, summary


def build(model_inputs):
    with gr.Tab("Explore", id="explore"):
        gr.Markdown(
            "Reconstruct one spectrum with a chosen number of quantizers and see how similarity "
            "grows as quantizers are added."
        )
        path = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                file, on_pick = host_file_picker(
                    "MGF or mzML path on the server",
                    SPECTRA_GLOB,
                    "/path/to/spectra.mgf",
                )
                info = gr.Markdown()
                idx = gr.Slider(0, 1, value=0, step=1, label="Spectrum index")
                q = gr.Slider(
                    defaults.MIN_QUANTIZERS,
                    defaults.MAX_QUANTIZERS,
                    value=defaults.EXPLORE_QUANTIZERS,
                    step=1,
                    label="Quantizers",
                )
                btn = gr.Button("Reconstruct", variant="primary")
            with gr.Column(scale=3):
                plot = gr.Plot(label="Input vs reconstruction")
                with gr.Row():
                    curve = gr.Plot(label="Similarity vs quantizers")
                    summary = gr.Markdown()
        file.submit(explore_load, file, [idx, info, path])
        file.blur(explore_load, file, [idx, info, path])
        on_pick(explore_load, [idx, info, path])
        args = [path, idx, q, *model_inputs]
        outs = [plot, curve, summary]
        btn.click(explore_run, args, outs)
        idx.release(explore_run, args, outs)
        q.release(explore_run, args, outs)
    return SimpleNamespace(file=file, info=info, idx=idx, path=path)
