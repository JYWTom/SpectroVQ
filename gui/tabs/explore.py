"""Explore tab: reconstruct one spectrum with a chosen number of quantizers."""

import os
from types import SimpleNamespace

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from pyteomics import mgf

from gui.components.file_picker import host_file_picker
from gui.config import CODE_BITS, MZ_RANGE
from gui.model import load_model
from gui.paths import host_path
from gui.spectra import (
    cosine,
    count_spectra,
    mirror_plot,
    reconstruct_vector,
    spectrum_title,
    spectrum_to_vector,
    vector_peaks,
)


def explore_load(mgf_file):
    if not (mgf_file or "").strip() or os.path.isdir(
        os.path.expanduser(mgf_file.strip())
    ):
        return gr.update(maximum=1, value=0), "Choose an MGF file.", None
    path = host_path(mgf_file, "MGF file")
    n = count_spectra(path)
    if n == 0:
        raise gr.Error("No spectra (BEGIN IONS blocks) found in this file.")
    return (
        gr.update(minimum=0, maximum=max(n - 1, 1), value=0, step=1),
        f"{n} spectra loaded.",
        path,
    )


def explore_run(mgf_path, idx, n_q, ckpt, yaml_path, device):
    if not mgf_path:
        raise gr.Error("Choose an MGF file first.")
    model = load_model(ckpt, yaml_path, device)
    max_q = int(model.n_q)
    n_q = int(min(max(1, n_q), max_q))
    with mgf.IndexedMGF(mgf_path) as reader:
        idx = min(int(idx), len(reader) - 1)
        spec = reader[idx]
    mz, it = np.asarray(spec["m/z array"]), np.asarray(spec["intensity array"])
    vec, _ = spectrum_to_vector(mz, it)
    if float(vec.sum()) == 0:
        raise gr.Error(
            f"Spectrum {int(idx)} has no peaks in the {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z range."
        )
    target = vec.cpu().numpy()

    recon = reconstruct_vector(model, vec, n_q)
    rec_mz, rec_it = vector_peaks(recon)
    in_mz, in_it = vector_peaks(target)
    title = spectrum_title(spec, int(idx))
    fig = mirror_plot(
        in_mz,
        in_it,
        rec_mz,
        rec_it,
        "input (binned)",
        f"reconstruction, {n_q} quantizer(s)",
        title=title,
    )

    sims = []
    for q in range(1, max_q + 1):
        sims.append(cosine(target, reconstruct_vector(model, vec, q)))
    curve, ax = plt.subplots(figsize=(5.2, 3.6))
    qs = np.arange(1, max_q + 1)
    ax.plot(qs, sims, marker="o", color="#1f6feb")
    ax.scatter(
        [n_q],
        [sims[n_q - 1]],
        s=90,
        facecolors="none",
        edgecolors="#d1242f",
        linewidths=2,
        zorder=3,
    )
    ax.set_xlabel("Number of quantizers")
    ax.set_ylabel("Cosine similarity to input")
    ax.set_xticks(qs)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    curve.tight_layout()
    plt.close(curve)

    bits = n_q * model.quantizedlen * CODE_BITS
    raw_bytes = len(mz) * 8  # float32 m/z + float32 intensity
    in_range = int(((mz >= MZ_RANGE[0]) & (mz < MZ_RANGE[1])).sum())
    summary = (
        f"**{title}**\n\n"
        f"| | |\n|---|---|\n"
        f"| Peaks in input | {len(mz)} ({in_range} within {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z) |\n"
        f"| Peaks in reconstruction | {len(rec_mz)} |\n"
        f"| Cosine similarity (binned) | {sims[n_q - 1]:.4f} |\n"
        f"| Code size | {n_q} × {model.quantizedlen} codes × {CODE_BITS} bit = {bits // 8} bytes |\n"
        f"| Raw peak list (float32 pairs) | {raw_bytes} bytes (≈{raw_bytes / max(bits / 8, 1):.1f}× larger) |\n"
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
                    "MGF path on the server", "*.mgf", "/path/to/spectra.mgf"
                )
                info = gr.Markdown()
                idx = gr.Slider(0, 1, value=0, step=1, label="Spectrum index")
                q = gr.Slider(1, 12, value=3, step=1, label="Quantizers")
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
