"""Denoise tab: MGF -> denoised MGF (MGFDenoiser), with a before/after viewer."""

from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from pyteomics import mgf

from gui.components.file_picker import host_file_picker
from gui.jobs import new_job_dir, run_logged
from gui.model import load_model
from gui.paths import export_path, host_path
from gui.sessions import SessionView, with_session_list
from gui.spectra import count_spectra, mirror_plot, spectrum_title
from utils.mgfHandler.denoiser import MGFDenoiser


def denoise_run(
    mgf_file, out_file, n_q, retain, threshold, compounded, ckpt, yaml_path, device
):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device)
    default_name = f"{Path(in_path).stem}_denoised.mgf"
    if (out_file or "").strip():
        export_path(out_file, None, default_name)  # fail before creating a session
    job = new_job_dir(
        "denoise",
        input=in_path,
        quantizers=int(n_q),
        retain_original_peaks=retain,
        one_percent_threshold=threshold,
        compounded_spectra=compounded,
    )
    out_path = export_path(out_file, job, default_name)

    def work():
        print(f"Input:  {in_path}\nOutput: {out_path}\nQuantizers: {int(n_q)}")
        MGFDenoiser(model=model, inputMGF=in_path, outputMGF=out_path).denoiseMGF(
            numQuantSingle=int(n_q),
            retainOriginalPeaks=retain,
            OnePercentThreshold=threshold,
            compoundedSpectra=compounded,
            num_workers=0,
        )
        return out_path

    yield "", gr.update(), None, None
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, gr.update(), None, None
            continue
        if err is not None:
            yield log + f"\n❌ {err}", gr.update(), None, None
            return
        n_in, n_out = count_spectra(in_path), count_spectra(result)
        log += (
            f"\n✅ Done. {n_in} input spectra → {n_out} output spectra "
            f"({n_out - n_in} extra chimeric spectra appended at the end).\n"
            f"Saved on the server: {result}"
        )
        yield (
            log,
            gr.update(minimum=0, maximum=max(n_in - 1, 1), value=0, step=1),
            in_path,
            result,
        )


def denoise_preview(in_path, out_path, idx):
    if not in_path or not out_path:
        return None
    with mgf.IndexedMGF(in_path) as r:
        idx = min(int(idx), len(r) - 1)
        a = r[idx]
    with mgf.IndexedMGF(out_path) as r:
        b = r[idx]
    return mirror_plot(
        a["m/z array"],
        a["intensity array"],
        b["m/z array"],
        b["intensity array"],
        f"original ({len(a['m/z array'])} peaks)",
        f"denoised ({len(b['m/z array'])} peaks)",
        title=spectrum_title(a, idx),
    )


def build(model_inputs, sidebar_outputs):
    with gr.Tab("Denoise", id="denoise"):
        in_path, out_path = gr.State(None), gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                file, _ = host_file_picker("Input MGF path on the server", "*.mgf")
                q = gr.Slider(
                    1,
                    12,
                    value=3,
                    step=1,
                    label="Quantizers",
                    info="Fewer = stronger denoising",
                )
                retain = gr.Checkbox(
                    True, label="Keep original peaks outside 150–1500 m/z"
                )
                thr = gr.Checkbox(True, label="Drop peaks below 1% of base peak")
                comp = gr.Checkbox(True, label="Reconstruct chimeric spectra")
                outfile, _ = host_file_picker(
                    "Export path on the server",
                    placeholder="empty = <session folder>/<name>_denoised.mgf",
                    folders=True,
                )
                btn = gr.Button("Denoise", variant="primary")
            with gr.Column(scale=2):
                log = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True)
                idx = gr.Slider(0, 1, value=0, step=1, label="Preview spectrum index")
                plot = gr.Plot(label="Original vs denoised")
        btn.click(
            with_session_list(denoise_run),
            [file, outfile, q, retain, thr, comp, *model_inputs],
            [log, idx, in_path, out_path, *sidebar_outputs],
            concurrency_id="pipeline",
        ).then(denoise_preview, [in_path, out_path, idx], plot)
        idx.release(denoise_preview, [in_path, out_path, idx], plot)
    return SimpleNamespace(
        session=SessionView(log, {file: "input", q: "quantizers"}),
    )
