"""Denoise tab: MGF -> denoised MGF (utils.pipelines.denoise), with a before/after viewer."""

from types import SimpleNamespace

import gradio as gr

from gui import config
from gui.components.file_picker import host_file_picker
from gui.errors import shows_errors
from gui.model import load_model
from gui.sessions import SessionView, with_session_list
from utils import config as defaults
from utils import pipelines
from utils.jobs import new_job, run_logged
from utils.paths import check_export_path, export_path, host_path
from utils.plots import mirror_plot
from utils.spectra import read_spectrum


@shows_errors
def denoise_run(
    mgf_file,
    out_file,
    n_q,
    retain,
    threshold,
    compounded,
    ckpt,
    yaml_path,
    device,
    compile_mode,
):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device, compile_mode)
    default_name = pipelines.denoised_name(in_path)
    check_export_path(out_file, default_name)
    job = new_job(
        config.OUTPUT_ROOT,
        "denoise",
        compile_mode,
        input=in_path,
        quantizers=int(n_q),
        retain_original_peaks=retain,
        one_percent_threshold=threshold,
        compounded_spectra=compounded,
    )
    out_path = export_path(out_file, job, default_name)

    def work():
        print(f"Input:  {in_path}\nOutput: {out_path}\nQuantizers: {int(n_q)}")
        return pipelines.denoise(
            model,
            in_path,
            out_path,
            n_q,
            retain,
            threshold,
            compounded,
            num_workers=defaults.DENOISE_NUM_WORKERS,
        )

    yield "", gr.update(), None, None
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, gr.update(), None, None
            continue
        if err is not None:
            yield log + f"\n❌ {err}", gr.update(), None, None
            return
        n_in, n_out = pipelines.denoise_counts(in_path, result)
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
    a_mz, a_it, title = read_spectrum(in_path, int(idx))
    b_mz, b_it, _ = read_spectrum(out_path, int(idx))
    return mirror_plot(
        a_mz,
        a_it,
        b_mz,
        b_it,
        f"original ({len(a_mz)} peaks)",
        f"denoised ({len(b_mz)} peaks)",
        title=title,
    )


def build(model_inputs, sidebar_outputs):
    with gr.Tab("Denoise", id="denoise"):
        in_path, out_path = gr.State(None), gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                file, _ = host_file_picker("Input MGF path on the server", "*.mgf")
                q = gr.Slider(
                    defaults.MIN_QUANTIZERS,
                    defaults.MAX_QUANTIZERS,
                    value=defaults.DENOISE_QUANTIZERS,
                    step=1,
                    label="Quantizers",
                    info="Fewer = stronger denoising",
                )
                retain = gr.Checkbox(
                    defaults.RETAIN_ORIGINAL_PEAKS,
                    label=f"Keep original peaks outside {defaults.MZ_RANGE[0]}–{defaults.MZ_RANGE[1]} m/z",
                )
                thr = gr.Checkbox(
                    defaults.ONE_PERCENT_THRESHOLD,
                    label="Drop peaks below 1% of base peak",
                )
                comp = gr.Checkbox(
                    defaults.COMPOUNDED_SPECTRA, label="Reconstruct chimeric spectra"
                )
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
