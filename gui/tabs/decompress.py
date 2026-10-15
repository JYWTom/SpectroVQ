"""Decompress tab: .vqms2 + side files -> MGF (MGFDecompressor)."""

import os
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from gui.components.file_picker import host_file_picker
from gui.jobs import new_job_dir, run_logged
from gui.model import load_model
from gui.paths import export_path, host_path
from gui.sessions import SessionView, with_session_list
from gui.spectra import count_spectra
from gui.tabs.compress import compression_levels
from utils.mgfHandler.decompressor import MGFDecompressor


def decompress_run(
    vq_file,
    out_file,
    n_q,
    method,
    level,
    batch_size,
    stored_raw,
    ckpt,
    yaml_path,
    device,
):
    vq = host_path(vq_file, ".vqms2 file")
    if not vq.endswith(".vqms2"):
        raise gr.Error(f"Expected a .vqms2 file, got {vq}")
    # The decompressor finds its side files next to the .vqms2 by swapping the extension.
    base = vq[: -len(".vqms2")]
    if not os.path.exists(base + ".parquet"):
        raise gr.Error(f"Metadata file not found next to it: {base}.parquet")
    model = load_model(ckpt, yaml_path, device)
    default_name = f"{Path(vq).stem}_decompressed.mgf"
    if (out_file or "").strip():
        export_path(out_file, None, default_name)  # fail before creating a session
    job = new_job_dir(
        "decompress",
        input=vq,
        quantizers=int(n_q),
        method=method,
        level=int(level),
        batch_size=int(batch_size),
        stored_raw=stored_raw,
    )
    out_path = export_path(out_file, job, default_name)

    def work():
        print(
            f"Decompressing {vq} with {int(n_q)} quantizers, {method} level {int(level)}\n"
            f"Side files: {base}.parquet"
            + (
                f", {base}_compounded.pkl"
                if os.path.exists(base + "_compounded.pkl")
                else ""
            )
        )
        MGFDecompressor(
            model=model,
            inputFileName=vq,
            outputfilemgf=out_path,
            batch_size=int(batch_size),
            quantizer=int(n_q),
            stored_raw=stored_raw,
            **compression_levels(method, level),
        ).DecompressAll(verbose=1)
        return out_path

    yield ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log
            continue
        if err is not None:
            yield (
                log
                + (
                    f"\n❌ {err}\n\nIf this is a decoding error, check that the quantizer count, method "
                    f"and level match the ones used for compression."
                )
            )
            return
        yield (
            log
            + f"\n✅ Done. {count_spectra(result)} spectra written.\nSaved on the server: {result}"
        )


def build(model_inputs, sidebar_outputs):
    with gr.Tab("Decompress", id="decompress"):
        with gr.Row():
            with gr.Column(scale=1):
                file, _ = host_file_picker(".vqms2 path on the server", "*.vqms2")
                gr.Markdown(
                    "The `.parquet` (and `_compounded.pkl`, if any) must sit next to it "
                    "with the same name."
                )
                q = gr.Slider(
                    1,
                    12,
                    value=4,
                    step=1,
                    label="Quantizers (same as compression)",
                )
                with gr.Row():
                    method = gr.Dropdown(
                        ["gzip", "zlib", "zstd"], value="gzip", label="Method"
                    )
                    level = gr.Number(6, precision=0, label="Level")
                bs = gr.Number(32, precision=0, label="Batch size")
                raw = gr.Checkbox(
                    False, label="Also write raw (non-chimeric) reconstructions"
                )
                outfile, _ = host_file_picker(
                    "Export path on the server",
                    placeholder="empty = <session folder>/<name>_decompressed.mgf",
                    folders=True,
                )
                btn = gr.Button("Decompress", variant="primary")
            with gr.Column(scale=2):
                log = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True)
        btn.click(
            with_session_list(decompress_run),
            [file, outfile, q, method, level, bs, raw, *model_inputs],
            [log, *sidebar_outputs],
            concurrency_id="pipeline",
        )
    return SimpleNamespace(
        file=file,
        q=q,
        method=method,
        level=level,
        session=SessionView(
            log, {file: "input", q: "quantizers", method: "method", level: "level"}
        ),
    )
