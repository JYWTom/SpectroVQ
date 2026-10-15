"""Compress tab: MGF -> .vqms2 + .parquet (+ _compounded.pkl) (MGFCompressor)."""

import os
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from gui.components.file_picker import host_file_picker
from gui.jobs import new_job_dir, run_logged
from gui.model import load_model
from gui.paths import export_dir, host_path, human_size
from gui.sessions import SessionView, file_listing, with_session_list
from utils.mgfHandler.compressor import MGFCompressor


def compression_levels(method, level):
    """Keyword arguments selecting the entropy coder, shared with the decompressor."""
    level = int(level) if level is not None else None
    return {
        "gzip_compression_level": level if method == "gzip" else None,
        "zlib_compression_level": level if method == "zlib" else None,
        "zstd_compression_level": level if method == "zstd" else None,
    }


def compress_run(
    mgf_file, out_dir, n_q, method, level, batch_size, ckpt, yaml_path, device
):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device)
    out_dir = export_dir(out_dir)
    if out_dir is not None:
        for ext in (".vqms2", ".parquet", "_compounded.pkl"):
            existing = out_dir / (Path(in_path).stem + ext)
            if existing.exists():
                raise gr.Error(
                    f"Already exists on the server, not overwriting: {existing}"
                )
    job = new_job_dir(
        "compress",
        input=in_path,
        quantizers=int(n_q),
        method=method,
        level=int(level),
        batch_size=int(batch_size),
    )
    stem = str((out_dir or job) / Path(in_path).stem)

    def work():
        print(
            f"Compressing {in_path} with {int(n_q)} quantizers, {method} level {int(level)}"
        )
        compressor = MGFCompressor(
            model=model,
            mgfFilePath=in_path,
            OutputFileName=stem,
            batch_size=int(batch_size),
            quantizer=int(n_q),
            **compression_levels(method, level),
        )
        compressor.CompressAll(verbose=1, num_workers=0)
        compressor.packer._raw_fo.close()
        return [
            p
            for p in (stem + ".vqms2", stem + ".parquet", stem + "_compounded.pkl")
            if os.path.exists(p)
        ]

    yield "", ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, ""
            continue
        if err is not None:
            yield log + f"\n❌ {err}", ""
            return
        original = os.path.getsize(in_path)
        total = sum(os.path.getsize(p) for p in result)
        # Same "name · size" list style as the sessions sidebar.
        rows = "\n".join(
            f"- `{Path(p).name}` · {human_size(os.path.getsize(p))}" for p in result
        )
        summary = (
            f"**Original**\n- `{Path(in_path).name}` · {human_size(original)}\n\n"
            f"**Compressed**\n{rows}\n- **total** · **{human_size(total)}** "
            f"(ratio **{original / max(total, 1):.1f}×**)\n\n"
            f"Decompress with **{int(n_q)} quantizers, {method}, level {int(level)}** — "
            f"the bitstream has no header, so these must match. Keep all files together.\n\n"
            f"Saved on the server in `{out_dir or job}`."
        )
        yield log + "\n✅ Done.", summary


def build(model_inputs, sidebar_outputs):
    with gr.Tab("Compress", id="compress"):
        with gr.Row():
            with gr.Column(scale=1):
                file, _ = host_file_picker("Input MGF path on the server", "*.mgf")
                q = gr.Slider(
                    1,
                    12,
                    value=4,
                    step=1,
                    label="Quantizers",
                    info="Must match at decompression (CLI uses 4)",
                )
                with gr.Row():
                    method = gr.Dropdown(
                        ["gzip", "zlib", "zstd"], value="gzip", label="Method"
                    )
                    level = gr.Number(6, precision=0, label="Level")
                bs = gr.Number(32, precision=0, label="Batch size")
                outdir, _ = host_file_picker(
                    "Export folder on the server",
                    placeholder="empty = the session folder",
                    folders=True,
                )
                btn = gr.Button("Compress", variant="primary")
            with gr.Column(scale=2):
                log = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)
                summary = gr.Markdown()
        btn.click(
            with_session_list(compress_run),
            [file, outdir, q, method, level, bs, *model_inputs],
            [log, summary, *sidebar_outputs],
            concurrency_id="pipeline",
        )
    return SimpleNamespace(
        session=SessionView(
            log,
            {file: "input", q: "quantizers", method: "method", level: "level"},
            summary,
            file_listing("Compressed files"),
        ),
    )
