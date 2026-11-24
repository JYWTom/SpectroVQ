"""Compress tab: MGF or mzML -> .vqms2 + .parquet (utils.pipelines.compress)."""

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from gui import config
from gui.components.file_picker import SPECTRA_GLOB, host_file_picker
from gui.errors import shows_errors
from gui.model import load_model
from gui.sessions import SessionView, file_listing, with_session_list
from utils import config as defaults
from utils import pipelines
from utils.config import codec_description
from utils.jobs import new_job, run_logged
from utils.paths import export_dir, host_path, human_size


@shows_errors
def compress_run(
    spectra_file,
    out_dir,
    n_q,
    method,
    level,
    batch_size,
    ckpt,
    yaml_path,
    device,
    compile_mode,
):
    in_path = host_path(spectra_file, "MGF or mzML file")
    model = load_model(ckpt, yaml_path, device, compile_mode)
    out_dir = export_dir(out_dir)
    pipelines.check_compress(in_path, out_dir)
    job = new_job(
        config.OUTPUT_ROOT,
        "compress",
        compile_mode,
        input=in_path,
        quantizers=int(n_q),
        method=method,
        level=int(level),
        batch_size=int(batch_size),
    )
    folder = out_dir or job
    coder = codec_description(method, level)

    def work():
        print(f"Compressing {in_path} with {int(n_q)} quantizers, {coder}")
        return pipelines.compress(
            model,
            in_path,
            pipelines.compressed_stem(in_path, folder),
            n_q,
            method,
            level,
            batch_size,
            verbose=1,
            num_workers=0,
        )

    yield "", ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, ""
            continue
        if err is not None:
            yield log + f"\n❌ {err}", ""
            return
        s = pipelines.compression_summary(in_path, result)
        # Same "name · size" list style as the sessions sidebar.
        rows = "\n".join(
            f"- `{Path(p).name}` · {human_size(size)}" for p, size in s["files"]
        )
        summary = (
            f"**Original**\n- `{Path(in_path).name}` · {human_size(s['original'][1])}\n\n"
            f"**Compressed**\n{rows}\n- **total** · **{human_size(s['total'])}** "
            f"(ratio **{s['ratio']:.1f}×**)\n\n"
            f"Decompress with **{int(n_q)} quantizers, {coder}** — "
            f"the bitstream has no header, so these must match. Keep all files together.\n\n"
            f"Saved on the server in `{folder}`."
        )
        yield log + "\n✅ Done.", summary


def build(model_inputs, sidebar_outputs):
    with gr.Tab("Compress", id="compress"):
        with gr.Row():
            with gr.Column(scale=1):
                file, _ = host_file_picker(
                    "Input MGF or mzML path on the server", SPECTRA_GLOB
                )
                gr.Markdown(
                    "Only MS2 spectra are compressed. An mzML keeps its metadata, so it can be "
                    "restored to mzML later."
                )
                q = gr.Slider(
                    defaults.MIN_QUANTIZERS,
                    defaults.MAX_QUANTIZERS,
                    value=defaults.QUANTIZERS,
                    step=1,
                    label="Quantizers",
                    info=f"Must match at decompression (CLI default: {defaults.QUANTIZERS})",
                )
                with gr.Row():
                    method = gr.Dropdown(
                        list(defaults.CODECS), value=defaults.GUI_CODEC, label="Method"
                    )
                    level = gr.Number(
                        defaults.GUI_CODEC_LEVEL, precision=0, label="Level"
                    )
                bs = gr.Number(
                    defaults.CODEC_BATCH_SIZE, precision=0, label="Batch size"
                )
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
