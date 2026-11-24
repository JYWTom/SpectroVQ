"""Decompress tab: .vqms2 + side files -> MGF or mzML (utils.pipelines.decompress)."""

from types import SimpleNamespace

import gradio as gr

from gui import config
from gui.components.file_picker import host_file_picker
from gui.errors import shows_errors
from gui.model import load_model
from gui.sessions import SessionView, with_session_list
from utils import config as defaults
from utils import pipelines
from utils.config import codec_description
from utils.jobs import new_job, run_logged
from utils.paths import check_export_path, export_path, host_path
from utils.spectra import count_spectra


@shows_errors
def decompress_run(
    vq_file,
    out_file,
    n_q,
    method,
    level,
    batch_size,
    stored_raw,
    out_format,
    ckpt,
    yaml_path,
    device,
    compile_mode,
):
    vq = host_path(vq_file, ".vqms2 file")
    pipelines.check_decompress(vq, out_format, out_file, stored_raw)
    model = load_model(ckpt, yaml_path, device, compile_mode)
    default_name = pipelines.decompressed_name(vq, out_format)
    check_export_path(out_file, default_name)
    job = new_job(
        config.OUTPUT_ROOT,
        "decompress",
        compile_mode,
        input=vq,
        quantizers=int(n_q),
        method=method,
        level=int(level),
        batch_size=int(batch_size),
        stored_raw=stored_raw,
        output_format=out_format,
    )
    out_path = export_path(out_file, job, default_name)

    def work():
        print(
            f"Decompressing {vq} with {int(n_q)} quantizers, {codec_description(method, level)}\n"
            f"Side files: {', '.join(pipelines.side_files(vq))}"
        )
        return pipelines.decompress(
            model, vq, out_path, n_q, method, level, batch_size, stored_raw, verbose=1
        )

    yield ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log
            continue
        if err is not None:
            yield (
                log
                + f"\n❌ {err}\n\nIf this is a decoding error, check that the quantizer count, "
                "method and level match the ones used for compression."
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
                    "The `.parquet` must sit next to it with the same name (older "
                    "archives also have a `_compounded.pkl`)."
                )
                q = gr.Slider(
                    defaults.MIN_QUANTIZERS,
                    defaults.MAX_QUANTIZERS,
                    value=defaults.QUANTIZERS,
                    step=1,
                    label="Quantizers (same as compression)",
                )
                with gr.Row():
                    method = gr.Dropdown(
                        # "none": archives compressed without an entropy coder (the
                        # command line's default when --compression_method is omitted)
                        [*defaults.CODECS, defaults.NO_CODEC],
                        value=defaults.GUI_CODEC,
                        label="Method",
                    )
                    level = gr.Number(
                        defaults.GUI_CODEC_LEVEL, precision=0, label="Level"
                    )
                bs = gr.Number(
                    defaults.CODEC_BATCH_SIZE, precision=0, label="Batch size"
                )
                raw = gr.Checkbox(
                    defaults.STORED_RAW,
                    label="Also write raw (non-chimeric) reconstructions",
                )
                out_format = gr.Dropdown(
                    ["mgf", "mzML"],
                    value="mgf",
                    label="Output format",
                    info="mzML needs an archive compressed from mzML",
                )
                outfile, _ = host_file_picker(
                    "Export path on the server",
                    placeholder="empty = <session folder>/<name>_decompressed.<format>",
                    folders=True,
                )
                btn = gr.Button("Decompress", variant="primary")
            with gr.Column(scale=2):
                log = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True)
        btn.click(
            with_session_list(decompress_run),
            [file, outfile, q, method, level, bs, raw, out_format, *model_inputs],
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
