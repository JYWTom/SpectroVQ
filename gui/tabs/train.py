"""Train tab: starts train.py --background through utils.training, exactly as from the command
line, and follows its session (record and log) until the run ends."""

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from gui import config
from gui.components.file_picker import host_file_picker
from gui.errors import shows_errors
from gui.sessions import SessionView, file_listing, with_session_list
from utils import config as defaults
from utils import training
from utils.sessions import COMPILE_MODES, follow_session, read_job

ENDINGS = {
    "done": "✅ Finished",
    "paused": "⏸️ Paused. Use Resume training on this session in the sidebar.",
    "paused_early": "⏸️ Stopped before any resume checkpoint was written: nothing to resume.",
}


@shows_errors
def train_run(
    data_path,
    output_dir,
    batch_size,
    lr,
    max_epochs,
    num_workers,
    compile_mode,
    checkpoint_minutes,
):
    training.check_idle()
    params = training.training_params(
        data_path,
        output_dir,
        config.DATA_DIR,
        batch_size,
        lr,
        max_epochs,
        num_workers,
        compile_mode,
        checkpoint_minutes,
    )
    yield from _launch(params)


@shows_errors
def train_resume(job):
    """Continue a stopped training session from its resume checkpoint."""
    training.check_idle()
    yield from _launch(training.resume_params(job), resume=True)


def _launch(params, resume=False):
    """Yield (log, result text) from the run's start until it ends."""
    yield "$ " + " ".join(training.train_command(params, resume)) + "\n", gr.update()
    job, said, code = training.start(params, config.DATA_DIR, resume)
    if job is None:  # refused: no session was started
        yield said + f"\n❌ train.py exited with code {code}", gr.update()
        return
    head = said if said.endswith("\n") else said + "\n"
    yield head + training.training_log(job), gr.update()  # also refreshes the sidebar
    for _ in follow_session(job):
        yield head + training.training_log(job), gr.update()
    meta = read_job(job)
    end = (
        ENDINGS.get(training.outcome(job, meta))
        or f"❌ {meta.get('error') or 'Interrupted.'}"
    )
    ckpts = [Path(p) for p in meta.get("outputs", [])]
    listing = "\n".join(f"- `{c}`" for c in ckpts) or "_No checkpoints written._"
    yield (
        head + training.training_log(job) + f"\n{end}",
        f"**Checkpoints in `{params['output_dir']}`:**\n\n{listing}",
    )


def train_pause():
    if not training.pause():
        return "No training run in progress."
    return (
        "Pausing: saving after the current batch. During validation it exits without saving, "
        "and a resume continues from the last periodic save."
    )


def train_stop():
    if not training.stop():
        return "No training run in progress."
    return "Stopped. Progress since the last periodic save is discarded."


def build(sidebar_outputs):
    with gr.Tab("Train", id="train"):
        gr.Markdown(
            "Runs `train.py` in a separate process. The library is a `.feather` consensus library "
            "with columns `PrecursorMZ`, `CompoundName`, `Comment`, `m/z`, `intensity`, `Charge`. "
            "Checkpoints are monitored on `val_mean_loss` with early stopping."
        )
        with gr.Row():
            with gr.Column(scale=1):
                data, _ = host_file_picker(
                    "Library (.feather) path on the server",
                    "*.feather",
                    defaults.TRAIN_DATA_PATH,
                )
                out, _ = host_file_picker(
                    "Output directory",
                    placeholder=f"empty = <data dir>/{training.CHECKPOINTS_DIR}/<timestamp>",
                    folders=True,
                )
                with gr.Row():
                    bs = gr.Number(
                        defaults.TRAIN_BATCH_SIZE, precision=0, label="Batch size"
                    )
                    lr = gr.Number(defaults.LEARNING_RATE, label="Learning rate")
                with gr.Row():
                    ep = gr.Number(defaults.MAX_EPOCHS, precision=0, label="Max epochs")
                    nw = gr.Number(
                        defaults.NUM_WORKERS, precision=0, label="Data loader workers"
                    )
                with gr.Row():
                    compile_mode = gr.Dropdown(
                        list(COMPILE_MODES),
                        value=defaults.DEFAULT_COMPILE,
                        label="torch.compile",
                        info="CUDA only",
                    )
                    minutes = gr.Number(
                        defaults.CHECKPOINT_MINUTES,
                        label="Minutes between resume checkpoints",
                    )
                with gr.Row():
                    btn = gr.Button("Start training", variant="primary")
                    pause = gr.Button("Pause (save and exit)")
                    stop = gr.Button("Stop (discard unsaved)", variant="stop")
                stop_msg = gr.Markdown()
            with gr.Column(scale=2):
                log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)
                ckpts = gr.Markdown()
        btn.click(
            with_session_list(train_run),
            [data, out, bs, lr, ep, nw, compile_mode, minutes],
            [log, ckpts, *sidebar_outputs],
            concurrency_limit=1,
        )
        pause.click(train_pause, None, stop_msg)
        stop.click(train_stop, None, stop_msg)
    return SimpleNamespace(
        session=SessionView(log, {data: "input"}, ckpts, file_listing("Checkpoints")),
    )
