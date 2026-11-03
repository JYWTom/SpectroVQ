"""Train tab: runs train.py as a subprocess and streams its log."""

import contextlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from gui.components.file_picker import host_file_picker
from gui.config import ROOT
from gui.jobs import (
    ACTIVE_JOBS,
    JOB_LOG,
    LogBuffer,
    finish_job,
    mirror_log,
    new_job_dir,
    read_job,
    reopen_job,
    write_job,
)
from gui.sessions import (
    SessionView,
    file_listing,
    resume_checkpoint,
    session_status,
    with_session_list,
)

# train.py's EXIT_PAUSED; train.py itself isn't imported, as that loads the whole training stack.
EXIT_PAUSED = 75

_TRAIN = {"proc": None}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _check_not_running():
    if _TRAIN["proc"] is not None and _TRAIN["proc"].poll() is None:
        raise gr.Error("A training run is already in progress.")


def train_run(
    data_path, output_dir, batch_size, lr, max_epochs, num_workers, deterministic
):
    _check_not_running()
    data_path = os.path.expanduser((data_path or "").strip())
    if not data_path or not os.path.exists(data_path):
        raise gr.Error(f"Training library not found: {data_path!r}")
    data_path = os.path.abspath(data_path)
    output_dir = os.path.expanduser((output_dir or "").strip())
    # Checked here too, so no failed session is created that would offer the other run's resume.
    if output_dir and (Path(output_dir) / "resume" / "resume.ckpt").exists():
        raise gr.Error(
            f"{output_dir} holds a paused run. Resume it from its session, or use another "
            "output directory."
        )
    params = {
        "input": data_path,
        "batch_size": int(batch_size),
        "learning_rate": lr,
        "max_epochs": int(max_epochs),
        "num_workers": int(num_workers),
        "deterministic": bool(deterministic),
    }
    # The session folder holds the record and log; checkpoints go there too unless an
    # output directory is given.
    job = new_job_dir("train", **params)
    params["output_dir"] = os.path.abspath(output_dir or str(job))
    write_job(job, {**read_job(job), "params": params})
    os.makedirs(params["output_dir"], exist_ok=True)
    yield from _launch(job, params)


def train_resume(job):
    """Continue a stopped training session from its resume checkpoint, in the same session."""
    _check_not_running()
    meta = read_job(job) if job else {}
    if (
        not meta
        or session_status(job, meta) not in ("paused", "interrupted", "failed")
        or resume_checkpoint(meta) is None
    ):
        raise gr.Error("This session has no paused run to resume.")
    reopen_job(job)
    yield from _launch(job, meta["params"], resume=True)


def _launch(job, params, resume=False):
    """Run train.py for a session, yielding (log, result text) until it ends. The outcome is
    recorded from a thread, so it is saved even if the browser disconnects mid-run."""
    output_dir = params["output_dir"]
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--data_path",
        params["input"],
        "--output_dir",
        output_dir,
        "--batch_size",
        str(params["batch_size"]),
        "--learning_rate",
        str(params["learning_rate"]),
        "--max_epochs",
        str(params["max_epochs"]),
        "--num_workers",
        str(params["num_workers"]),
    ]
    if params.get("deterministic"):
        cmd.append("--deterministic")
    if resume:
        cmd.append("--resume")
    proc = subprocess.Popen(
        cmd,
        cwd=output_dir,  # Lightning writes lightning_logs/ to the working directory
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    _TRAIN["proc"] = proc
    buf = LogBuffer()
    if resume:
        # job.log gets overwritten by this run's log, so that log has to start with the old one.
        with contextlib.suppress(OSError):
            buf.write((Path(job) / JOB_LOG).read_text().rstrip("\n") + "\n")
        buf.write("--- resumed ---\n")
    ACTIVE_JOBS[str(job)] = buf
    buf.write("$ " + " ".join(cmd) + "\n")
    mirror_log(job, buf, lambda: proc.poll() is None)
    yield buf.get(), gr.update()  # first update right away (also refreshes the sidebar)

    def pump():
        assert proc.stdout is not None  # started with stdout=PIPE
        for line in proc.stdout:
            line = _ANSI.sub("", line)
            # Lightning redraws its progress bar line by line; keep only the latest one
            if "━" in line and "━" in buf.last_line():
                buf.replace_last_line(line)
            else:
                buf.write(line)

    outcome = {}

    def supervise():
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        code = proc.wait()
        reader.join(timeout=5)
        ckpts = sorted(Path(output_dir).glob("*.ckpt"), key=os.path.getmtime)
        paused = code == EXIT_PAUSED
        error = None if code == 0 or paused else f"Exited with code {code}"
        finish_job(job, ckpts, error, buf.get(), status="paused" if paused else None)
        outcome.update(code=code, ckpts=ckpts)

    watcher = threading.Thread(target=supervise, daemon=True)
    watcher.start()
    while watcher.is_alive():
        time.sleep(1)
        yield buf.get(), gr.update()
    code, ckpts = outcome["code"], outcome["ckpts"]
    if code == 0:
        status = "✅ Finished"
    elif code == EXIT_PAUSED:
        status = (
            "⏸️ Paused. Use Resume training on this session in the sidebar."
            if resume_checkpoint(read_job(job))
            else "⏸️ Stopped before any resume checkpoint was written: nothing to resume."
        )
    else:
        status = f"❌ Exited with code {code}"
    listing = "\n".join(f"- `{c}`" for c in ckpts) or "_No checkpoints written._"
    yield buf.get() + f"\n{status}", f"**Checkpoints in `{output_dir}`:**\n\n{listing}"


def train_pause():
    proc = _TRAIN["proc"]
    if proc is None or proc.poll() is not None:
        return "No training run in progress."
    # The trainer only: its data loader workers must keep serving until it has saved.
    proc.send_signal(signal.SIGTERM)
    return (
        "Pausing: saving after the current batch. During validation it exits without saving, "
        "and a resume continues from the last periodic save."
    )


def train_stop():
    proc = _TRAIN["proc"]
    if proc is None or proc.poll() is not None:
        return "No training run in progress."
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
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
                    "library.feather",
                )
                out, _ = host_file_picker(
                    "Output directory",
                    placeholder="empty = gui_outputs/<timestamp>_train",
                    folders=True,
                )
                with gr.Row():
                    bs = gr.Number(32, precision=0, label="Batch size")
                    lr = gr.Number(1e-4, label="Learning rate")
                with gr.Row():
                    ep = gr.Number(50, precision=0, label="Max epochs")
                    nw = gr.Number(4, precision=0, label="Data loader workers")
                det = gr.Checkbox(
                    False,
                    label="Deterministic",
                    info="For exact resumes on GPU. Slower; needs the same GPU model and "
                    "PyTorch version when resuming.",
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
            [data, out, bs, lr, ep, nw, det],
            [log, ckpts, *sidebar_outputs],
            concurrency_limit=1,
        )
        pause.click(train_pause, None, stop_msg)
        stop.click(train_stop, None, stop_msg)
    return SimpleNamespace(
        session=SessionView(log, {data: "input"}, ckpts, file_listing("Checkpoints")),
    )
