"""Train tab: runs train.py as a subprocess and streams its log."""

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
    LogBuffer,
    finish_job,
    mirror_log,
    new_job_dir,
    read_job,
    write_job,
)
from gui.sessions import SessionView, file_listing, with_session_list

_TRAIN = {"proc": None}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def train_run(data_path, output_dir, batch_size, lr, max_epochs, num_workers):
    if _TRAIN["proc"] is not None and _TRAIN["proc"].poll() is None:
        raise gr.Error("A training run is already in progress.")
    data_path = os.path.expanduser((data_path or "").strip())
    if not data_path or not os.path.exists(data_path):
        raise gr.Error(f"Training library not found: {data_path!r}")
    data_path = os.path.abspath(data_path)
    params = {
        "input": data_path,
        "batch_size": int(batch_size),
        "learning_rate": lr,
        "max_epochs": int(max_epochs),
        "num_workers": int(num_workers),
    }
    # The session folder holds the record and log; checkpoints go there too unless an
    # output directory is given.
    job = new_job_dir("train", **params)
    output_dir = os.path.abspath(
        os.path.expanduser((output_dir or "").strip()) or str(job)
    )
    write_job(job, {**read_job(job), "params": {**params, "output_dir": output_dir}})
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--data_path",
        data_path,
        "--output_dir",
        output_dir,
        "--batch_size",
        str(int(batch_size)),
        "--learning_rate",
        str(lr),
        "--max_epochs",
        str(int(max_epochs)),
        "--num_workers",
        str(int(num_workers)),
    ]
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

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while proc.poll() is None:
        time.sleep(1)
        yield buf.get(), gr.update()
    reader.join(timeout=5)
    ckpts = sorted(Path(output_dir).glob("*.ckpt"), key=os.path.getmtime)
    status = (
        "✅ Finished"
        if proc.returncode == 0
        else f"❌ Exited with code {proc.returncode}"
    )
    listing = "\n".join(f"- `{c}`" for c in ckpts) or "_No checkpoints written._"
    finish_job(
        job,
        ckpts,
        None if proc.returncode == 0 else f"Exited with code {proc.returncode}",
        buf.get(),
    )
    yield buf.get() + f"\n{status}", f"**Checkpoints in `{output_dir}`:**\n\n{listing}"


def train_stop():
    proc = _TRAIN["proc"]
    if proc is None or proc.poll() is not None:
        return "No training run in progress."
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    return "Sent stop signal to the training process."


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
                with gr.Row():
                    btn = gr.Button("Start training", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                stop_msg = gr.Markdown()
            with gr.Column(scale=2):
                log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)
                ckpts = gr.Markdown()
        btn.click(
            with_session_list(train_run),
            [data, out, bs, lr, ep, nw],
            [log, ckpts, *sidebar_outputs],
            concurrency_limit=1,
        )
        stop.click(train_stop, None, stop_msg)
    return SimpleNamespace(
        session=SessionView(log, {data: "input"}, ckpts, file_listing("Checkpoints")),
    )
