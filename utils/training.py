"""Starting, following, pausing and stopping training runs from another process (the GUI).

A run is started as `train.py --background`, exactly as from the command line, and followed
through its session (record and log): it records its own outcome, so it carries on and is
recorded whatever happens to the process that started it. One run at a time per process.
"""

import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils import config
from utils.errors import UserError
from utils.session_cli import SESSION_LINE
from utils.sessions import (
    JOB_LOG,
    can_resume,
    collapse_redraws,
    is_running,
    read_job,
    session_status,
)

ROOT = Path(__file__).resolve().parent.parent  # the project folder, where train.py is
CHECKPOINTS_DIR = (
    "checkpoints"  # in the data dir: where runs without an output dir save
)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_MAX_LOG = 30000

# The session of the run this process started, for pause and stop.
_CURRENT = {"job": None}


def current_run():
    """The session of the run this process started, if it is still running."""
    job = _CURRENT["job"]
    return job if job is not None and is_running(job, read_job(job)) else None


def check_idle():
    if current_run():
        raise UserError("A training run is already in progress.")


def default_output_dir(data_dir):
    return str(Path(data_dir) / CHECKPOINTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}")


def training_params(
    data_path,
    output_dir,
    data_dir,
    batch_size=config.TRAIN_BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    max_epochs=config.MAX_EPOCHS,
    num_workers=config.NUM_WORKERS,
    compile_mode=config.DEFAULT_COMPILE,
    checkpoint_minutes=config.CHECKPOINT_MINUTES,
):
    """The settings of a new run, as its session records them; creates its output dir
    (empty output_dir: a new one in data_dir)."""
    data_path = os.path.expanduser((data_path or "").strip())
    if not data_path or not os.path.exists(data_path):
        raise UserError(f"Training library not found: {data_path!r}")
    output_dir = os.path.expanduser((output_dir or "").strip()) or default_output_dir(
        data_dir
    )
    params = {
        "input": os.path.abspath(data_path),
        "batch_size": int(batch_size),
        "learning_rate": learning_rate,
        "max_epochs": int(max_epochs),
        "num_workers": int(num_workers),
        "compile": compile_mode,
        "checkpoint_minutes": checkpoint_minutes,
        "output_dir": os.path.abspath(output_dir),
    }
    os.makedirs(params["output_dir"], exist_ok=True)
    return params


def resume_params(job):
    """The settings to continue a stopped training session with."""
    meta = read_job(job) if job else {}
    if not meta or not can_resume(job, meta):
        raise UserError("This session has no paused run to resume.")
    return meta["params"]


def train_command(params, resume=False):
    """The train.py --background command line for a session's settings. train.py --resume
    continues the newest stopped session of the output directory, which wrote the checkpoint."""
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--background",
        "--data_path",
        params["input"],
        "--output_dir",
        params["output_dir"],
        "--batch_size",
        str(params["batch_size"]),
        "--learning_rate",
        str(params["learning_rate"]),
        "--max_epochs",
        str(params["max_epochs"]),
        "--num_workers",
        str(params["num_workers"]),
        # Sessions from before these settings existed don't record them.
        "--compile",
        params.get("compile", config.DEFAULT_COMPILE),
        "--checkpoint_minutes",
        str(params.get("checkpoint_minutes", config.CHECKPOINT_MINUTES)),
    ]
    return [*cmd, "--resume"] if resume else cmd


def start(params, data_dir, resume=False):
    """Run train.py --background; returns (session folder or None if it refused, its output,
    its exit code). The run's session lands in data_dir, where this process lists sessions."""
    started = subprocess.run(
        train_command(params, resume),
        cwd=params[
            "output_dir"
        ],  # Lightning writes lightning_logs/ to the working directory
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1", "SPECTROVQ_DATA_DIR": str(data_dir)},
        check=False,
    )
    said = started.stdout + started.stderr
    job = next(
        (
            Path(line[len(SESSION_LINE) :].strip())
            for line in said.splitlines()
            if line.startswith(SESSION_LINE)
        ),
        None,
    )
    if started.returncode != 0:
        job = None
    if job is not None:
        _CURRENT["job"] = job
    return job, said, started.returncode


def training_log(job):
    """A training session's log as a terminal shows it, without colours, its latest part only."""
    try:
        text = (Path(job) / JOB_LOG).read_text(errors="replace")
    except OSError:
        text = ""
    return _ANSI.sub("", collapse_redraws(text))[-_MAX_LOG:]


def outcome(job, meta):
    """How a training session ended: "done", "paused" (resumable), "paused_early" (stopped
    before any resume checkpoint) or "failed"."""
    if session_status(job, meta) == "done":
        return "done"
    if meta.get("status") == "paused":
        return "paused" if can_resume(job, meta) else "paused_early"
    return "failed"


def pause():
    """Ask the running training to save and exit; False if none is running."""
    job = current_run()
    if job is None:
        return False
    # The trainer only: its data loader workers must keep serving until it has saved.
    os.kill(read_job(job)["pid"], signal.SIGTERM)
    return True


def stop():
    """Kill the running training and its workers at once; False if none is running."""
    job = current_run()
    if job is None:
        return False
    # A --background run leads its own process group: the trainer and its workers.
    os.killpg(read_job(job)["pid"], signal.SIGKILL)
    return True
