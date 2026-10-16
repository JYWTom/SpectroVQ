"""Session (job) folders, their job.json records and logs, and running jobs with a live log."""

import contextlib
import io
import json
import os
import re
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from gui import config

# Every run gets a gui_outputs/<YYYYmmdd-HHMMSS>_<kind>_<random>/ folder ("session") holding its
# outputs, a job.json record (kind, settings, status, outputs) and its log (job.log).
JOB_FILE = "job.json"
JOB_LOG = "job.log"
JOB_NAME = re.compile(r"^(\d{8}-\d{6})_([a-z]+)_")
# Sessions still running in this process -> their live log buffer (None until it starts).
ACTIVE_JOBS = {}


def _now():
    return datetime.now().isoformat(timespec="seconds")


def read_job(job):
    try:
        return json.loads((Path(job) / JOB_FILE).read_text())
    except (OSError, ValueError):
        return {}


def write_job(job, meta):
    (Path(job) / JOB_FILE).write_text(json.dumps(meta, indent=2))


def new_job_dir(prefix, **params):
    config.OUTPUT_ROOT.mkdir(exist_ok=True)
    job = Path(
        tempfile.mkdtemp(
            prefix=f"{datetime.now():%Y%m%d-%H%M%S}_{prefix}_", dir=config.OUTPUT_ROOT
        )
    )
    write_job(
        job,
        {
            "kind": prefix,
            "status": "running",
            "started": _now(),
            "pid": os.getpid(),  # lets other app instances tell running from interrupted
            "params": params,
        },
    )
    ACTIVE_JOBS[str(job)] = None
    return job


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, OSError):
        return False
    return True


def job_running(job, meta):
    """Whether a session is still running, in this app process or another one."""
    if meta.get("status") != "running":
        return False
    return str(job) in ACTIVE_JOBS or (
        meta.get("pid") not in (None, os.getpid()) and _pid_alive(meta["pid"])
    )


def mirror_log(job, buf, alive, every=2.0):
    """Copy a running session's log to job.log every few seconds, so any app instance (or a
    reloaded page) can follow it."""

    def run():
        last = None
        while alive():
            text = buf.get()
            if text != last:
                with contextlib.suppress(OSError):
                    (Path(job) / JOB_LOG).write_text(text)
                last = text
            time.sleep(every)

    threading.Thread(target=run, daemon=True).start()


def finish_job(job, outputs=None, error=None, log=""):
    """Record a session's outcome: status, finish time, output paths, error and log."""
    if isinstance(outputs, (str, os.PathLike)):
        outputs = [outputs]
    meta = read_job(job)
    meta.update(
        status="failed" if error else "done",
        finished=_now(),
        outputs=[str(p) for p in outputs or []],
    )
    if error:
        meta["error"] = str(error)
    write_job(job, meta)
    if log:
        (Path(job) / JOB_LOG).write_text(log)
    ACTIVE_JOBS.pop(str(job), None)


class LogBuffer(io.TextIOBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.text = ""

    def write(self, s):
        # tqdm redraws its bar with \r; replace the current line instead of appending
        with self.lock:
            for i, chunk in enumerate(s.split("\r")):
                if i > 0:
                    self.text = self.text[: self.text.rfind("\n") + 1]
                self.text += chunk
        return len(s)

    def last_line(self):
        with self.lock:
            return self.text.rstrip("\n").rpartition("\n")[2]

    def replace_last_line(self, line):
        with self.lock:
            self.text = self.text.rstrip("\n").rpartition("\n")[0] + "\n" + line

    def get(self, max_chars=30000):
        with self.lock:
            return self.text[-max_chars:]


# The pipelines print to stdout and redirect_stdout is process-wide,
# so in-process jobs are serialized through this lock and one Gradio concurrency group.
_JOB_LOCK = threading.Lock()


def run_logged(fn, job=None):
    """Run fn() in a thread, yielding (done, log_text, result, error) while it runs.

    With a session folder (job), its outcome is recorded from the thread itself, so it is
    saved even if the browser disconnects mid-run.
    """
    buf = LogBuffer()
    if job is not None:
        ACTIVE_JOBS[str(job)] = buf  # lets the sessions sidebar re-attach to the run
    state = {"result": None, "error": None}

    def target():
        with (
            _JOB_LOCK,
            contextlib.redirect_stdout(buf),
            contextlib.redirect_stderr(buf),
        ):
            try:
                state["result"] = fn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                state["error"] = e
        if job is not None:
            finish_job(job, state["result"], state["error"], buf.get())

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if job is not None:
        mirror_log(job, buf, t.is_alive)
    while t.is_alive():
        time.sleep(0.5)
        yield False, buf.get(), None, None
    yield True, buf.get(), state["result"], state["error"]
