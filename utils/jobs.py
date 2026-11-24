"""Sessions run in-process, in a thread, with a live log (the GUI's runs, except training).

They are started and recorded by the same calls as every other run (utils.sessions:
open_session and recording), so an in-process run and a run of main.py or train.py, in the
foreground or with --background, end up the same. Training runs are not in-process:
utils.training starts them as train.py --background.
"""

import contextlib
import io
import threading
import time
import traceback
from pathlib import Path

from utils.sessions import JOB_LOG, is_running, open_session, recording

# Sessions still running in this process -> their live log buffer (None until it starts).
ACTIVE_JOBS = {}


def new_job(root, kind, compile_mode="none", **params):
    """Open the session of an in-process run in root, which loaded its model with
    compile_mode (the run's own choice)."""
    job = open_session(kind, params, root=root, compile=compile_mode)
    ACTIVE_JOBS[str(job)] = None
    return job


def job_running(job, meta):
    """Whether a session is still running, in this process or another one."""
    return is_running(job, meta, ACTIVE_JOBS)


def live_log(job):
    """A running session's log so far: its live buffer if it runs in this process, else its
    log file; None if neither exists yet."""
    buf = ACTIVE_JOBS.get(str(job))
    if buf is not None:
        return buf.get()
    log_file = Path(job) / JOB_LOG
    return log_file.read_text() if log_file.exists() else None


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


# The pipelines print to stdout and redirecting it is process-wide,
# so in-process jobs are serialized through this lock (and the GUI's one concurrency group).
_JOB_LOCK = threading.Lock()


def run_logged(fn, job=None, interval=0.5):
    """Run fn() in a thread, yielding (done, log_text, result, error) while it runs.

    With a session folder (job), fn runs inside its sessions.recording, as a command-line
    run does: the output goes to job.log as well, and the outcome is recorded from the
    thread itself, so it is saved even if the caller goes away mid-run.
    """
    buf = LogBuffer()
    if job is not None:
        ACTIVE_JOBS[str(job)] = buf  # lets a viewer re-attach to the run
    state = {"result": None, "error": None}

    def work():
        try:
            state["result"] = fn()
        except Exception:
            traceback.print_exc()  # into the log, as a command-line run shows it
            raise

    def target():
        with _JOB_LOCK:
            try:
                if job is None:
                    with (
                        contextlib.redirect_stdout(buf),
                        contextlib.redirect_stderr(buf),
                    ):
                        work()
                else:
                    with recording(job, lambda: state["result"], echo=buf):
                        work()
            except Exception as e:  # noqa: BLE001
                state["error"] = e
            finally:
                ACTIVE_JOBS.pop(str(job), None)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    while t.is_alive():
        time.sleep(interval)
        yield False, buf.get(), None, None
    yield True, buf.get(), state["result"], state["error"]
