"""The session repository: one folder per run, whoever started it.

Every run of main.py or train.py, in the foreground, with --background or from the GUI, is
recorded the same way, in <data dir>/gui_outputs/<YYYYmmdd-HHMMSS>_<kind>_<random>/
("session"):

    job.json  kind, settings, status, pid, start and finish time, output paths, error, and
              for training the settings a resume must match (RESUME_SETTINGS)
    job.log   the run's log

so the GUI's sessions sidebar and `main.py --list` show them all. Every run goes through the
same two calls, whoever starts it: open_session() starts or continues its session, and its
work runs inside recording(), which writes the log and records the outcome. A training session's
checkpoints, the resume checkpoint among them, stay in its output directory (utils.resume). The data dir is ~/.spectrovq,
or $SPECTROVQ_DATA_DIR, or the GUI's --data-dir.

The GUI, main.py, train.py (through utils.session_cli) and the tests all go through this
module. Like utils.resume it imports nothing from torch, Lightning or Gradio.
"""

import contextlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from utils.config import COMPILE_MODES  # re-exported for callers
from utils.resume import EXIT_PAUSED

try:
    import fcntl
except ImportError:  # Windows: updates are not serialized, as before
    fcntl = None

SESSIONS_DIR = "gui_outputs"  # named before the command line had sessions too
JOB_FILE = "job.json"
JOB_LOG = "job.log"
JOB_NAME = re.compile(r"^(\d{8}-\d{6})_([a-z]+)_")
# A stopped training session with a resume checkpoint can be continued.
RESUMABLE = ("paused", "interrupted", "failed")
# job.json key: the settings a training run started with, which a resume must match.
RESUME_SETTINGS = "resume_settings"


def default_data_dir():
    """This user's data dir unless the GUI's --data-dir says otherwise."""
    return Path(os.environ.get("SPECTROVQ_DATA_DIR", "~/.spectrovq")).expanduser()


def sessions_root(data_dir=None):
    """The folder holding the sessions, in data_dir or the default data dir. Readable by
    its owner only."""
    data = Path(data_dir or default_data_dir()).resolve()
    # Each level on its own: parents=True would create the data dir with the umask's
    # mode, and it later holds gui.log with the GUI's token URL.
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = data / SESSIONS_DIR
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def now():
    return datetime.now().isoformat(timespec="seconds")


def read_job(job):
    try:
        return json.loads((Path(job) / JOB_FILE).read_text())
    except (OSError, ValueError):
        return {}


def write_job(job, meta):
    path = Path(job) / JOB_FILE
    # A temp file of its own: two processes updating one session must not share one
    fd, tmp = tempfile.mkstemp(prefix=".job-", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(meta, indent=2))
    os.replace(tmp, path)  # atomic, so a crash never leaves half a record


@contextlib.contextmanager
def _updating(job):
    """Serialize read-modify-write updates of a session record between processes
    (the launching process's set_pid and the detached copy's finish_session)."""
    if fcntl is None:
        yield
        return
    with open(Path(job) / ".job.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def process_start(pid):
    """When a process started (Linux clock ticks since boot), or None where unknown.
    Tells a session's process from a later one that reused its PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        return int(stat.rpartition(")")[2].split()[19])  # field 22, starttime
    except (IndexError, ValueError):
        return None


def _owner(pid):
    return {"pid": pid, "pid_start": process_start(pid)}


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, but someone else's
    except (TypeError, OSError):
        return False
    return True


def new_session(root, kind, params, pid=None):
    """Create a running session of this process (or of pid) and return its folder."""
    Path(root).mkdir(exist_ok=True)
    job = Path(
        tempfile.mkdtemp(prefix=f"{datetime.now():%Y%m%d-%H%M%S}_{kind}_", dir=root)
    )
    write_job(
        job,
        {
            "kind": kind,
            "status": "running",
            "started": now(),
            # Lets every other process tell running from interrupted.
            **_owner(pid or os.getpid()),
            "params": params,
        },
    )
    return job


def reopen_session(job, pid=None, params=None):
    """Mark a stopped session as running again, for a resumed run. params: settings the
    resumed run may change (its compile mode), merged into the recorded ones."""
    with _updating(job):
        meta = read_job(job)
        for key in ("finished", "error", "outputs"):
            meta.pop(key, None)
        if params:
            meta["params"] = {**meta.get("params", {}), **params}
        write_job(job, {**meta, "status": "running", **_owner(pid or os.getpid())})


def update_job(job, **fields):
    """Set fields of a session's record, leaving the others as they are."""
    with _updating(job):
        write_job(job, {**read_job(job), **fields})


def set_pid(job, pid):
    """Hand a running session to the process that does its work."""
    with _updating(job):
        meta = read_job(job)
        if meta.get("status") == "running":  # not if that process already finished it
            write_job(job, {**meta, **_owner(pid)})


def finish_session(job, outputs=None, error=None, status=None):
    """Record a session's outcome: status, finish time, output paths and error.

    The status follows from the error unless given (a paused run is neither done nor failed).
    """
    if isinstance(outputs, (str, os.PathLike)):
        outputs = [outputs]
    with _updating(job):
        meta = read_job(job)
        meta.update(
            status=status or ("failed" if error else "done"),
            finished=now(),
            outputs=[str(p) for p in outputs or []],
        )
        if error:
            meta["error"] = str(error)
        write_job(job, meta)


def open_session(kind, params, continues=None, root=None, pid=None, compile="none"):
    """Start the session of a run of this process (or of pid) and return its folder.

    continues: settings identifying an earlier run (a resumed training run); its newest
    stopped session of this kind is continued, else a new session is created.
    root: the sessions folder, by default that of the default data dir.
    compile: the run's torch.compile mode (COMPILE_MODES), recorded as params["compile"].
        A resumed run records its own: it may compile differently from the paused one.
    """
    if compile not in COMPILE_MODES:
        raise ValueError(
            f"compile mode must be one of {COMPILE_MODES}, got {compile!r}"
        )
    root = sessions_root() if root is None else Path(root)
    job = continues and find_stopped(root, kind, **continues)
    if not job:
        return new_session(root, kind, {**params, "compile": compile}, pid)
    reopen_session(job, pid, params={"compile": compile})
    with open(Path(job) / JOB_LOG, "a") as log:
        log.write("--- resumed ---\n")
    return job


@contextlib.contextmanager
def recording(job, outputs=list, echo=None, log=True):
    """Run a session's work inside this. It records the outcome: done, failed with the
    error, or paused (exit code EXIT_PAUSED), with the output paths outputs() returns. A
    killed run records nothing and shows as interrupted.

    log: copy what this process prints to job.log, as well as to echo (default: where it
        goes now, the terminal). False where the output already goes to job.log (a
        detached --background copy).
    """
    with _teed(Path(job) / JOB_LOG, echo) if log else contextlib.nullcontext():
        try:
            yield
        except SystemExit as e:
            if e.code in (None, 0):
                finish_session(job, outputs())
            elif e.code == EXIT_PAUSED:
                finish_session(job, outputs(), status="paused")
            else:
                error = (
                    e.code if isinstance(e.code, str) else f"Exited with code {e.code}"
                )
                finish_session(job, outputs(), error)
            raise
        except BaseException as e:
            finish_session(job, outputs(), f"{type(e).__name__}: {e}")
            raise
        else:
            finish_session(job, outputs())


class _Tee:
    """A text stream that also writes to a session's log. A full disk or a removed log
    never stops the run."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, s):
        with contextlib.suppress(OSError, ValueError):
            self._log.write(s)
        return self._stream.write(s)

    def flush(self):
        with contextlib.suppress(OSError, ValueError):
            self._log.flush()
        self._stream.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding... are the stream's
        return getattr(self._stream, name)


@contextlib.contextmanager
def _teed(log_path, echo=None):
    """Send what this process prints to log_path (appended, owner-only) and to echo."""
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", buffering=1) as log:
        out, err = sys.stdout, sys.stderr
        sys.stdout = _Tee(echo or out, log)
        sys.stderr = _Tee(echo or err, log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = out, err


def is_running(job, meta, active=()):
    """Whether a session is still running: in this process (active: the folders of its own
    running sessions) or in another one."""
    if meta.get("status") != "running":
        return False
    if str(job) in active:
        return True
    pid = meta.get("pid")
    if pid in (None, os.getpid()) or not pid_alive(pid):
        return False
    # A live PID may belong to a later process that reused it
    recorded, current = meta.get("pid_start"), process_start(pid)
    return recorded is None or current is None or recorded == current


def session_status(job, meta, active=()):
    status = meta.get("status")
    if status == "running" and not is_running(job, meta, active):
        return "interrupted"  # its process stopped before the run finished
    if status:
        return status
    # Sessions from before job.json existed: all-empty files mean the run never finished.
    files = [p for p in Path(job).iterdir() if p.is_file()]
    return "empty" if all(p.stat().st_size == 0 for p in files) else "unknown"


def all_sessions(root, active=()):
    """(folder, record, start time, kind, status) of every session, newest first."""
    found = []
    root = Path(root)
    if root.is_dir():
        for job in sorted(root.iterdir(), reverse=True):
            m = JOB_NAME.match(job.name)
            if not (m and job.is_dir()):
                continue
            meta = read_job(job)
            when = datetime.strptime(m[1], "%Y%m%d-%H%M%S")
            found.append(
                (
                    job,
                    meta,
                    when,
                    meta.get("kind", m[2]),
                    session_status(job, meta, active),
                )
            )
    return found


def find_stopped(root, kind, **params):
    """The newest stopped session of a kind whose settings include params, or None."""
    for job, meta, _, found_kind, status in all_sessions(root):
        recorded = meta.get("params", {})
        if (
            found_kind == kind
            and status in RESUMABLE
            and all(recorded.get(k) == v for k, v in params.items())
        ):
            return job
    return None


def collapse_redraws(text):
    """Log text as a terminal would show it: a progress bar redrawn with \\r keeps only its
    latest state."""
    return "\n".join(line.rstrip("\r").rpartition("\r")[2] for line in text.split("\n"))


def last_log_line(job, width=100):
    """The latest line of a session's log, or "" if there is none yet."""
    path = Path(job) / JOB_LOG
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - 4096))
            text = f.read().decode(errors="replace")
    except OSError:
        return ""
    lines = collapse_redraws(text).split("\n")
    return next((ln.strip()[:width] for ln in reversed(lines) if ln.strip()), "")


def print_sessions(root=None, recent=10):
    """Print the running sessions and the latest few others, like `gui.py --list` does for
    servers."""
    root = sessions_root() if root is None else root
    found = all_sessions(root)
    running = [s for s in found if s[4] == "running"]
    shown = running + [s for s in found if s[4] != "running"][:recent]
    if not shown:
        print(f"No SpectroVQ sessions in {root}.")
        return
    if not running:
        print("No SpectroVQ sessions running. Latest:")
    for job, meta, when, kind, status in shown:
        source = meta.get("params", {}).get("input", "")
        print(f"{when:%m-%d %H:%M}  {kind:<10}  {status:<11}  {source}")
        if status == "running":
            print(f"    pid {meta.get('pid')}, log: {job / JOB_LOG}")
            progress = last_log_line(job)
            if progress:
                print(f"    now: {progress}")
        elif meta.get("error"):
            print(f"    error: {meta['error']}")
        else:
            for out in meta.get("outputs", []):
                print(f"    {out}")


# --- Describing one session (the GUI's sidebar, and anything else that shows sessions)


def session_kind(job, meta):
    """What kind of run a session is: its record's kind, else the kind in its folder name."""
    m = JOB_NAME.match(Path(job).name)
    return meta.get("kind") or (m[2] if m else Path(job).name)


def session_files(job, meta):
    """Output files of a session: its folder's files plus outputs written elsewhere."""
    files = [
        p
        for p in sorted(Path(job).iterdir())
        if p.is_file() and p.name not in (JOB_FILE, JOB_LOG)
    ]
    seen = {p.resolve() for p in files}
    files += [
        Path(p)
        for p in meta.get("outputs", [])
        if Path(p).is_file() and Path(p).resolve() not in seen
    ]
    return files


def existing_outputs(meta):
    """The recorded outputs of a session that are still there."""
    return [p for p in meta.get("outputs", []) if os.path.isfile(p)]


def session_duration(meta):
    """How long a finished session took, as "1h 5m" or "3m 20s"; None if unknown."""
    try:
        start = datetime.fromisoformat(meta["started"])
        end = datetime.fromisoformat(meta["finished"])
    except (KeyError, ValueError):
        return None
    secs = int((end - start).total_seconds())
    return (
        f"{secs // 3600}h {secs // 60 % 60}m"
        if secs >= 3600
        else f"{secs // 60}m {secs % 60}s"
    )


def read_log(job):
    """A session's log as a terminal shows it ("" if it has none)."""
    try:
        # A --background run logs straight to the file, progress bar redraws included.
        return collapse_redraws((Path(job) / JOB_LOG).read_text())
    except OSError:
        return ""


def resume_checkpoint(meta):
    """Path of a training session's resume checkpoint if it has one, else None. Goes by its
    output directory, which can be outside the session folder."""
    from utils.resume import ResumeStore

    output_dir = meta.get("params", {}).get("output_dir")
    if meta.get("kind") != "train" or not output_dir:
        return None
    return ResumeStore(output_dir).find()


def can_resume(job, meta, active=()):
    """Whether a session is a stopped training run with a checkpoint to continue from."""
    return (
        session_status(job, meta, active) in RESUMABLE
        and resume_checkpoint(meta) is not None
    )


def session_targets(job, meta, files=None, active=()):
    """What a session's outputs can be used for next: {mgf: newest spectra file, vqms2: newest
    archive, ckpt: newest checkpoint (each a path or None), resume: bool, params}. Empty files
    are never offered; a resume checkpoint is never offered as a model."""
    files = session_files(job, meta) if files is None else files

    def newest(*suffixes):
        found = [p for p in files if p.suffix in suffixes and p.stat().st_size > 0]
        return str(max(found, key=os.path.getmtime)) if found else None

    return {
        "mgf": newest(".mgf", ".mzML", ".mzml"),
        "vqms2": newest(".vqms2"),
        "ckpt": newest(".ckpt"),
        "resume": can_resume(job, meta, active),
        "params": meta.get("params", {}),
    }


def follow_session(job, active=(), interval=1.0):
    """Yield a running session's record about every interval seconds until it stops."""
    import time

    while is_running(job, read_job(job), active):
        yield read_job(job)
        time.sleep(interval)
