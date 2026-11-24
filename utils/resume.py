"""The checkpoint folder of a training session, and the checks for starting or continuing
a run there.

A training run is a session (utils.sessions), whose job.json records the settings it started
with (RESUME_SETTINGS). Its checkpoints stay in its output directory, the checkpoint folder:
the best ones at the top, and in <output_dir>/resume/:

    resume.ckpt  Lightning checkpoint of the paused run (written by train.ResumeCheckpoint)
    train.lock   held by the process training here
    run.json     only from runs paused before the session recorded the settings; read on
                 --resume when the session has none

train.py, the GUI and the tests all go through this module. It imports nothing from torch or
Lightning, so the GUI can use it without loading the training stack.
"""

import contextlib
import hashlib
import json
import os
import shutil
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no lock, as before
    fcntl = None

# Exit code of a graceful pause (EX_TEMPFAIL): train.py can be continued with --resume.
EXIT_PAUSED = 75

RESUME_DIR = "resume"  # a subfolder: the sessions sidebar scans only the top level
CHECKPOINT_NAME = "resume"  # ModelCheckpoint's filename, without .ckpt
MANIFEST = "run.json"  # where older runs recorded their settings
LOCK = "train.lock"  # flock-ed by the process that is training here

_SAMPLE = 1 << 20  # bytes hashed at each end of the library


class ResumeError(Exception):
    """A run cannot start or continue in this output directory. The message says why."""


def library_fingerprint(path):
    """Size plus a hash of the first and last MiB: cheap on a multi-GB library, and unchanged
    when the file is copied or moved (unlike its mtime or path)."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(_SAMPLE))
        if size > _SAMPLE:
            f.seek(max(_SAMPLE, size - _SAMPLE))
            h.update(f.read(_SAMPLE))
    return {"size": size, "sha256_ends": h.hexdigest()}


class ResumeStore:
    """The checkpoint folder of one output directory."""

    def __init__(self, output_dir):
        self.dir = Path(output_dir).absolute() / RESUME_DIR
        self.checkpoint = self.dir / f"{CHECKPOINT_NAME}.ckpt"
        self.manifest = self.dir / MANIFEST
        self.lock_file = self.dir / LOCK
        self._held = None

    def find(self):
        """The resume checkpoint if there is one, else None."""
        return self.checkpoint if self.checkpoint.is_file() else None

    def begin(self, settings, resume, recorded=None):
        """Check that a run may start (resume=False) or continue (resume=True) here.

        settings: everything that must stay the same across a pause, as JSON-able values;
        the caller records them in the run's session. recorded: what the paused run's
        session recorded, or None. A resume compares against it (or, for an older run,
        against run.json) and raises ResumeError listing what differs. Returns warnings to
        print (empty if none). Also raises while another process is training in this output
        directory.
        """
        self._check_free()
        if not resume:
            if self.find():
                raise ResumeError(
                    f"{self.checkpoint} exists, so a paused run would be overwritten. "
                    "Continue it with --resume, or use another output directory (or "
                    "delete the file) to start over."
                )
            return []
        if not self.find():
            raise ResumeError(f"Nothing to resume: {self.checkpoint} does not exist.")
        if recorded is None:
            recorded = self._read_manifest()
        if recorded is None:
            return [
                (
                    "No session of the paused run records its settings (it was paused "
                    "before they were recorded, or its session was deleted): the settings "
                    "of this resume are not checked against the first run."
                )
            ]
        diffs = [
            f"  {k}: first run {recorded.get(k)!r}, now {settings.get(k)!r}"
            for k in sorted(recorded.keys() | settings.keys())
            if recorded.get(k) != settings.get(k)
        ]
        if diffs:
            raise ResumeError(
                "This resume uses other settings than the paused run started with:\n"
                + "\n".join(diffs)
                + "\nResume with the first run's settings, or delete "
                f"{self.dir} to start over."
            )
        return []

    def lock(self):
        """Hold this output directory until unlock() or the process exits, so a second
        run here (a --resume while this one trains, or the GUI) is refused."""
        if fcntl is None or self._held is not None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_file, "a")  # noqa: SIM115 - held for the process lifetime
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise ResumeError(self._busy_message()) from None
        self._held = handle

    def unlock(self):
        """Release lock(); exiting the process does the same."""
        if self._held is not None:
            self._held.close()
            self._held = None

    def _check_free(self):
        if fcntl is None or self._held is not None or not self.lock_file.exists():
            return
        with open(self.lock_file, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ResumeError(self._busy_message()) from None
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _busy_message(self):
        return (
            f"Another training run is using {self.dir.parent}. Wait for it to finish or "
            "pause it (kill -TERM <PID>), or use another output directory."
        )

    def discard(self):
        """Remove the resume checkpoint once training has finished (frees about 619 MB)."""
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read_manifest(self):
        try:
            return json.loads(self.manifest.read_text())["settings"]
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError) as e:
            raise ResumeError(f"{self.manifest} is unreadable ({e}).") from e
