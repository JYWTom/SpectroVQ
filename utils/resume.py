"""Where a paused training run lives, and the checks for starting or continuing one.

One ResumeStore per output directory, holding <output_dir>/resume/:

    resume.ckpt  Lightning checkpoint of the paused run (written by train.ResumeCheckpoint)
    run.json     the settings the run started with, checked on --resume

train.py, the GUI and the tests all go through this module. It imports nothing from torch or
Lightning, so the GUI can use it without loading the training stack.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

# Exit code of a graceful pause (EX_TEMPFAIL): train.py can be continued with --resume.
EXIT_PAUSED = 75

RESUME_DIR = "resume"  # a subfolder: the sessions sidebar scans only the top level
CHECKPOINT_NAME = "resume"  # ModelCheckpoint's filename, without .ckpt
MANIFEST = "run.json"

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
    """The paused-run state of one output directory."""

    def __init__(self, output_dir):
        self.dir = Path(output_dir).absolute() / RESUME_DIR
        self.checkpoint = self.dir / f"{CHECKPOINT_NAME}.ckpt"
        self.manifest = self.dir / MANIFEST

    def find(self):
        """The resume checkpoint if there is one, else None."""
        return self.checkpoint if self.checkpoint.is_file() else None

    def begin(self, settings, resume):
        """Check that a run may start (resume=False) or continue (resume=True) here.

        settings: everything that must stay the same across a pause, as JSON-able values.
        A fresh start records them; a resume compares against what was recorded and raises
        ResumeError listing what differs. Returns warnings to print (empty if none).
        """
        if not resume:
            if self.find():
                raise ResumeError(
                    f"{self.checkpoint} exists, so a paused run would be overwritten. "
                    "Continue it with --resume, or use another output directory (or "
                    "delete the file) to start over."
                )
            self._write_manifest(settings)
            return []
        if not self.find():
            raise ResumeError(f"Nothing to resume: {self.checkpoint} does not exist.")
        recorded = self._read_manifest()
        if recorded is None:
            return [
                (
                    f"{self.manifest} is missing (a run paused before settings were "
                    "recorded): the settings of this resume are not checked against the "
                    "first run."
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

    def discard(self):
        """Remove the paused-run state once training has finished (frees about 619 MB)."""
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read_manifest(self):
        try:
            return json.loads(self.manifest.read_text())["settings"]
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError) as e:
            raise ResumeError(f"{self.manifest} is unreadable ({e}).") from e

    def _write_manifest(self, settings):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "settings": settings}, indent=2))
        os.replace(tmp, self.manifest)  # never a half-written manifest
