"""Validating and resolving paths on the machine a run happens on (typed in the GUI or given on
the command line). Problems raise UserError."""

import os
from pathlib import Path

from utils.errors import UserError


def host_path(path, what="file"):
    """An existing file, as an absolute path."""
    path = os.path.expanduser((path or "").strip())
    if not path:
        raise UserError(f"Choose a {what} on the server first.")
    if not os.path.isfile(path):
        raise UserError(f"Not found on the server: {path}")
    return os.path.abspath(path)


def export_path(text, folder, default_name):
    """Resolve an export path, refusing to overwrite existing files.

    Empty text means default_name inside folder; an existing folder (or text ending in '/') gets
    default_name inside it. Missing parent folders are created.
    """
    text = os.path.expanduser((text or "").strip())
    if not text:
        return str(Path(folder) / default_name)
    path = os.path.abspath(text)
    if os.path.isdir(path) or text.endswith(os.sep):
        path = os.path.join(path, default_name)
    if os.path.exists(path):
        raise UserError(f"Already exists on the server, not overwriting: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def check_export_path(text, default_name):
    """Fail now, before a session is created, if a typed export path is unusable."""
    if (text or "").strip():
        export_path(text, None, default_name)


def export_dir(text):
    """Resolve an export folder (None if empty), creating it if needed."""
    text = os.path.expanduser((text or "").strip())
    if not text:
        return None
    path = Path(text).resolve()
    if path.exists() and not path.is_dir():
        raise UserError(f"Not a folder on the server: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def refuse_existing(folder, stem, suffixes):
    """Raise if folder (None: a new session folder, always empty) already has stem + a suffix."""
    if folder is None:
        return
    for suffix in suffixes:
        existing = Path(folder) / (stem + suffix)
        if existing.exists():
            raise UserError(
                f"Already exists on the server, not overwriting: {existing}"
            )


def browse_folder(text, root):
    """Folder a file browser should show for a typed path.

    A folder shows itself and an existing file its folder; anything else (a half-typed path)
    shows its deepest existing parent. Empty text shows root.
    """
    text = os.path.expanduser((text or "").strip())
    if not text:
        return root
    folder = os.path.abspath(text)
    while not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    return folder


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
