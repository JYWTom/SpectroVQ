"""Validating and resolving paths on the server typed or picked in the GUI."""

import os
from pathlib import Path

import gradio as gr

from gui import config


def host_path(path, what="file"):
    """Validate a path on the host machine typed or picked in the GUI."""
    path = os.path.expanduser((path or "").strip())
    if not path:
        raise gr.Error(f"Choose a {what} on the server first.")
    if not os.path.isfile(path):
        raise gr.Error(f"Not found on the server: {path}")
    return os.path.abspath(path)


def export_path(text, job, default_name):
    """Resolve an export path typed in the GUI, refusing to overwrite existing files.

    Empty text means default_name inside the run's job folder; an existing folder (or text ending
    in '/') gets default_name inside it. Missing parent folders are created.
    """
    text = os.path.expanduser((text or "").strip())
    if not text:
        return str(Path(job) / default_name)
    path = os.path.abspath(text)
    if os.path.isdir(path) or text.endswith(os.sep):
        path = os.path.join(path, default_name)
    if os.path.exists(path):
        raise gr.Error(f"Already exists on the server, not overwriting: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def export_dir(text):
    """Resolve an export folder typed in the GUI (None if empty), creating it if needed."""
    text = os.path.expanduser((text or "").strip())
    if not text:
        return None
    path = Path(text).resolve()
    if path.exists() and not path.is_dir():
        raise gr.Error(f"Not a folder on the server: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def browse_folder(text):
    """Folder the file browser should show for a typed path.

    A folder shows itself and an existing file its folder; anything else (a half-typed path)
    shows its deepest existing parent. Empty text shows --root.
    """
    text = os.path.expanduser((text or "").strip())
    if not text:
        return config.BROWSE_ROOT
    folder = os.path.abspath(text)
    while not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    return folder


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
