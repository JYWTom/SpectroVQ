"""How the sessions sidebar and the tabs show sessions (utils.sessions holds what they are)."""

import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import gradio as gr

from gui import config
from utils import sessions
from utils.jobs import ACTIVE_JOBS
from utils.paths import human_size
from utils.sessions import read_job

SESSION_HEADERS = ["When", "Job", "Status"]
STATUS_LABELS = {
    "running": "⏳ running",
    "paused": "⏸️ paused",
    "done": "✅ done",
    "failed": "❌ failed",
    "interrupted": "⚠️ interrupted",
    "empty": "⚠️ empty",
    "unknown": "– no record",
}


def list_sessions():
    """(table rows, session folders), newest first."""
    rows, jobs = [], []
    for job, _, when, kind, status in sessions.all_sessions(
        config.OUTPUT_ROOT, ACTIVE_JOBS
    ):
        rows.append([f"{when:%m-%d %H:%M}", kind, STATUS_LABELS[status]])
        jobs.append(str(job))
    return rows, jobs


def refresh_sessions():
    rows, jobs = list_sessions()
    return gr.update(value=rows or None), jobs


def with_session_list(job_fn):
    """Wrap a job generator so its event also outputs the sessions sidebar (table, folders):
    refreshed on its first update, which comes right after it creates its session folder, and
    again when it ends. The sidebar then reacts to the run starting and finishing."""

    @functools.wraps(job_fn)
    def run(*args):
        n = None
        for out in job_fn(*args):
            out = out if isinstance(out, tuple) else (out,)
            sidebar = refresh_sessions() if n is None else (gr.skip(), gr.skip())
            n = len(out)
            yield (*out, *sidebar)
        if n is not None:
            yield (*(gr.skip() for _ in range(n)), *refresh_sessions())

    return run


def session_view(job):
    """Details markdown, log text and the send-to-tab targets of one session."""
    job = Path(job)
    meta = read_job(job)
    status = sessions.session_status(job, meta, ACTIVE_JOBS)
    lines = [
        f"### {sessions.session_kind(job, meta)} · {STATUS_LABELS[status]}",
        f"`{job.name}`",
        "",
    ]
    if "started" in meta:
        took = sessions.session_duration(meta)
        lines.append(
            f"Started {meta['started'].replace('T', ' ')}"
            + (f" · took {took}" if took else "")
        )
    params = meta.get("params", {})
    # A list, not a table: long paths would squeeze a table's first column in the sidebar.
    for k, v in params.items():
        name = f"**{k.replace('_', ' ')}**"
        lines.append(f"- {name}<br>`{v}`" if os.sep in str(v) else f"- {name}: `{v}`")
    files = sessions.session_files(job, meta)
    if files:
        # A list, like the settings: long file names would squeeze a table's size column.
        lines += ["", "**Files**"]
        lines += [f"- `{p.name}` · {human_size(p.stat().st_size)}" for p in files]
    if meta.get("error"):
        lines += ["", f"❌ {meta['error']}"]
    targets = sessions.session_targets(job, meta, files, ACTIVE_JOBS)
    return "\n".join(lines), sessions.read_log(job), targets


def file_listing(title):
    """Formatter showing a finished session's output paths in a tab's result box."""

    def show(outs):
        return f"**{title}:**\n\n" + (
            "\n".join(f"- `{p}`" for p in outs) or "_None written._"
        )

    return show


class SessionView(NamedTuple):
    """How a tab shows a session of its kind: its log box, its inputs (component -> job.json
    setting) and, optionally, a result box with the formatter for the session's outputs."""

    log_box: gr.components.Component
    inputs: dict
    result_box: gr.components.Component | None = None
    show: Callable | None = None
