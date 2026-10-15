"""Listing and describing past sessions for the sessions sidebar."""

import functools
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import gradio as gr

from gui.config import OUTPUT_ROOT
from gui.jobs import JOB_FILE, JOB_LOG, JOB_NAME, job_running, read_job
from gui.paths import human_size

SESSION_HEADERS = ["When", "Job", "Status"]
STATUS_LABELS = {
    "running": "⏳ running",
    "done": "✅ done",
    "failed": "❌ failed",
    "interrupted": "⚠️ interrupted",
    "empty": "⚠️ empty",
    "unknown": "– no record",
}


def session_status(job, meta):
    status = meta.get("status")
    if status == "running" and not job_running(job, meta):
        return "interrupted"  # the app stopped before the run finished
    if status:
        return status
    # Sessions from before job.json existed: all-empty files mean the run never finished.
    files = [p for p in Path(job).iterdir() if p.is_file()]
    return "empty" if all(p.stat().st_size == 0 for p in files) else "unknown"


def list_sessions():
    """(table rows, session folders), newest first."""
    rows, jobs = [], []
    if OUTPUT_ROOT.is_dir():
        for job in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
            m = JOB_NAME.match(job.name)
            if not (m and job.is_dir()):
                continue
            meta = read_job(job)
            when = datetime.strptime(m[1], "%Y%m%d-%H%M%S")
            status = session_status(job, meta)
            rows.append(
                [f"{when:%m-%d %H:%M}", meta.get("kind", m[2]), STATUS_LABELS[status]]
            )
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
            sessions = refresh_sessions() if n is None else (gr.skip(), gr.skip())
            n = len(out)
            yield (*out, *sessions)
        if n is not None:
            yield (*(gr.skip() for _ in range(n)), *refresh_sessions())

    return run


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


def _duration(meta):
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


def session_view(job):
    """Details markdown, log text and the send-to-tab targets of one session."""
    job = Path(job)
    meta = read_job(job)
    m = JOB_NAME.match(job.name)
    kind = meta.get("kind") or (m[2] if m else job.name)
    lines = [
        f"### {kind} · {STATUS_LABELS[session_status(job, meta)]}",
        f"`{job.name}`",
        "",
    ]
    if "started" in meta:
        took = _duration(meta)
        lines.append(
            f"Started {meta['started'].replace('T', ' ')}"
            + (f" · took {took}" if took else "")
        )
    params = meta.get("params", {})
    # A list, not a table: long paths would squeeze a table's first column in the sidebar.
    for k, v in params.items():
        name = f"**{k.replace('_', ' ')}**"
        lines.append(f"- {name}<br>`{v}`" if os.sep in str(v) else f"- {name}: `{v}`")
    files = session_files(job, meta)
    if files:
        # A list, like the settings: long file names would squeeze a table's size column.
        lines += ["", "**Files**"]
        lines += [f"- `{p.name}` · {human_size(p.stat().st_size)}" for p in files]
    if meta.get("error"):
        lines += ["", f"❌ {meta['error']}"]
    try:
        log = (job / JOB_LOG).read_text()
    except OSError:
        log = ""

    def newest(suffix):
        found = [p for p in files if p.suffix == suffix and p.stat().st_size > 0]
        return str(max(found, key=os.path.getmtime)) if found else None

    targets = {
        "mgf": newest(".mgf"),
        "vqms2": newest(".vqms2"),
        "ckpt": newest(".ckpt"),
        "params": params,
    }
    return "\n".join(lines), log, targets


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
