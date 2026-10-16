"""Sidebar listing past runs in gui_outputs/, with details, logs and send-to-tab buttons."""

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr

from gui import config
from gui.jobs import ACTIVE_JOBS, JOB_LOG, job_running, read_job
from gui.sessions import (
    SESSION_HEADERS,
    list_sessions,
    refresh_sessions,
    session_view,
)
from gui.tabs.explore import explore_load


def build():
    with gr.Sidebar(label="Sessions", open=True, width=380):
        gr.Markdown(
            f"### Sessions\nPast runs in `{config.OUTPUT_ROOT}`. Click one for details."
        )
        sb = SimpleNamespace(
            refresh=gr.Button("↻ Refresh", size="sm"),
            table=gr.Dataframe(
                headers=SESSION_HEADERS,
                interactive=False,
                wrap=True,
                max_height=360,
                show_label=False,
                column_widths=["32%", "33%", "35%"],
            ),
            jobs=gr.State([]),
            targets=gr.State({}),
            job=gr.State(None),  # the selected session folder
            details=gr.Markdown(),
            to_explore=gr.Button("Open output in Explore", size="sm", visible=False),
            to_decompress=gr.Button("Send to Decompress", size="sm", visible=False),
            to_ckpt=gr.Button("Use as model checkpoint", size="sm", visible=False),
            log=gr.Textbox(
                label="Log", lines=8, max_lines=8, visible=False, autoscroll=True
            ),
        )
    # Outputs a job event updates to refresh the sidebar (see sessions.with_session_list).
    sb.outputs = [sb.table, sb.jobs]
    return sb


def select_session(jobs, evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row is None or row >= len(jobs):
        return tuple(gr.skip() for _ in range(7))
    details, log, targets = session_view(jobs[row])
    return (
        details,
        gr.update(value=log, visible=bool(log)),
        targets,
        gr.update(visible=bool(targets["mgf"])),
        gr.update(visible=bool(targets["vqms2"])),
        gr.update(visible=bool(targets["ckpt"])),
        jobs[row],
    )


def send_to_decompress(targets):
    params = targets.get("params", {})
    return (
        targets["vqms2"],
        params.get("quantizers", gr.skip()),
        params.get("method", gr.skip()),
        params.get("level", gr.skip()),
        gr.Tabs(selected="decompress"),
    )


def wire(demo, sb, tabs, views, explore, decompress, ckpt):
    """Connect the sidebar to the page.

    views maps each session kind (also its tab id) to the tab's sessions.SessionView;
    explore and decompress are those tabs' components, ckpt the model panel's checkpoint box.
    """
    light: dict[str, Any] = {"queue": False, "show_progress": "hidden"}

    watch_outputs = [tabs, sb.table, sb.jobs, sb.details, sb.log]
    for view in views.values():
        watch_outputs += [view.log_box, *view.inputs]
        if view.result_box:
            watch_outputs.append(view.result_box)

    def watch_session(job):
        """Put a running session back in its tab and stream its log there until it ends."""
        meta = read_job(job) if job else {}
        kind = meta.get("kind")
        view = views.get(kind)
        if view is None or not job_running(job, meta):
            yield {c: gr.skip() for c in watch_outputs}
            return
        params = meta.get("params", {})
        yield {
            tabs: gr.Tabs(selected=kind),
            **{c: params[k] for c, k in view.inputs.items() if k in params},
        }
        log_file = Path(job) / JOB_LOG
        while job_running(job, read_job(job)):
            buf = ACTIVE_JOBS.get(job)  # live buffer if the run is in this process
            if buf is not None or log_file.exists():
                text = buf.get() if buf is not None else log_file.read_text()
                yield {view.log_box: text, sb.log: gr.update(value=text, visible=True)}
            time.sleep(1)
        meta = read_job(job)
        details, log, _ = session_view(job)
        outs = [p for p in meta.get("outputs", []) if os.path.isfile(p)]
        if meta.get("status") == "done":
            end = "✅ Done." + "".join(f"\nSaved on the server: {p}" for p in outs)
        else:
            end = f"❌ {meta.get('error')}"
        rows, jobs = list_sessions()
        yield {
            view.log_box: f"{log}\n{end}",
            **({view.result_box: view.show(outs)} if view.result_box else {}),
            sb.table: gr.update(value=rows or None),
            sb.jobs: jobs,
            sb.details: details,
            sb.log: gr.update(value=log, visible=bool(log)),
        }

    demo.load(refresh_sessions, None, sb.outputs, **light)
    sb.refresh.click(refresh_sessions, None, sb.outputs, **light)
    sb.table.select(
        select_session,
        sb.jobs,
        [
            sb.details,
            sb.log,
            sb.targets,
            sb.to_explore,
            sb.to_decompress,
            sb.to_ckpt,
            sb.job,
        ],
        **light,
    ).then(
        watch_session,
        sb.job,
        watch_outputs,
        concurrency_limit=None,
        show_progress="hidden",
    )
    sb.to_explore.click(
        lambda t: (t["mgf"], gr.Tabs(selected="explore")),
        sb.targets,
        [explore.file, tabs],
        **light,
    ).then(explore_load, explore.file, [explore.idx, explore.info, explore.path])
    sb.to_decompress.click(
        send_to_decompress,
        sb.targets,
        [decompress.file, decompress.q, decompress.method, decompress.level, tabs],
        **light,
    )
    sb.to_ckpt.click(lambda t: t["ckpt"], sb.targets, ckpt, **light)
