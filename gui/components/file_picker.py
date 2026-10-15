"""Path textbox with a collapsible explorer over the server's filesystem."""

import os
from typing import Any

import gradio as gr
from gradio.components.base import server

from gui.paths import browse_folder


class FolderExplorer(gr.FileExplorer):
    """A FileExplorer that lists folders only; expanding one shows a "." entry that picks it."""

    def get_block_name(self):
        return "fileexplorer"  # reuse FileExplorer's frontend

    @server
    def ls(self, subdirectory=None):
        items = super().ls(subdirectory) or []
        return [{**item, "valid": True} for item in items if item["type"] == "folder"]


def host_file_picker(label, pattern="*", placeholder="", value="", folders=False):
    """Collapsible path textbox and explorer over the server's filesystem.

    The explorer starts at --root (or the pre-filled path's folder); typing a path in the textbox
    moves it to that path's folder, and the ".." button puts the parent folder in the textbox
    (with no limit). Picking a file (or with folders=True, a folder) fills the textbox.

    Returns (textbox, on_pick); on_pick(fn, outputs) also runs fn(path) after a file is picked.
    """
    # value may be a function, re-run on every page load (it fills the box without firing
    # .change); the explorer starts from its value at startup.
    start = browse_folder(value() if callable(value) else value)
    with gr.Accordion(f"Browse {label}", open=False) as panel:
        folder = gr.State(start)
        with gr.Row():
            up = gr.Button("⬆ ..", size="sm", scale=0, min_width=64)
            box = gr.Textbox(label=label, placeholder=placeholder, value=value)
        explorer = (FolderExplorer if folders else gr.FileExplorer)(
            glob=pattern,
            ignore_glob="*/.*",  # hidden files and folders
            file_count="single",
            root_dir=start,
            label=None,
            show_label=False,
            max_height=320,
        )

    def go_to(target, current):
        if target == current:
            return gr.skip(), gr.skip()
        return target, gr.update(root_dir=target, value=[])

    def follow_typing(text, picked, current):
        # A file just picked in the explorer is already shown; don't re-root onto it.
        typed = os.path.abspath(os.path.expanduser(text or ""))
        if picked and typed == os.path.abspath(picked):
            return gr.skip(), gr.skip()
        return go_to(browse_folder(text), current)

    # .change, not .input: Gradio's Textbox fires "input" only on keypress, so it misses
    # Backspace and paste. Unqueued so a running job can't hold up browsing.
    browse: dict[str, Any] = {"queue": False, "show_progress": "hidden"}
    box.change(follow_typing, [box, explorer, folder], [folder, explorer], **browse)
    # Show a path filled in from elsewhere (e.g. the sessions sidebar).
    box.change(
        lambda text: gr.Accordion(open=True) if text else gr.skip(),
        box,
        panel,
        **browse,
    )
    up.click(os.path.dirname, folder, box, **browse)  # box.change then re-roots
    # A folder is picked through the "." entry inside it, so normalise "<folder>/.".
    picked = explorer.change(
        lambda p: os.path.normpath(p) if p else gr.skip(), explorer, box, **browse
    )

    def on_pick(fn, outputs):
        def run(path):
            if not path:
                return tuple(gr.skip() for _ in outputs)
            return fn(path)

        picked.then(run, explorer, outputs)

    return box, on_pick
