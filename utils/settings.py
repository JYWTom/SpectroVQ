"""Per-user state of an install: the data dir's layout and the settings file kept in it.

Several people can use one shared install (e.g. a lab GPU server), so each user's sessions,
settings and running GUI servers live in their own data dir (~/.spectrovq, $SPECTROVQ_DATA_DIR or
the GUI's --data-dir), readable by its owner only.
"""

import json
import os
from pathlib import Path
from typing import NamedTuple

from utils.sessions import SESSIONS_DIR

SETTINGS_FILE = (
    "gui_config.json"  # settings that outlive restarts: the last model that loaded
)
RUNTIME_DIR = "runtime"  # running GUI servers' URLs and tokens (utils.servers)


class DataDir(NamedTuple):
    root: Path
    sessions: Path
    settings: Path
    runtime: Path


def data_dir(path, create=False):
    """The layout of the data dir at path; with create, its folders are made owner-only."""
    root = Path(path).expanduser().resolve()
    layout = DataDir(
        root, root / SESSIONS_DIR, root / SETTINGS_FILE, root / RUNTIME_DIR
    )
    if create:
        for d in (layout.root, layout.sessions, layout.runtime):
            d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return layout


def read_settings(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save_settings(path, **updates):
    path = Path(path)
    settings = read_settings(path)
    if all(settings.get(k) == v for k, v in updates.items()):
        return
    settings.update(updates)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2))
    os.replace(tmp, path)  # atomic, so a crash never leaves half a file


def remember_model(path, ckpt_path, yaml_path=None):
    """Record the model that last loaded, so the next start offers it again."""
    save_settings(
        path,
        ckpt=os.path.abspath(os.path.expanduser(ckpt_path)),
        yaml=os.path.abspath(os.path.expanduser(yaml_path)) if yaml_path else "",
    )
