"""The registry of a user's running GUI servers, for `gui.py --list` (like `jupyter server list`).

Each server writes <data dir>/runtime/gui-<port>.json with its URL (token included) and PID, and
removes it when it exits.
"""

import json
import os
from pathlib import Path

from utils.sessions import pid_alive


def register_server(runtime_dir, port, url):
    """Record a running server; returns the file to remove when it stops."""
    path = Path(runtime_dir) / f"gui-{port}.json"
    # Created owner-only before the token is written, so other users can never read it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "pid": os.getpid(), "port": port}, f)
    return path


def running_servers(runtime_dir):
    """The {url, pid, port} records of the servers still running; drops stale ones."""
    servers = []
    for f in sorted(Path(runtime_dir).glob("gui-*.json")):
        try:
            info = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if pid_alive(info.get("pid", -1)):
            servers.append(info)
        else:
            f.unlink(missing_ok=True)  # left behind by a server that was killed
    return servers
