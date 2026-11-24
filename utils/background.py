"""Keeping a script running after the terminal (an SSH session) closes.

detach() starts the script again in a new session with its output in a log file. A new
session has no controlling terminal, so the copy gets no SIGHUP when the connection drops,
and it never writes to a closed terminal.

Started again rather than forked: torch is already imported by then, and forking without
exec is unsafe on macOS and once CUDA is initialised.

This module knows nothing about what the script does. gui.py uses it directly for the
server; main.py and train.py go through utils.session_cli, which runs them as sessions.
It imports no torch: gui.py --list must stay fast.
"""

import os
import subprocess
import sys

# Set for the copy, which gets the same arguments (--background included).
_TOKEN_ENV = "SPECTROVQ_BACKGROUND_COPY"


def add_background_argument(parser, log_help):
    parser.add_argument(
        "--background",
        "-bg",
        action="store_true",
        help=f"Keep running after the terminal or SSH session closes; output goes to {log_help}",
    )


def copy_token(take=True):
    """In a detached copy: the token detach() was given. None in a process the user started.

    take=False only looks. Take it once: the copy's own subprocesses must not inherit it.
    """
    return os.environ.pop(_TOKEN_ENV, None) if take else os.environ.get(_TOKEN_ENV)


def detach(script, argv, log_path, token="1"):
    """Start `python script argv` detached from the terminal, say where, return its PID.

    argv: the arguments this process was given, or None for sys.argv[1:].
    token: what copy_token() returns in the copy, e.g. where it should record its outcome.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new_session = (
        {
            "creationflags": subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        }
        if os.name == "nt"
        else {"start_new_session": True}
    )
    # Appended: a resumed run must not erase the log of the run it continues.
    # Owner-only, whatever the folder's mode: gui.py's log holds its token URL.
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)  # also a log created before this, with the umask's mode
    with os.fdopen(fd, "a") as log:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(script), *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Unbuffered, so `tail -f` on the log shows progress as it happens.
            env={**os.environ, _TOKEN_ENV: str(token), "PYTHONUNBUFFERED": "1"},
            **new_session,
        )
    print(f"Running in the background, PID {proc.pid}")
    print(f"Log: {log_path}")
    return proc.pid
