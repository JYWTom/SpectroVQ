"""The command-line side of sessions: --background and --list for main.py and train.py.

Every run is a session of utils.sessions, started and recorded by the same calls as a run in
the GUI (which starts its training runs as train.py --background). In the foreground the
output goes to the terminal and the session's log; with --background the script is detached
(utils.background) and the detached copy records the outcome.
"""

import argparse
import contextlib
import sys
from pathlib import Path

from utils import background, sessions
from utils.config import COMPILE_MODES, DEFAULT_COMPILE


def add_background_argument(parser):
    background.add_background_argument(
        parser, "the log of its session (see main.py --list)"
    )


def add_compile_argument(parser, help):
    """--compile, the run's torch.compile mode; run_session records it in the session."""
    parser.add_argument(
        "--compile", choices=COMPILE_MODES, default=DEFAULT_COMPILE, help=help
    )


def add_list_argument(parser):
    """--list answers and exits like --help, so the required arguments are not needed."""

    class ListSessions(argparse.Action):
        def __call__(self, parser, *_):
            sessions.print_sessions()
            parser.exit()

    parser.add_argument(
        "--list",
        nargs=0,
        action=ListSessions,
        help="List your sessions (runs from the command line and from the GUI): the running "
        "ones with their log and progress, and the latest finished ones. Then exit",
    )


def existing_session(kind, continues=None):
    """The session run_session() will use if it exists already, else None (it will create
    one): in a --background copy the session its starter made, else for a resumed run the
    newest stopped session of this kind with the continues settings. Looks only, so a script
    can check a run against its session before starting it."""
    copy = background.copy_token(take=False)
    if copy:
        return Path(copy)
    if continues:
        return sessions.find_stopped(sessions.sessions_root(), kind, **continues)
    return None


@contextlib.contextmanager
def run_session(
    args, script, argv, kind, params, outputs=list, continues=None, hint=None
):
    """Put a script's work inside this; it yields the run's session folder.

    In the foreground it opens the session (sessions.open_session) and runs the work inside
    sessions.recording, which copies the output to the session's log and records the
    outcome. With --background, in the process the user started it opens the session,
    starts the detached copy, prints the session folder and exits; the copy runs the work
    and records the outcome.

    kind, params: what the session records, named as the GUI names them (params["input"]
        is what --list shows). args.compile (add_compile_argument) is recorded with them.
    continues: settings identifying an earlier run; its newest stopped session of this kind
        is continued instead of starting a new one (a resumed training run).
    hint: one more line to print after starting in the background, e.g. how to pause.
    """
    copy = background.copy_token()
    if copy is not None:
        with sessions.recording(Path(copy), outputs, log=False):  # detach() logs it
            yield Path(copy)
        return
    compile = getattr(args, "compile", "none")
    if args.background:
        _start(script, argv, kind, params, continues, hint, compile)
    job = sessions.open_session(kind, params, continues, compile=compile)
    with sessions.recording(job, outputs):
        yield job


# The line by which --background tells its starter (the GUI) the session folder.
SESSION_LINE = "Session: "


def _start(script, argv, kind, params, continues, hint, compile):
    job = sessions.open_session(kind, params, continues, compile=compile)
    pid = background.detach(script, argv, job / sessions.JOB_LOG, token=job)
    sessions.set_pid(job, pid)  # until here the session was this process's
    print(f"{SESSION_LINE}{job}")
    if hint:
        print(hint)
    print("`python main.py --list` and the GUI's sessions sidebar show it.")
    sys.exit(0)
