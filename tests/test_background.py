"""utils.background: a script restarted detached from the terminal."""

import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = """
import argparse, os, sys
sys.path.insert(0, {root!r})
from utils import background

parser = argparse.ArgumentParser()
parser.add_argument("--out")
background.add_background_argument(parser, "a log")
args = parser.parse_args()
token = background.copy_token()
if args.background and token is None:
    background.detach(__file__, None, args.out + ".log", token="from the starter")
    sys.exit(0)
print("working", flush=True)
with open(args.out, "w") as f:
    f.write(f"{{os.getpid()}} {{os.getsid(0)}} {{sys.stdin.isatty()}} ")
    f.write(f"{{'SPECTROVQ_BACKGROUND_COPY' in os.environ}} {{token}}")
"""


def wait_for_exit(pid, timeout=120):
    """Wait for a process that is not our child (its starter has already exited)."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} still running after {timeout} s")


def started_pid(stdout):
    return int(stdout.split("PID ")[1].split()[0])


@pytest.mark.skipif(os.name == "nt", reason="sessions are POSIX")
@pytest.mark.parametrize("flag", ["--background", "-bg"])
def test_runs_detached_and_logs(tmp_path, flag):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT.format(root=ROOT))
    out = tmp_path / "out.txt"
    log = tmp_path / "out.txt.log"
    log.write_text("earlier run\n")

    r = subprocess.run(
        [sys.executable, str(script), "--out", str(out), flag],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert f"Log: {log}" in r.stdout
    assert "working" not in r.stdout  # the work is not done by the process we started
    pid = started_pid(r.stdout)
    wait_for_exit(pid)

    child_pid, sid, stdin_tty, marker, token = out.read_text().split(" ", 4)
    assert int(child_pid) == pid
    # Its own session: no controlling terminal, so no SIGHUP when ours closes.
    assert int(sid) == pid != os.getsid(0)
    assert stdin_tty == "False"
    assert marker == "False"  # not passed on to the copy's own subprocesses
    assert token == "from the starter"
    assert log.read_text() == "earlier run\nworking\n"  # appended, not overwritten
