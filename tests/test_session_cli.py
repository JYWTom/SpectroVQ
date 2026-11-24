"""utils.session_cli: a run of a script, in the foreground or --background, recorded as a session."""

import os
import sys
from types import SimpleNamespace

import pytest

from utils import background, sessions
from utils.session_cli import SESSION_LINE, existing_session, run_session

BACKGROUND = SimpleNamespace(background=True, compile="none")


@pytest.fixture
def copy(tmp_path, monkeypatch):
    """Make this process the detached copy of a new session, as detach() would."""
    job = sessions.new_session(tmp_path, "train", {})
    monkeypatch.setenv(background._TOKEN_ENV, str(job))
    return job


def _session(outputs=list):
    return run_session(BACKGROUND, "x.py", [], "train", {}, outputs)


@pytest.mark.parametrize(
    "code, status, error",
    [
        (None, "done", None),
        (0, "done", None),
        (75, "paused", None),
        (3, "failed", "Exited with code 3"),
        ("refused", "failed", "refused"),
    ],
)
def test_the_copy_records_how_it_exited(copy, code, status, error):
    with pytest.raises(SystemExit), _session(lambda: ["/m.ckpt"]):
        sys.exit(code)
    meta = sessions.read_job(copy)
    assert (meta["status"], meta.get("error")) == (status, error)
    assert meta["outputs"] == ["/m.ckpt"]


def test_the_copy_records_a_finished_run(copy):
    with _session(lambda: ["/out.vqms2"]):
        pass
    meta = sessions.read_job(copy)
    assert meta["status"] == "done" and meta["outputs"] == ["/out.vqms2"]


def test_the_copy_records_the_error_of_a_crash(copy):
    with pytest.raises(ValueError), _session():
        raise ValueError("bad archive")
    assert sessions.read_job(copy)["error"] == "ValueError: bad archive"


FOREGROUND = SimpleNamespace(background=False, compile="default")


def test_a_foreground_run_is_a_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPECTROVQ_DATA_DIR", str(tmp_path))
    with run_session(
        FOREGROUND, "x.py", [], "compress", {"input": "/in.mgf"}, lambda: ["/o"]
    ) as job:
        print("working")
        meta = sessions.read_job(job)
        assert meta["status"] == "running" and meta["pid"] == os.getpid()
    assert "working" in capsys.readouterr().out  # still on the terminal
    assert "working" in (job / "job.log").read_text()  # and in the session's log
    meta = sessions.read_job(job)
    assert meta["status"] == "done" and meta["outputs"] == ["/o"]
    assert (
        meta["params"]["compile"] == "default"
    )  # args.compile, recorded by the manager
    assert job.parent == sessions.sessions_root()


def test_a_foreground_resume_continues_the_paused_session(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTROVQ_DATA_DIR", str(tmp_path))
    root = sessions.sessions_root()
    paused = sessions.new_session(root, "train", {"output_dir": "/models"})
    sessions.finish_session(paused, status="paused")
    continues = {"output_dir": "/models"}
    assert existing_session("train", continues) == paused
    with (
        pytest.raises(SystemExit),
        run_session(FOREGROUND, "x.py", [], "train", {}, continues=continues) as job,
    ):
        sys.exit(75)
    assert job == paused and sessions.read_job(job)["status"] == "paused"
    assert "--- resumed ---" in (job / "job.log").read_text()


@pytest.mark.parametrize("resume", [False, True])
def test_the_starter_makes_or_continues_a_session(
    tmp_path, monkeypatch, resume, capsys
):
    monkeypatch.setenv("SPECTROVQ_DATA_DIR", str(tmp_path))
    root = sessions.sessions_root()
    paused = sessions.new_session(root, "train", {"output_dir": "/models"})
    sessions.finish_session(paused, status="paused")
    started = []
    monkeypatch.setattr(
        background, "detach", lambda *a, **kw: started.append((a, kw)) or 4242
    )

    with (
        pytest.raises(SystemExit) as stop,
        run_session(
            BACKGROUND,
            "train.py",
            ["--x"],
            "train",
            {"output_dir": "/models"},
            continues={"output_dir": "/models"} if resume else None,
        ),
    ):
        raise AssertionError("the starter must not do the work")
    assert stop.value.code == 0

    ((script, argv, log), kw) = started[0]
    job = kw["token"]
    assert (script, argv, log) == ("train.py", ["--x"], job / "job.log")
    meta = sessions.read_job(job)
    assert meta["status"] == "running" and meta["pid"] == 4242
    assert (job == paused) == resume  # a resume continues the paused session
    assert len(list(root.iterdir())) == (1 if resume else 2)
    assert ("--- resumed ---" in log.read_text()) if resume else not log.exists()
    assert f"{SESSION_LINE}{job}" in capsys.readouterr().out  # how the GUI finds it
