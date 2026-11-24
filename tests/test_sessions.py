"""utils.sessions: the session repository and manager every run goes through."""

import os
import subprocess
import sys

import pytest

from utils import sessions


def test_status_follows_the_process(tmp_path):
    job = sessions.new_session(tmp_path, "compress", {"input": "/in.mgf"})
    meta = sessions.read_job(job)
    # Its own pid proves nothing: this process may have restarted with the same number.
    assert sessions.session_status(job, meta) == "interrupted"
    assert sessions.session_status(job, meta, active={str(job)}) == "running"

    other = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    sessions.set_pid(job, other.pid)
    assert sessions.session_status(job, sessions.read_job(job)) == "running"
    other.communicate(b"\n")
    assert sessions.session_status(job, sessions.read_job(job)) == "interrupted"

    sessions.finish_session(job, [tmp_path / "out.vqms2"])
    sessions.set_pid(job, 1)  # too late: the run has already recorded its outcome
    meta = sessions.read_job(job)
    assert meta["status"] == "done" and meta["pid"] == other.pid
    assert meta["outputs"] == [str(tmp_path / "out.vqms2")]


def test_find_stopped_matches_kind_and_settings(tmp_path):
    a = sessions.new_session(tmp_path, "train", {"output_dir": "/a"})
    b = sessions.new_session(tmp_path, "train", {"output_dir": "/b"})
    sessions.finish_session(a, status="paused")
    sessions.finish_session(b)  # done: nothing to continue
    assert sessions.find_stopped(tmp_path, "train", output_dir="/a") == a
    assert sessions.find_stopped(tmp_path, "train", output_dir="/b") is None
    assert sessions.find_stopped(tmp_path, "compress", output_dir="/a") is None

    sessions.reopen_session(a, pid=os.getpid())
    meta = sessions.read_job(a)
    assert meta["status"] == "running" and "finished" not in meta


def test_last_log_line_reads_progress_bars(tmp_path):
    assert sessions.last_log_line(tmp_path) == ""
    (tmp_path / "job.log").write_text("start\nbatch 1/9\rbatch 2/9\r\n\n")
    assert sessions.last_log_line(tmp_path) == "batch 2/9"
    assert sessions.collapse_redraws("a\n1/9\r2/9\nb") == "a\n2/9\nb"


def test_list_shows_running_first_then_the_latest(tmp_path, capsys):
    sessions.print_sessions(tmp_path)
    assert "No SpectroVQ sessions in" in capsys.readouterr().out

    done = sessions.new_session(tmp_path, "compress", {"input": "/in.mgf"})
    sessions.finish_session(done, ["/out.vqms2"])
    failed = sessions.new_session(tmp_path, "decompress", {"input": "/x.vqms2"})
    sessions.finish_session(failed, error="boom")
    sessions.print_sessions(tmp_path)
    out = capsys.readouterr().out
    assert "No SpectroVQ sessions running. Latest:" in out
    assert "/out.vqms2" in out and "error: boom" in out


def test_an_in_process_run_is_recorded_like_a_command_line_one(tmp_path):
    """The GUI's in-process jobs go through open_session and recording too."""
    import io

    echo = io.StringIO()
    job = sessions.open_session("compress", {"input": "/in.mgf"}, root=tmp_path)
    with pytest.raises(ValueError), sessions.recording(job, list, echo=echo):
        print("half way")
        raise ValueError("bad input")
    assert echo.getvalue() == "half way\n"  # the GUI's live view
    assert "half way" in (job / "job.log").read_text()
    meta = sessions.read_job(job)
    assert (meta["status"], meta["error"]) == ("failed", "ValueError: bad input")


def test_every_session_records_its_compile_mode(tmp_path):
    job = sessions.open_session("train", {"output_dir": "/m"}, root=tmp_path)
    assert sessions.read_job(job)["params"]["compile"] == "none"
    sessions.finish_session(job, status="paused")
    # A resume may compile differently; the session then records the resumed run's mode.
    again = sessions.open_session(
        "train", {}, {"output_dir": "/m"}, root=tmp_path, compile="default"
    )
    assert again == job
    assert sessions.read_job(job)["params"] == {
        "output_dir": "/m",
        "compile": "default",
    }
    with pytest.raises(ValueError, match="compile mode"):
        sessions.open_session("train", {}, root=tmp_path, compile="fast")
