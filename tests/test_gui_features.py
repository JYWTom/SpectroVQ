"""GUI wiring for mzML, compile and resume settings. No model or GPU needed."""

import base64
import fnmatch
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pyteomics import mgf

from gui.components.file_picker import SPECTRA_GLOB
from gui.sessions import session_view
from gui.tabs import decompress
from utils import training
from utils.spectra import count_spectra, mzml_ms2_ids, read_spectrum
from utils.compile import COMPILE_MODES, compile_model, python_headers_available
from utils.model.getModel import getModel

ROOT = Path(__file__).resolve().parent.parent


def _b64(values):
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode()


def _mzml(path):
    """An indented mzML, as ProteoWizard writes it: one MS1 and two MS2 scans."""

    def spectrum(index, level, mz):
        arrays = "".join(
            f"""
          <binaryDataArray encodedLength="0">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
            <cvParam cvRef="MS" accession="{accession}" name="{name}" value=""/>
            <binary>{_b64(mz)}</binary>
          </binaryDataArray>"""
            for accession, name in (
                ("MS:1000514", "m/z array"),
                ("MS:1000515", "intensity array"),
            )
        )
        return f"""
      <spectrum index="{index}" id="scan={index + 1}" defaultArrayLength="{len(mz)}">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}"/>
        <binaryDataArrayList count="2">{arrays}
        </binaryDataArrayList>
      </spectrum>"""

    body = (
        spectrum(0, 1, [100.0])
        + spectrum(1, 2, [200.0, 300.0])
        + spectrum(2, 2, [400.0])
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n'
        '  <cvList count="1"><cv id="MS" fullName="MS" version="4" URI="x"/></cvList>\n'
        f'  <run id="r">\n    <spectrumList count="3">{body}\n    </spectrumList>\n  </run>\n'
        "</mzML>\n"
    )
    return str(path)


@pytest.mark.parametrize("name", ["a.mgf", "a.MGF", "a.mzML", "a.mzml", "a.MZML"])
def test_spectra_picker_offers_mgf_and_mzml(name):
    assert fnmatch.fnmatch(f"/data/{name}", SPECTRA_GLOB)


@pytest.mark.parametrize("name", ["a.ckpt", "a.vqms2", "a.parquet", "a.raw", "a.txt"])
def test_spectra_picker_hides_other_files(name):
    assert not fnmatch.fnmatch(f"/data/{name}", SPECTRA_GLOB)


def test_mzml_counts_ms2_only(tmp_path):
    path = _mzml(tmp_path / "r.mzML")
    assert mzml_ms2_ids(path) == ["scan=2", "scan=3"]
    assert count_spectra(path) == 2


def test_read_spectrum_mzml_skips_ms1(tmp_path):
    path = _mzml(tmp_path / "r.mzML")
    mz, intensity, title = read_spectrum(path, 0)
    np.testing.assert_allclose(mz, [200.0, 300.0])
    assert len(intensity) == 2
    assert title == "scan=2"
    assert read_spectrum(path, 99)[2] == "scan=3"  # clamped, like the MGF path


def test_read_spectrum_mgf(tmp_path):
    path = tmp_path / "r.mgf"
    mgf.write(
        [
            {
                "m/z array": np.array([100.0]),
                "intensity array": np.array([1.0]),
                "params": {"title": "first", "pepmass": 500.0},
            }
        ],
        str(path),
    )
    assert count_spectra(str(path)) == 1
    assert read_spectrum(str(path), 0)[2] == "first"


def test_explore_button_offers_mzml_output(tmp_path):
    job = tmp_path / "20260101-000000_decompress_abc"
    job.mkdir()
    (job / "job.json").write_text(
        '{"kind": "decompress", "status": "done", "outputs": ["%s"]}' % (job / "x.mzML")
    )
    (job / "x.mzML").write_text("<mzML/>")
    assert session_view(job)[2]["mgf"] == str(job / "x.mzML")


def test_train_command_carries_compile_and_checkpoint_settings(tmp_path):
    params = {
        "input": "l.feather",
        "output_dir": str(tmp_path),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 2,
        "num_workers": 0,
        "compile": "default",
        "checkpoint_minutes": 5,
    }
    cmd = training.train_command(params)
    assert "--background" in cmd  # the run is a command-line session like any other
    assert cmd[cmd.index("--compile") + 1] == "default"
    assert cmd[cmd.index("--checkpoint_minutes") + 1] == "5"


def test_old_sessions_resume_with_default_settings(tmp_path):
    params = {
        "input": "l.feather",
        "output_dir": str(tmp_path),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 2,
        "num_workers": 0,
    }
    cmd = training.train_command(params, resume=True)
    assert cmd[cmd.index("--compile") + 1] == "none"
    assert "--resume" in cmd


def test_gui_training_follows_the_session_train_py_started(tmp_path, monkeypatch):
    """The Train tab only starts train.py --background and reads the session it reports."""
    from utils import sessions

    job = sessions.new_session(tmp_path, "train", {})
    (job / "job.log").write_text("epoch 1\r\x1b[1mepoch 2\x1b[0m\n")
    sessions.finish_session(job, [str(tmp_path / "a.ckpt")])

    def run(cmd, **kwargs):
        assert "--background" in cmd
        return subprocess.CompletedProcess(cmd, 0, f"Session: {job}\n", "")

    monkeypatch.setattr(training.subprocess, "run", run)
    params = {
        "input": "l.feather",
        "output_dir": str(tmp_path),
        "batch_size": 1,
        "learning_rate": 1e-4,
        "max_epochs": 1,
        "num_workers": 0,
    }
    from gui.tabs import train

    *_, (log, result) = train._launch(params)
    assert "epoch 2" in log and "epoch 1" not in log and "\x1b" not in log
    assert "✅ Finished" in log and "a.ckpt" in result


def test_gui_training_shows_a_refused_start(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "resume.ckpt exists\n")

    monkeypatch.setattr(training.subprocess, "run", run)
    params = {
        "input": "l.feather",
        "output_dir": str(tmp_path),
        "batch_size": 1,
        "learning_rate": 1e-4,
        "max_epochs": 1,
        "num_workers": 0,
    }
    from gui.tabs import train

    *_, (log, _) = train._launch(params)
    assert "resume.ckpt exists" in log and "code 1" in log


def test_compile_modes_are_the_session_setting():
    """One list of modes: the session manager's, which utils.compile and the GUI use."""
    from utils import sessions

    assert COMPILE_MODES is sessions.COMPILE_MODES


def test_gui_sessions_record_the_compile_mode(tmp_path, monkeypatch):
    from utils.jobs import ACTIVE_JOBS, new_job
    from utils.sessions import read_job

    job = new_job(tmp_path, "compress", "default", input="/in.mgf")  # the run's own choice
    ACTIVE_JOBS.pop(str(job), None)
    assert read_job(job)["params"] == {"input": "/in.mgf", "compile": "default"}


def test_mzml_with_stored_raw_refused_before_a_session_starts(tmp_path, monkeypatch):
    (tmp_path / "x.vqms2").write_bytes(b"")
    (tmp_path / "x.parquet").write_bytes(b"")
    monkeypatch.setattr(decompress, "load_model", lambda *a: object())
    run = decompress.decompress_run(
        str(tmp_path / "x.vqms2"),
        "",
        4,
        "gzip",
        6,
        32,
        True,
        "mzML",
        "c",
        None,
        "cpu",
        "none",
    )
    with pytest.raises(Exception, match="Raw reconstructions"):
        next(run)


def test_compile_warns_instead_of_silently_running_eager(monkeypatch):
    monkeypatch.setattr("utils.compile.python_headers_available", lambda: False)
    model = getModel()
    with pytest.warns(UserWarning, match="Python.h"):
        assert compile_model(model, "default", "cpu", force=True) is False
    assert model.encoder._compiled_call_impl is None


def test_headers_check_runs():
    assert isinstance(python_headers_available(), bool)


def test_gui_has_no_server_wide_compile():
    """torch.compile is chosen per run in the Model panel, not for the whole server."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "gui.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    assert "--compile" not in out
