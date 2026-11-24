"""ResumeStore on its own: the start/resume checks against recorded settings, no training."""

import json
import shutil

import pytest

from utils.resume import ResumeError, ResumeStore, library_fingerprint

SETTINGS = {"batch_size": 2, "max_epochs": 3, "model": {"ratios": [9, 7]}}


def _pause(store):
    """What a paused run leaves in its checkpoint folder."""
    store.dir.mkdir(exist_ok=True)
    store.checkpoint.write_bytes(b"ckpt")


def test_layout(tmp_path):
    store = ResumeStore(tmp_path)
    assert store.checkpoint == tmp_path / "resume" / "resume.ckpt"
    assert store.find() is None


def test_fresh_start_writes_nothing(tmp_path):
    """The settings go to the run's session; the checkpoint folder holds checkpoints only."""
    store = ResumeStore(tmp_path)
    assert store.begin(SETTINGS, resume=False) == []
    assert not store.manifest.exists()
    assert store.find() is None  # no checkpoint yet


def test_fresh_start_refused_over_paused_run(tmp_path):
    store = ResumeStore(tmp_path)
    _pause(store)
    with pytest.raises(ResumeError, match="--resume"):
        store.begin(SETTINGS, resume=False)
    assert store.checkpoint.read_bytes() == b"ckpt"


def test_resume_needs_a_checkpoint(tmp_path):
    store = ResumeStore(tmp_path)
    with pytest.raises(ResumeError, match="Nothing to resume"):
        store.begin(SETTINGS, resume=True, recorded=SETTINGS)


def test_resume_with_same_settings(tmp_path):
    store = ResumeStore(tmp_path)
    _pause(store)
    assert store.begin(SETTINGS, resume=True, recorded=SETTINGS) == []


@pytest.mark.parametrize(
    "changed",
    [
        {**SETTINGS, "batch_size": 4},
        {**SETTINGS, "model": {"ratios": [9, 8]}},
        {k: v for k, v in SETTINGS.items() if k != "max_epochs"},
        {**SETTINGS, "extra": 1},
    ],
)
def test_resume_with_other_settings_refused(tmp_path, changed):
    store = ResumeStore(tmp_path)
    _pause(store)
    with pytest.raises(ResumeError, match="other settings") as e:
        store.begin(changed, resume=True, recorded=SETTINGS)
    assert str(store.dir) in str(e.value)


def test_resume_of_an_older_run_checks_its_manifest(tmp_path):
    """A run paused before its session recorded the settings left them in run.json."""
    store = ResumeStore(tmp_path)
    _pause(store)
    store.manifest.write_text(json.dumps({"version": 1, "settings": SETTINGS}))
    assert store.begin(SETTINGS, resume=True) == []
    with pytest.raises(ResumeError, match="other settings"):
        store.begin({**SETTINGS, "batch_size": 4}, resume=True)


def test_resume_without_recorded_settings_warns(tmp_path):
    """No session and no run.json: the run can still be continued."""
    store = ResumeStore(tmp_path)
    _pause(store)
    (warning,) = store.begin(SETTINGS, resume=True)
    assert "not checked" in warning


def test_unreadable_manifest(tmp_path):
    store = ResumeStore(tmp_path)
    _pause(store)
    store.manifest.write_text("{not json")
    with pytest.raises(ResumeError, match="unreadable"):
        store.begin(SETTINGS, resume=True)


def test_discard(tmp_path):
    store = ResumeStore(tmp_path)
    _pause(store)
    store.discard()
    assert not store.dir.exists()
    store.discard()  # nothing left: no error


@pytest.mark.parametrize("size", [10, 3 << 20])
def test_library_fingerprint(tmp_path, size):
    a = tmp_path / "a.feather"
    a.write_bytes(bytes(range(256)) * (size // 256) + b"x" * (size % 256))
    b = tmp_path / "moved" / "b.feather"
    b.parent.mkdir()
    shutil.copy(a, b)
    assert library_fingerprint(a) == library_fingerprint(b)  # copies and moves keep it
    data = bytearray(a.read_bytes())
    data[-1] ^= 1
    a.write_bytes(bytes(data))
    assert library_fingerprint(a) != library_fingerprint(b)
