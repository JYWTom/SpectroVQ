"""Pause and resume of train.py, through real subprocesses on --accelerator cpu.

Each run is a separate process. Runs that need a signal or crash at an exact point go through
this file's __main__ driver, which calls train.main() with a callback that fires it; the
signal itself is real. test_external_sigterm runs plain train.py and signals it from outside.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATCHES = 10  # per epoch: 22 rows, 20 train + 2 val, batch size 2
EPOCHS = 3
FINAL = "final.ckpt"
# Overrides for train.model_kwargs2: about 0.25M parameters instead of 49M.
TINY_MODEL = {
    "n_filters": 4,
    "dimension": 16,
    "quantizationdim": 16,
    "codebook_size": 32,
    "n_q": 2,
    "transformer": 1,
}


def _driver():
    """Run train.main() with the callback described by the SVQ_HOOKS environment variable."""
    import lightning as L

    import train

    hooks = json.loads(os.environ.get("SVQ_HOOKS", "{}"))
    if os.environ.get("SVQ_TINY") == "1":
        # Pause/resume does not depend on model size; the full model makes every run slower
        # (a 49M-parameter model and its optimizer state are written at each checkpoint).
        train.model_kwargs2.update(TINY_MODEL)
    if os.environ.get("SVQ_FORK") == "1":
        # Loader workers are started again every epoch. Spawn (the macOS default) re-imports
        # torch, Lightning and transformers in each of them, which dominates the run time.
        import torch.multiprocessing as mp

        mp.set_start_method("fork", force=True)

    def at(name, *where):
        return list(where) in hooks.get(name, [])

    class Hooks(L.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            epoch = trainer.current_epoch
            if "trace" in hooks:
                with open(hooks["trace"], "a") as f:
                    for spectrum in batch[0]:
                        digest = hashlib.sha256(spectrum.numpy().tobytes()).hexdigest()
                        f.write(f"{epoch} {batch_idx} {digest}\n")
            if at("sigterm", epoch, batch_idx):
                os.kill(os.getpid(), signal.SIGTERM)
            if at("sigkill", epoch, batch_idx):
                os.kill(os.getpid(), signal.SIGKILL)
            if at("crash", epoch, batch_idx):
                raise RuntimeError("injected crash")
            if at("sigkill_next_save", epoch, batch_idx):
                from fsspec.implementations.local import LocalFileOpener

                real_write = LocalFileOpener.write

                def write_half_then_die(self, data):
                    real_write(self, data[: len(data) // 2])
                    self.flush()
                    os.kill(os.getpid(), signal.SIGKILL)

                LocalFileOpener.write = write_half_then_die

        def on_train_epoch_end(self, trainer, pl_module):
            if at("sigterm_epoch_end", trainer.current_epoch):
                os.kill(os.getpid(), signal.SIGTERM)
            if at("stop_epoch_end", trainer.current_epoch):
                trainer.should_stop = True

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            if not trainer.sanity_checking and at(
                "sigterm_val", trainer.current_epoch, batch_idx
            ):
                os.kill(os.getpid(), signal.SIGTERM)

        def on_train_end(self, trainer, pl_module):
            trainer.save_checkpoint(os.path.join(trainer.default_root_dir, FINAL))

    train.main(sys.argv[1:], extra_callbacks=[Hooks()])


if __name__ == "__main__":
    _driver()
    sys.exit(0)


import numpy as np
import pandas as pd
import pytest
import torch


def _library(path, rows=22):
    r = np.random.default_rng(0)
    records = []
    for i in range(rows):
        k = int(r.integers(15, 60))
        records.append(
            {
                "Name": f"PEPTIDE{'A' * (i % 8)}/{2 + i % 3}",
                "Comment": "x_HLA_y" if i % 2 else "z",
                "MZ": np.sort(r.uniform(160, 1490, k)).astype("float32").tolist(),
                "IT": r.uniform(1, 1000, k).astype("float32").tolist(),
                "PrecursorMZ": float(r.uniform(300, 900)),
            }
        )
    pd.DataFrame(records).to_feather(path)


class Run:
    """One output folder and the train.py processes run in it."""

    def __init__(
        self, library, folder, batch_size=2, workers=0, epochs=EPOCHS, tiny=True
    ):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace = self.dir / "trace.txt"
        self.tiny = tiny
        self.args = [
            "--data_path", str(library),
            "--output_dir", str(self.dir),
            "--batch_size", str(batch_size),
            "--num_workers", str(workers),
            "--max_epochs", str(epochs),
            "--accelerator", "cpu",
        ]  # fmt: skip

    @property
    def resume_ckpt(self):
        from utils.resume import ResumeStore

        return ResumeStore(self.dir).checkpoint

    def env(self, hooks=None):
        return {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "NO_COLOR": "1",
            "SVQ_TINY": "1" if self.tiny else "0",
            "SVQ_FORK": "1",
            "SVQ_HOOKS": json.dumps({"trace": str(self.trace), **(hooks or {})}),
        }

    def __call__(self, *extra, hooks=None, expect=0, script=__file__):
        p = subprocess.run(
            [sys.executable, "-u", str(script), *self.args, *extra],
            check=False,  # the exit code is asserted below
            cwd=self.dir,
            env=self.env(hooks),
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = p.stdout + p.stderr
        assert p.returncode == expect, (
            f"exit {p.returncode}, wanted {expect}\n{out[-4000:]}"
        )
        return out

    def final(self):
        return torch.load(self.dir / FINAL, weights_only=False)


def _diffs(a, b, path=""):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [d for k in a for d in _diffs(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        return [
            d for i, (x, y) in enumerate(zip(a, b)) for d in _diffs(x, y, f"{path}/{i}")
        ]
    if torch.is_tensor(a):
        return [] if torch.equal(a, b) else [path]
    return [] if a == b else [f"{path}: {a} != {b}"]


def _callback(ckpt, name):
    return next(v for k, v in ckpt["callbacks"].items() if k.startswith(name))


def assert_same_end(run, baseline):
    """An interrupted run ends where the uninterrupted one does: same step, epoch, schedule and
    sampler position. Weights are not compared: model RNG is not restored, so they differ."""
    a, b = run.final(), baseline
    assert a["global_step"] == b["global_step"]
    assert a["epoch"] == b["epoch"]
    assert not _diffs(a["lr_schedulers"], b["lr_schedulers"]), "scheduler differs"
    d = _diffs(_callback(a, "ResumeState"), _callback(b, "ResumeState"))
    assert not d, f"ResumeState differs: {d}"


@pytest.fixture(scope="session")
def work():
    d = tempfile.mkdtemp(prefix="svq_resume_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def library(work):
    path = work / "library.feather"
    _library(path)
    return path


@pytest.fixture(scope="session")
def baselines(work, library):
    """Uninterrupted final states, by (workers, epochs, batch size). Only state is kept."""
    cache = {}

    def get(workers=0, epochs=EPOCHS, batch_size=2):
        key = (workers, epochs, batch_size)
        if key not in cache:
            run = Run(library, work / f"base_{workers}_{epochs}_{batch_size}",
                      batch_size, workers, epochs)  # fmt: skip
            run()
            assert not (run.dir / "resume").exists()
            cache[key] = run
        return cache[key]

    return get


@pytest.fixture
def folder():
    d = tempfile.mkdtemp(prefix="svq_run_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# 1. Exactness


@pytest.mark.parametrize("workers", [0, 2])
def test_resumes_after_one_pause(library, baselines, folder, workers):
    base = baselines(workers)
    run = Run(library, folder, workers=workers)
    run(hooks={"sigterm": [[1, 4]]}, expect=75)
    assert torch.load(run.resume_ckpt, weights_only=False)["global_step"] == BATCHES + 5
    run("--resume")
    assert_same_end(run, base.final())


@pytest.mark.parametrize("workers", [0, 2])
def test_resumes_after_two_pauses_in_one_epoch(library, baselines, folder, workers):
    base = baselines(workers)
    run = Run(library, folder, workers=workers)
    run(hooks={"sigterm": [[1, 2]]}, expect=75)
    run("--resume", hooks={"sigterm": [[1, 6]]}, expect=75)
    run("--resume")
    assert_same_end(run, base.final())
    # 3. Sample coverage: in the interrupted epoch, every sample once, in the same order.
    seen = [ln.split() for ln in run.trace.read_text().splitlines()]
    epoch1 = [h for e, _, h in seen if e == "1"]
    assert len(epoch1) == 20
    assert len(set(epoch1)) == 20
    wanted = [ln.split() for ln in base.trace.read_text().splitlines()]
    assert epoch1 == [h for e, _, h in wanted if e == "1"]


def test_workers_do_not_change_the_result(baselines):
    assert not _diffs(
        baselines(0).final()["state_dict"], baselines(2).final()["state_dict"]
    )


# 2. Pause mechanics, on plain train.py with a signal from outside


def test_external_sigterm(library, folder):
    run = Run(library, folder)
    p = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "train.py"), *run.args],
        cwd=run.dir,
        env=run.env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.stdout is not None  # started with stdout=PIPE
    lines = []
    for line in p.stdout:
        lines.append(line)
        # Printed by the first batch of each epoch. Not startswith: the progress bar leaves
        # the line open, so the text often follows a stray "." on the same line.
        if "Training Loss" in line:
            break
    assert any(f"PID: {p.pid}" in ln for ln in lines)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.stdout.read()
    assert p.wait(timeout=120) == 75
    assert run.resume_ckpt.is_file()
    assert torch.load(run.resume_ckpt, weights_only=False)["global_step"] > 0

    out = run("--resume", script=ROOT / "train.py")
    assert not (run.dir / "resume").exists()
    # The run ends in validation, so this top-k file holds the last step.
    last = sorted(run.dir.glob("SpectroVQ-epoch=02-*.ckpt"))
    assert last, out[-2000:]
    assert torch.load(last[0], weights_only=False)["global_step"] == EPOCHS * BATCHES


def test_real_train_py_reaches_the_same_step(library, baselines, folder):
    base = baselines(0)
    # Plain train.py has no test driver, so it always builds the full model: pause with it too.
    run = Run(library, folder, tiny=False)
    run(hooks={"sigterm": [[0, 3]]}, expect=75)
    run("--resume", script=ROOT / "train.py")
    a = torch.load(next(run.dir.glob("SpectroVQ-epoch=02-*.ckpt")), weights_only=False)
    b = torch.load(next(base.dir.glob("SpectroVQ-epoch=02-*.ckpt")), weights_only=False)
    assert a["global_step"] == b["global_step"]
    assert not _diffs(a["lr_schedulers"], b["lr_schedulers"])


# 4. Pause during validation


def test_pause_during_validation(library, baselines, folder):
    # 22 rows at batch size 1: 2 validation batches, so the first is not the last.
    base = baselines(0, EPOCHS, 1)
    run = Run(library, folder, batch_size=1)
    run(hooks={"sigterm": [[2, 10]]}, expect=75)
    before = run.resume_ckpt.stat().st_mtime_ns
    run("--resume", hooks={"sigterm_val": [[2, 0]]}, expect=75)
    assert run.resume_ckpt.stat().st_mtime_ns == before, "validation pause wrote a file"
    assert not list(run.dir.glob("SpectroVQ-*.ckpt"))
    run("--resume")
    assert_same_end(run, base.final())


# 5. Epoch boundaries


@pytest.mark.parametrize(
    "hooks",
    [
        pytest.param(
            {"sigterm": [[0, BATCHES - 1]]}, id="last-batch-no-validation-next"
        ),
        pytest.param({"sigterm": [[1, BATCHES - 1]]}, id="last-batch-validation-next"),
        pytest.param(
            {"sigterm": [[2, BATCHES - 1]]}, id="last-batch-of-validation-epoch"
        ),
        pytest.param({"sigterm_epoch_end": [[0]]}, id="in-epoch-end-hook"),
    ],
)
def test_pause_at_epoch_boundary(library, baselines, folder, hooks):
    base = baselines(0, 4)
    run = Run(library, folder, epochs=4)
    run(hooks=hooks, expect=75)
    run("--resume")
    assert_same_end(run, base.final())
    seen = [ln.split() for ln in run.trace.read_text().splitlines()]
    assert len(seen) == 4 * 20, "an epoch lost or repeated batches"


def test_periodic_save_timed_for_last_batch(library, baselines, folder):
    base = baselines(0)
    run = Run(library, folder)
    run("--checkpoint_minutes", "0", hooks={"sigkill": [[0, BATCHES - 1]]}, expect=-9)
    ckpt = torch.load(run.resume_ckpt, weights_only=False)
    assert ckpt["global_step"] == BATCHES - 1, "saved on the last batch of the epoch"
    del ckpt
    run("--resume")
    assert_same_end(run, base.final())


def test_early_stop_at_pause_is_kept(library, folder):
    run = Run(library, folder, epochs=4)
    run(hooks={"stop_epoch_end": [[0]], "sigterm_epoch_end": [[0]]}, expect=75)
    run("--resume")
    assert run.final()["global_step"] == BATCHES, "training went on after the stop"


# 6. Crash


def test_crash_keeps_last_periodic_save(library, baselines, folder):
    base = baselines(0)
    run = Run(library, folder)
    run("--checkpoint_minutes", "0", hooks={"crash": [[0, 5]]}, expect=1)
    # Lightning runs checkpoint callbacks after other callbacks, so the crash at batch 5 comes
    # before that batch's save: the file holds the save of batch 4, which is step 5.
    step = torch.load(run.resume_ckpt, weights_only=False)["global_step"]
    assert step == 5, "the crash overwrote or lost the periodic checkpoint"
    run("--resume")
    assert_same_end(run, base.final())


def test_sigkill_during_save_leaves_a_loadable_file(library, baselines, folder):
    base = baselines(0)
    run = Run(library, folder)
    run(
        "--checkpoint_minutes", "0",
        hooks={"sigkill_next_save": [[0, 3]]},
        expect=-9,
    )  # fmt: skip
    ckpt = torch.load(run.resume_ckpt, weights_only=False)
    assert ckpt["global_step"] == 3
    del ckpt
    run("--resume")
    assert_same_end(run, base.final())


# 7. Fresh start guard


def test_fresh_start_refused_when_paused_run_exists(library, folder):
    run = Run(library, folder)
    run.resume_ckpt.parent.mkdir()
    run.resume_ckpt.write_bytes(b"paused")
    out = run(expect=1, script=ROOT / "train.py")
    assert "--resume" in out
    assert run.resume_ckpt.read_bytes() == b"paused"
    assert not (run.dir / "lightning_logs").exists()


def test_resume_without_checkpoint_refused(library, folder):
    out = Run(library, folder)("--resume", expect=1, script=ROOT / "train.py")
    assert "Nothing to resume" in out


def test_resume_with_other_settings_refused(library, folder):
    run = Run(library, folder)
    run(hooks={"sigterm": [[0, 3]]}, expect=75)
    before = run.resume_ckpt.stat().st_mtime_ns
    out = run("--resume", "--batch_size", "1", "--max_epochs", "5", expect=1)
    assert "batch_size: first run 2, now 1" in out
    assert "max_epochs: first run 3, now 5" in out
    assert run.resume_ckpt.stat().st_mtime_ns == before
    # Settings that may change, do not block it.
    run("--resume", "--num_workers", "2", "--checkpoint_minutes", "60")
    assert run.final()["global_step"] == EPOCHS * BATCHES


# 8. Consumers


def test_apply_weights_loads_resume_checkpoint(library, folder):
    from utils.model.getModel import ApplyWeights, getModel

    run = Run(
        library, folder, tiny=False
    )  # ApplyWeights is given the default architecture
    run(hooks={"sigterm": [[0, 3]]}, expect=75)
    state = _callback(torch.load(run.resume_ckpt, weights_only=False), "ResumeState")
    assert "rng" not in state, "RNG state is no longer stored"
    model = ApplyWeights(getModel(), str(run.resume_ckpt))  # default torch.load
    assert sum(p.numel() for p in model.parameters()) > 0


def test_shipped_weights_still_load():
    from utils.model.getModel import ApplyWeights, getModel

    weights = ROOT / "models" / "spectroVQ_weights.ckpt"
    if not weights.exists():
        pytest.skip("no shipped weights here")
    ApplyWeights(getModel(), str(weights))


def test_sidebar_never_offers_resume_checkpoint(folder):
    from gui.sessions import session_view

    job = folder / "20260101-000000_train_abc"
    (job / "resume").mkdir(parents=True)
    (job / "resume" / "resume.ckpt").write_bytes(b"x")
    (job / "job.json").write_text(json.dumps({"kind": "train", "status": "paused"}))
    _, _, targets = session_view(job)
    assert targets["ckpt"] is None


def test_sidebar_offers_resume_only_with_a_checkpoint(folder):
    from gui.sessions import session_view

    job = folder / "20260101-000000_train_abc"
    outputs = folder / "elsewhere"
    job.mkdir()
    meta = {"kind": "train", "status": "paused", "params": {"output_dir": str(outputs)}}
    (job / "job.json").write_text(json.dumps(meta))
    assert session_view(job)[2]["resume"] is False
    (outputs / "resume").mkdir(parents=True)
    (outputs / "resume" / "resume.ckpt").write_bytes(b"x")
    assert session_view(job)[2]["resume"] is True
    (job / "job.json").write_text(json.dumps({**meta, "status": "done"}))
    assert session_view(job)[2]["resume"] is False
