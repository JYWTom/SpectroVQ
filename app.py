"""
Browser GUI for SpectroVQ (Gradio).

Tabs:
    Explore    - reconstruct a single spectrum with 1..n_q quantizers and compare it to the input
    Denoise    - MGF -> denoised MGF (MGFDenoiser), with a before/after viewer
    Compress   - MGF -> .vqms2 + .parquet (+ _compounded.pkl) (MGFCompressor)
    Decompress - .vqms2 + side files -> MGF (MGFDecompressor)
    Train      - runs train.py as a subprocess and streams its log

The GUI is a remote front end for the machine it runs on (e.g. a GPU server): every input is a
path on that machine, picked by typing it or with the "Browse ..." file browser, and outputs are
written to an export path on that machine (default: the run's gui_outputs/ folder). Nothing is
uploaded from or downloaded to the browser.

Run on the server, then reach it e.g. through an SSH tunnel (ssh -L 7860:localhost:7860 server):
    python app.py [--host 127.0.0.1] [--port 7860] [--root /data] [--ckpt model.ckpt]
"""

import argparse
import contextlib
import functools
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyteomics import mgf

from utils.Compression.preprocessing import rescaleSpectra
from utils.mgfHandler.compressor import MGFCompressor
from utils.mgfHandler.decompressor import MGFDecompressor
from utils.mgfHandler.denoiser import MGFDenoiser
from utils.model.getModel import ApplyWeights, getModel
from utils.SpectrumProcessing.SpectraLoading import massSpectrumToVector

ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = ROOT / "gui_outputs"
# Starting folder of the "Browse ..." file browsers; set with --root. '..' and typed paths go anywhere.
BROWSE_ROOT = str(ROOT)

# Binning parameters; must match train.py, the compressor/decompressor and the denoiser.
BIN_SIZE = 0.1
SPECTRA_DIMENSION = 13500
MZ_RANGE = [150, 1500]
CODE_BITS = 10


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------


def available_devices():
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        devices.append("mps")
    devices.append("cpu")
    return devices


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()

# Server-side GUI settings that outlive restarts: the last model that loaded successfully.
CONFIG_PATH = ROOT / "gui_config.json"


def read_config():
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_config(**updates):
    config = read_config()
    if all(config.get(k) == v for k, v in updates.items()):
        return
    config.update(updates)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    os.replace(tmp, CONFIG_PATH)  # atomic, so a crash never leaves half a file


def initial_ckpt():
    """Checkpoint shown when a page opens: --ckpt wins, else the last one that loaded."""
    return os.environ.get("SPECTROVQ_CKPT") or read_config().get("ckpt", "")


def initial_yaml():
    return read_config().get("yaml", "")


def initial_model_status():
    ckpt = initial_ckpt()
    if not ckpt:
        return "_No model loaded yet. It is loaded automatically on first use._"
    source = "from --ckpt" if os.environ.get("SPECTROVQ_CKPT") else "last used"
    return (
        f"_Checkpoint ({source}): `{ckpt}`. It is loaded automatically on first use._"
    )


def load_model(ckpt_path, yaml_path, device):
    ckpt_path = os.path.expanduser((ckpt_path or "").strip())
    yaml_path = os.path.expanduser((yaml_path or "").strip()) or None
    if not ckpt_path:
        raise gr.Error("Set the model checkpoint path in the Model panel first.")
    if not os.path.exists(ckpt_path):
        raise gr.Error(f"Checkpoint not found: {ckpt_path}")
    if yaml_path is not None and not os.path.exists(yaml_path):
        raise gr.Error(f"YAML config not found: {yaml_path}")
    device = resolve_device(device)
    key = (os.path.abspath(ckpt_path), os.path.getmtime(ckpt_path), yaml_path, device)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()
            model = ApplyWeights(getModel(yaml_path), ckpt_path)
            model.to(device)
            model.eval()
            _MODEL_CACHE[key] = model
            save_config(
                ckpt=os.path.abspath(ckpt_path),
                yaml=os.path.abspath(yaml_path) if yaml_path else "",
            )
        return _MODEL_CACHE[key]


def model_status(ckpt_path, yaml_path, device):
    try:
        model = load_model(ckpt_path, yaml_path, device)
    except gr.Error as e:
        return f"⚠️ {e.message}"
    except Exception as e:
        return f"⚠️ Failed to load model: {e}"
    n_q = model.n_q
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return (
        f"✅ Loaded on **{resolve_device(device)}** · {n_params:.1f}M parameters · "
        f"{n_q} quantizers · {model.quantizedlen} codes per quantizer per spectrum"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Every run gets a gui_outputs/<YYYYmmdd-HHMMSS>_<kind>_<random>/ folder ("session") holding its
# outputs, a job.json record (kind, settings, status, outputs) and its log (job.log).
JOB_FILE = "job.json"
JOB_LOG = "job.log"
_JOB_NAME = re.compile(r"^(\d{8}-\d{6})_([a-z]+)_")
# Sessions still running in this process -> their live log buffer (None until it starts).
_ACTIVE_JOBS = {}


def _now():
    return datetime.now().isoformat(timespec="seconds")


def read_job(job):
    try:
        return json.loads((Path(job) / JOB_FILE).read_text())
    except (OSError, ValueError):
        return {}


def write_job(job, meta):
    (Path(job) / JOB_FILE).write_text(json.dumps(meta, indent=2))


def new_job_dir(prefix, **params):
    OUTPUT_ROOT.mkdir(exist_ok=True)
    job = Path(
        tempfile.mkdtemp(
            prefix=f"{datetime.now():%Y%m%d-%H%M%S}_{prefix}_", dir=OUTPUT_ROOT
        )
    )
    write_job(
        job,
        {
            "kind": prefix,
            "status": "running",
            "started": _now(),
            "pid": os.getpid(),  # lets other app instances tell running from interrupted
            "params": params,
        },
    )
    _ACTIVE_JOBS[str(job)] = None
    return job


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, OSError):
        return False
    return True


def job_running(job, meta):
    """Whether a session is still running, in this app process or another one."""
    if meta.get("status") != "running":
        return False
    return str(job) in _ACTIVE_JOBS or (
        meta.get("pid") not in (None, os.getpid()) and _pid_alive(meta["pid"])
    )


def mirror_log(job, buf, alive, every=2.0):
    """Copy a running session's log to job.log every few seconds, so any app instance (or a
    reloaded page) can follow it."""

    def run():
        last = None
        while alive():
            text = buf.get()
            if text != last:
                with contextlib.suppress(OSError):
                    (Path(job) / JOB_LOG).write_text(text)
                last = text
            time.sleep(every)

    threading.Thread(target=run, daemon=True).start()


def finish_job(job, outputs=None, error=None, log=""):
    """Record a session's outcome: status, finish time, output paths, error and log."""
    if isinstance(outputs, (str, os.PathLike)):
        outputs = [outputs]
    meta = read_job(job)
    meta.update(
        status="failed" if error else "done",
        finished=_now(),
        outputs=[str(p) for p in outputs or []],
    )
    if error:
        meta["error"] = str(error)
    write_job(job, meta)
    if log:
        (Path(job) / JOB_LOG).write_text(log)
    _ACTIVE_JOBS.pop(str(job), None)


def host_path(path, what="file"):
    """Validate a path on the host machine typed or picked in the GUI."""
    path = os.path.expanduser((path or "").strip())
    if not path:
        raise gr.Error(f"Choose a {what} on the server first.")
    if not os.path.isfile(path):
        raise gr.Error(f"Not found on the server: {path}")
    return os.path.abspath(path)


def export_path(text, job, default_name):
    """Resolve an export path typed in the GUI, refusing to overwrite existing files.

    Empty text means default_name inside the run's job folder; an existing folder (or text ending
    in '/') gets default_name inside it. Missing parent folders are created.
    """
    text = os.path.expanduser((text or "").strip())
    if not text:
        return str(Path(job) / default_name)
    path = os.path.abspath(text)
    if os.path.isdir(path) or text.endswith(os.sep):
        path = os.path.join(path, default_name)
    if os.path.exists(path):
        raise gr.Error(f"Already exists on the server, not overwriting: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def export_dir(text):
    """Resolve an export folder typed in the GUI (None if empty), creating it if needed."""
    text = os.path.expanduser((text or "").strip())
    if not text:
        return None
    path = Path(text).resolve()
    if path.exists() and not path.is_dir():
        raise gr.Error(f"Not a folder on the server: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def browse_folder(text):
    """Folder the file browser should show for a typed path.

    A folder shows itself and an existing file its folder; anything else (a half-typed path)
    shows its deepest existing parent. Empty text shows --root.
    """
    text = os.path.expanduser((text or "").strip())
    if not text:
        return BROWSE_ROOT
    folder = os.path.abspath(text)
    while not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    return folder


def host_file_picker(label, pattern, placeholder="", value=""):
    """Collapsible path textbox and explorer over the server's filesystem.

    The explorer starts at --root (or the pre-filled path's folder); typing a path in the textbox
    moves it to that path's folder, and the ".." button puts the parent folder in the textbox
    (with no limit). Picking a file fills the textbox.

    Returns (textbox, on_pick); on_pick(fn, outputs) also runs fn(path) after a file is picked.
    """
    # value may be a function, re-run on every page load (it fills the box without firing
    # .change); the explorer starts from its value at startup.
    start = browse_folder(value() if callable(value) else value)
    with gr.Accordion(f"Browse {label}", open=False) as panel:
        folder = gr.State(start)
        with gr.Row():
            up = gr.Button("⬆ ..", size="sm", scale=0, min_width=64)
            box = gr.Textbox(label=label, placeholder=placeholder, value=value)
        explorer = gr.FileExplorer(
            glob=pattern,
            ignore_glob="*/.*",  # hidden files and folders
            file_count="single",
            root_dir=start,
            label=None,
            show_label=False,
            max_height=320,
        )

    def go_to(target, current):
        if target == current:
            return gr.skip(), gr.skip()
        return target, gr.update(root_dir=target, value=[])

    def follow_typing(text, picked, current):
        # A file just picked in the explorer is already shown; don't re-root onto it.
        if picked and os.path.abspath(os.path.expanduser(text or "")) == picked:
            return gr.skip(), gr.skip()
        return go_to(browse_folder(text), current)

    # .change, not .input: Gradio's Textbox fires "input" only on keypress, so it misses
    # Backspace and paste. Unqueued so a running job can't hold up browsing.
    browse = {"queue": False, "show_progress": "hidden"}
    box.change(follow_typing, [box, explorer, folder], [folder, explorer], **browse)
    # Show a path filled in from elsewhere (e.g. the sessions sidebar).
    box.change(
        lambda text: gr.Accordion(open=True) if text else gr.skip(),
        box,
        panel,
        **browse,
    )
    up.click(os.path.dirname, folder, box, **browse)  # box.change then re-roots
    picked = explorer.change(lambda p: p or gr.skip(), explorer, box, **browse)

    def on_pick(fn, outputs):
        def run(path):
            if not path:
                return tuple(gr.skip() for _ in outputs)
            return fn(path)

        picked.then(run, explorer, outputs)

    return box, on_pick


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


class _LogBuffer(io.TextIOBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.text = ""

    def write(self, s):
        # tqdm redraws its bar with \r; replace the current line instead of appending
        with self.lock:
            for i, chunk in enumerate(s.split("\r")):
                if i > 0:
                    self.text = self.text[: self.text.rfind("\n") + 1]
                self.text += chunk
        return len(s)

    def last_line(self):
        with self.lock:
            return self.text.rstrip("\n").rpartition("\n")[2]

    def replace_last_line(self, line):
        with self.lock:
            self.text = self.text.rstrip("\n").rpartition("\n")[0] + "\n" + line

    def get(self, max_chars=30000):
        with self.lock:
            return self.text[-max_chars:]


# The pipelines print to stdout and redirect_stdout is process-wide,
# so in-process jobs are serialized through this lock and one Gradio concurrency group.
_JOB_LOCK = threading.Lock()


def run_logged(fn, job=None):
    """Run fn() in a thread, yielding (done, log_text, result, error) while it runs.

    With a session folder (job), its outcome is recorded from the thread itself, so it is
    saved even if the browser disconnects mid-run.
    """
    buf = _LogBuffer()
    if job is not None:
        _ACTIVE_JOBS[str(job)] = buf  # lets the sessions sidebar re-attach to the run
    state = {"result": None, "error": None}

    def target():
        with (
            _JOB_LOCK,
            contextlib.redirect_stdout(buf),
            contextlib.redirect_stderr(buf),
        ):
            try:
                state["result"] = fn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                state["error"] = e
        if job is not None:
            finish_job(job, state["result"], state["error"], buf.get())

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if job is not None:
        mirror_log(job, buf, t.is_alive)
    while t.is_alive():
        time.sleep(0.5)
        yield False, buf.get(), None, None
    yield True, buf.get(), state["result"], state["error"]


def spectrum_to_vector(mz, intensity):
    vec, max_val = massSpectrumToVector(
        mz,
        intensity,
        bin_size=BIN_SIZE,
        SPECTRA_DIMENSION=SPECTRA_DIMENSION,
        rawIT=False,
        Mode=None,
        mean0=False,
        CenterIntegerBins=False,
        AlterMZ=False,
        GenerateMZList=False,
        MZDiff=False,
        mzRange=MZ_RANGE,
    )
    return vec, max_val


def reconstruct_vector(model, vec, n_q):
    """Same postprocessing as MGFDenoiser / SpectrumBatchDecoder: rescale, square, drop < 1e-3."""
    device = next(model.parameters()).device
    x = torch.sqrt(vec).view(1, 1, -1).to(device)
    with torch.no_grad():
        out, _, _ = model.forwardWithnumQuantizers(x, n_q)
        out = torch.square(rescaleSpectra(out, dim=2))
        out = torch.where(out < 1e-3, torch.zeros_like(out), out)
    return out.view(-1).cpu().numpy()


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize(intensity):
    intensity = np.asarray(intensity, dtype=float)
    return (
        intensity / intensity.max()
        if intensity.size and intensity.max() > 0
        else intensity
    )


def mirror_plot(
    top_mz, top_it, bottom_mz, bottom_it, top_label, bottom_label, title=""
):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.vlines(
        top_mz, 0, _normalize(top_it), color="#1f6feb", linewidth=0.9, label=top_label
    )
    ax.vlines(
        bottom_mz,
        0,
        -_normalize(bottom_it),
        color="#d1242f",
        linewidth=0.9,
        label=bottom_label,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def vector_peaks(vec):
    idx = np.nonzero(vec > 0)[0]
    return idx * BIN_SIZE + MZ_RANGE[0] + BIN_SIZE / 2, vec[idx]


def spectrum_title(spec, i):
    params = spec.get("params", {})
    return str(params.get("title", f"spectrum {i}"))


def count_spectra(path):
    n = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("BEGIN IONS"):
                n += 1
    return n


# ---------------------------------------------------------------------------
# Explore tab
# ---------------------------------------------------------------------------


def explore_load(mgf_file):
    if not (mgf_file or "").strip():
        return gr.update(maximum=1, value=0), "Choose an MGF file.", None
    path = host_path(mgf_file, "MGF file")
    n = count_spectra(path)
    if n == 0:
        raise gr.Error("No spectra (BEGIN IONS blocks) found in this file.")
    return (
        gr.update(minimum=0, maximum=max(n - 1, 1), value=0, step=1),
        f"{n} spectra loaded.",
        path,
    )


def explore_run(mgf_path, idx, n_q, ckpt, yaml_path, device):
    if not mgf_path:
        raise gr.Error("Choose an MGF file first.")
    model = load_model(ckpt, yaml_path, device)
    max_q = int(model.n_q)
    n_q = int(min(max(1, n_q), max_q))
    with mgf.IndexedMGF(mgf_path) as reader:
        idx = min(int(idx), len(reader) - 1)
        spec = reader[idx]
    mz, it = np.asarray(spec["m/z array"]), np.asarray(spec["intensity array"])
    vec, _ = spectrum_to_vector(mz, it)
    if float(vec.sum()) == 0:
        raise gr.Error(
            f"Spectrum {int(idx)} has no peaks in the {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z range."
        )
    target = vec.cpu().numpy()

    recon = reconstruct_vector(model, vec, n_q)
    rec_mz, rec_it = vector_peaks(recon)
    in_mz, in_it = vector_peaks(target)
    title = spectrum_title(spec, int(idx))
    fig = mirror_plot(
        in_mz,
        in_it,
        rec_mz,
        rec_it,
        "input (binned)",
        f"reconstruction, {n_q} quantizer(s)",
        title=title,
    )

    sims = []
    for q in range(1, max_q + 1):
        sims.append(cosine(target, reconstruct_vector(model, vec, q)))
    curve, ax = plt.subplots(figsize=(5.2, 3.6))
    qs = np.arange(1, max_q + 1)
    ax.plot(qs, sims, marker="o", color="#1f6feb")
    ax.scatter(
        [n_q],
        [sims[n_q - 1]],
        s=90,
        facecolors="none",
        edgecolors="#d1242f",
        linewidths=2,
        zorder=3,
    )
    ax.set_xlabel("Number of quantizers")
    ax.set_ylabel("Cosine similarity to input")
    ax.set_xticks(qs)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    curve.tight_layout()
    plt.close(curve)

    bits = n_q * model.quantizedlen * CODE_BITS
    raw_bytes = len(mz) * 8  # float32 m/z + float32 intensity
    in_range = int(((mz >= MZ_RANGE[0]) & (mz < MZ_RANGE[1])).sum())
    summary = (
        f"**{title}**\n\n"
        f"| | |\n|---|---|\n"
        f"| Peaks in input | {len(mz)} ({in_range} within {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z) |\n"
        f"| Peaks in reconstruction | {len(rec_mz)} |\n"
        f"| Cosine similarity (binned) | {sims[n_q - 1]:.4f} |\n"
        f"| Code size | {n_q} × {model.quantizedlen} codes × {CODE_BITS} bit = {bits // 8} bytes |\n"
        f"| Raw peak list (float32 pairs) | {raw_bytes} bytes (≈{raw_bytes / max(bits / 8, 1):.1f}× larger) |\n"
    )
    return fig, curve, summary


# ---------------------------------------------------------------------------
# Denoise tab
# ---------------------------------------------------------------------------


def denoise_run(
    mgf_file, out_file, n_q, retain, threshold, compounded, ckpt, yaml_path, device
):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device)
    default_name = f"{Path(in_path).stem}_denoised.mgf"
    if (out_file or "").strip():
        export_path(out_file, None, default_name)  # fail before creating a session
    job = new_job_dir(
        "denoise",
        input=in_path,
        quantizers=int(n_q),
        retain_original_peaks=retain,
        one_percent_threshold=threshold,
        compounded_spectra=compounded,
    )
    out_path = export_path(out_file, job, default_name)

    def work():
        print(f"Input:  {in_path}\nOutput: {out_path}\nQuantizers: {int(n_q)}")
        MGFDenoiser(model=model, inputMGF=in_path, outputMGF=out_path).denoiseMGF(
            numQuantSingle=int(n_q),
            retainOriginalPeaks=retain,
            OnePercentThreshold=threshold,
            compoundedSpectra=compounded,
            num_workers=0,
        )
        return out_path

    yield "", gr.update(), None, None
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, gr.update(), None, None
            continue
        if err is not None:
            yield log + f"\n❌ {err}", gr.update(), None, None
            return
        n_in, n_out = count_spectra(in_path), count_spectra(result)
        log += (
            f"\n✅ Done. {n_in} input spectra → {n_out} output spectra "
            f"({n_out - n_in} extra chimeric spectra appended at the end).\n"
            f"Saved on the server: {result}"
        )
        yield (
            log,
            gr.update(minimum=0, maximum=max(n_in - 1, 1), value=0, step=1),
            in_path,
            result,
        )


def denoise_preview(in_path, out_path, idx):
    if not in_path or not out_path:
        return None
    with mgf.IndexedMGF(in_path) as r:
        idx = min(int(idx), len(r) - 1)
        a = r[idx]
    with mgf.IndexedMGF(out_path) as r:
        b = r[idx]
    return mirror_plot(
        a["m/z array"],
        a["intensity array"],
        b["m/z array"],
        b["intensity array"],
        f"original ({len(a['m/z array'])} peaks)",
        f"denoised ({len(b['m/z array'])} peaks)",
        title=spectrum_title(a, idx),
    )


# ---------------------------------------------------------------------------
# Compress / decompress tabs
# ---------------------------------------------------------------------------


def _levels(method, level):
    level = int(level) if level is not None else None
    return {
        "gzip_compression_level": level if method == "gzip" else None,
        "zlib_compression_level": level if method == "zlib" else None,
        "zstd_compression_level": level if method == "zstd" else None,
    }


def compress_run(
    mgf_file, out_dir, n_q, method, level, batch_size, ckpt, yaml_path, device
):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device)
    out_dir = export_dir(out_dir)
    if out_dir is not None:
        for ext in (".vqms2", ".parquet", "_compounded.pkl"):
            existing = out_dir / (Path(in_path).stem + ext)
            if existing.exists():
                raise gr.Error(
                    f"Already exists on the server, not overwriting: {existing}"
                )
    job = new_job_dir(
        "compress",
        input=in_path,
        quantizers=int(n_q),
        method=method,
        level=int(level),
        batch_size=int(batch_size),
    )
    stem = str((out_dir or job) / Path(in_path).stem)

    def work():
        print(
            f"Compressing {in_path} with {int(n_q)} quantizers, {method} level {int(level)}"
        )
        compressor = MGFCompressor(
            model=model,
            mgfFilePath=in_path,
            OutputFileName=stem,
            batch_size=int(batch_size),
            quantizer=int(n_q),
            **_levels(method, level),
        )
        compressor.CompressAll(verbose=1, num_workers=0)
        compressor.packer._raw_fo.close()
        return [
            p
            for p in (stem + ".vqms2", stem + ".parquet", stem + "_compounded.pkl")
            if os.path.exists(p)
        ]

    yield "", ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log, ""
            continue
        if err is not None:
            yield log + f"\n❌ {err}", ""
            return
        original = os.path.getsize(in_path)
        total = sum(os.path.getsize(p) for p in result)
        # Same "name · size" list style as the sessions sidebar.
        rows = "\n".join(
            f"- `{Path(p).name}` · {human_size(os.path.getsize(p))}" for p in result
        )
        summary = (
            f"**Original**\n- `{Path(in_path).name}` · {human_size(original)}\n\n"
            f"**Compressed**\n{rows}\n- **total** · **{human_size(total)}** "
            f"(ratio **{original / max(total, 1):.1f}×**)\n\n"
            f"Decompress with **{int(n_q)} quantizers, {method}, level {int(level)}** — "
            f"the bitstream has no header, so these must match. Keep all files together.\n\n"
            f"Saved on the server in `{out_dir or job}`."
        )
        yield log + "\n✅ Done.", summary


def decompress_run(
    vq_file,
    out_file,
    n_q,
    method,
    level,
    batch_size,
    stored_raw,
    ckpt,
    yaml_path,
    device,
):
    vq = host_path(vq_file, ".vqms2 file")
    if not vq.endswith(".vqms2"):
        raise gr.Error(f"Expected a .vqms2 file, got {vq}")
    # The decompressor finds its side files next to the .vqms2 by swapping the extension.
    base = vq[: -len(".vqms2")]
    if not os.path.exists(base + ".parquet"):
        raise gr.Error(f"Metadata file not found next to it: {base}.parquet")
    model = load_model(ckpt, yaml_path, device)
    default_name = f"{Path(vq).stem}_decompressed.mgf"
    if (out_file or "").strip():
        export_path(out_file, None, default_name)  # fail before creating a session
    job = new_job_dir(
        "decompress",
        input=vq,
        quantizers=int(n_q),
        method=method,
        level=int(level),
        batch_size=int(batch_size),
        stored_raw=stored_raw,
    )
    out_path = export_path(out_file, job, default_name)

    def work():
        print(
            f"Decompressing {vq} with {int(n_q)} quantizers, {method} level {int(level)}\n"
            f"Side files: {base}.parquet"
            + (
                f", {base}_compounded.pkl"
                if os.path.exists(base + "_compounded.pkl")
                else ""
            )
        )
        MGFDecompressor(
            model=model,
            inputFileName=vq,
            outputfilemgf=out_path,
            batch_size=int(batch_size),
            quantizer=int(n_q),
            stored_raw=stored_raw,
            **_levels(method, level),
        ).DecompressAll(verbose=1)
        return out_path

    yield ""
    for done, log, result, err in run_logged(work, job):
        if not done:
            yield log
            continue
        if err is not None:
            yield (
                log
                + (
                    f"\n❌ {err}\n\nIf this is a decoding error, check that the quantizer count, method "
                    f"and level match the ones used for compression."
                )
            )
            return
        yield (
            log
            + f"\n✅ Done. {count_spectra(result)} spectra written.\nSaved on the server: {result}"
        )


# ---------------------------------------------------------------------------
# Train tab (subprocess)
# ---------------------------------------------------------------------------

_TRAIN = {"proc": None}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def train_run(data_path, output_dir, batch_size, lr, max_epochs, num_workers):
    if _TRAIN["proc"] is not None and _TRAIN["proc"].poll() is None:
        raise gr.Error("A training run is already in progress.")
    data_path = os.path.expanduser((data_path or "").strip())
    if not data_path or not os.path.exists(data_path):
        raise gr.Error(f"Training library not found: {data_path!r}")
    data_path = os.path.abspath(data_path)
    params = {
        "input": data_path,
        "batch_size": int(batch_size),
        "learning_rate": lr,
        "max_epochs": int(max_epochs),
        "num_workers": int(num_workers),
    }
    # The session folder holds the record and log; checkpoints go there too unless an
    # output directory is given.
    job = new_job_dir("train", **params)
    output_dir = os.path.abspath(
        os.path.expanduser((output_dir or "").strip()) or str(job)
    )
    write_job(job, {**read_job(job), "params": {**params, "output_dir": output_dir}})
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--data_path",
        data_path,
        "--output_dir",
        output_dir,
        "--batch_size",
        str(int(batch_size)),
        "--learning_rate",
        str(lr),
        "--max_epochs",
        str(int(max_epochs)),
        "--num_workers",
        str(int(num_workers)),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=output_dir,  # Lightning writes lightning_logs/ to the working directory
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    _TRAIN["proc"] = proc
    buf = _LogBuffer()
    _ACTIVE_JOBS[str(job)] = buf
    buf.write("$ " + " ".join(cmd) + "\n")
    mirror_log(job, buf, lambda: proc.poll() is None)
    yield buf.get(), gr.update()  # first update right away (also refreshes the sidebar)

    def pump():
        for line in proc.stdout:
            line = _ANSI.sub("", line)
            # Lightning redraws its progress bar line by line; keep only the latest one
            if "━" in line and "━" in buf.last_line():
                buf.replace_last_line(line)
            else:
                buf.write(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while proc.poll() is None:
        time.sleep(1)
        yield buf.get(), gr.update()
    reader.join(timeout=5)
    ckpts = sorted(Path(output_dir).glob("*.ckpt"), key=os.path.getmtime)
    status = (
        "✅ Finished"
        if proc.returncode == 0
        else f"❌ Exited with code {proc.returncode}"
    )
    listing = "\n".join(f"- `{c}`" for c in ckpts) or "_No checkpoints written._"
    finish_job(
        job,
        ckpts,
        None if proc.returncode == 0 else f"Exited with code {proc.returncode}",
        buf.get(),
    )
    yield buf.get() + f"\n{status}", f"**Checkpoints in `{output_dir}`:**\n\n{listing}"


def train_stop():
    proc = _TRAIN["proc"]
    if proc is None or proc.poll() is not None:
        return "No training run in progress."
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    return "Sent stop signal to the training process."


# ---------------------------------------------------------------------------
# Sessions sidebar
# ---------------------------------------------------------------------------

SESSION_HEADERS = ["When", "Job", "Status"]
STATUS_LABELS = {
    "running": "⏳ running",
    "done": "✅ done",
    "failed": "❌ failed",
    "interrupted": "⚠️ interrupted",
    "empty": "⚠️ empty",
    "unknown": "– no record",
}


def session_status(job, meta):
    status = meta.get("status")
    if status == "running" and not job_running(job, meta):
        return "interrupted"  # the app stopped before the run finished
    if status:
        return status
    # Sessions from before job.json existed: all-empty files mean the run never finished.
    files = [p for p in Path(job).iterdir() if p.is_file()]
    return "empty" if all(p.stat().st_size == 0 for p in files) else "unknown"


def list_sessions():
    """(table rows, session folders), newest first."""
    rows, jobs = [], []
    if OUTPUT_ROOT.is_dir():
        for job in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
            m = _JOB_NAME.match(job.name)
            if not (m and job.is_dir()):
                continue
            meta = read_job(job)
            when = datetime.strptime(m[1], "%Y%m%d-%H%M%S")
            status = session_status(job, meta)
            rows.append(
                [f"{when:%m-%d %H:%M}", meta.get("kind", m[2]), STATUS_LABELS[status]]
            )
            jobs.append(str(job))
    return rows, jobs


def refresh_sessions():
    rows, jobs = list_sessions()
    return gr.update(value=rows or None), jobs


def with_session_list(job_fn):
    """Wrap a job generator so its event also outputs the sessions sidebar (table, folders):
    refreshed on its first update, which comes right after it creates its session folder, and
    again when it ends. The sidebar then reacts to the run starting and finishing."""

    @functools.wraps(job_fn)
    def run(*args):
        n = None
        for out in job_fn(*args):
            out = out if isinstance(out, tuple) else (out,)
            sessions = refresh_sessions() if n is None else (gr.skip(), gr.skip())
            n = len(out)
            yield (*out, *sessions)
        if n is not None:
            yield (*(gr.skip() for _ in range(n)), *refresh_sessions())

    return run


def session_files(job, meta):
    """Output files of a session: its folder's files plus outputs written elsewhere."""
    files = [
        p
        for p in sorted(Path(job).iterdir())
        if p.is_file() and p.name not in (JOB_FILE, JOB_LOG)
    ]
    seen = {p.resolve() for p in files}
    files += [
        Path(p)
        for p in meta.get("outputs", [])
        if Path(p).is_file() and Path(p).resolve() not in seen
    ]
    return files


def _duration(meta):
    try:
        start = datetime.fromisoformat(meta["started"])
        end = datetime.fromisoformat(meta["finished"])
    except (KeyError, ValueError):
        return None
    secs = int((end - start).total_seconds())
    return (
        f"{secs // 3600}h {secs // 60 % 60}m"
        if secs >= 3600
        else f"{secs // 60}m {secs % 60}s"
    )


def session_view(job):
    """Details markdown, log text and the send-to-tab targets of one session."""
    job = Path(job)
    meta = read_job(job)
    kind = meta.get("kind") or _JOB_NAME.match(job.name)[2]
    lines = [
        f"### {kind} · {STATUS_LABELS[session_status(job, meta)]}",
        f"`{job.name}`",
        "",
    ]
    if "started" in meta:
        took = _duration(meta)
        lines.append(
            f"Started {meta['started'].replace('T', ' ')}"
            + (f" · took {took}" if took else "")
        )
    params = meta.get("params", {})
    # A list, not a table: long paths would squeeze a table's first column in the sidebar.
    for k, v in params.items():
        name = f"**{k.replace('_', ' ')}**"
        lines.append(f"- {name}<br>`{v}`" if os.sep in str(v) else f"- {name}: `{v}`")
    files = session_files(job, meta)
    if files:
        # A list, like the settings: long file names would squeeze a table's size column.
        lines += ["", "**Files**"]
        lines += [f"- `{p.name}` · {human_size(p.stat().st_size)}" for p in files]
    if meta.get("error"):
        lines += ["", f"❌ {meta['error']}"]
    try:
        log = (job / JOB_LOG).read_text()
    except OSError:
        log = ""

    def newest(suffix):
        found = [p for p in files if p.suffix == suffix and p.stat().st_size > 0]
        return str(max(found, key=os.path.getmtime)) if found else None

    targets = {
        "mgf": newest(".mgf"),
        "vqms2": newest(".vqms2"),
        "ckpt": newest(".ckpt"),
        "params": params,
    }
    return "\n".join(lines), log, targets


def select_session(jobs, evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row is None or row >= len(jobs):
        return tuple(gr.skip() for _ in range(7))
    details, log, targets = session_view(jobs[row])
    return (
        details,
        gr.update(value=log, visible=bool(log)),
        targets,
        gr.update(visible=bool(targets["mgf"])),
        gr.update(visible=bool(targets["vqms2"])),
        gr.update(visible=bool(targets["ckpt"])),
        jobs[row],
    )


def send_to_decompress(targets):
    params = targets.get("params", {})
    return (
        targets["vqms2"],
        params.get("quantizers", gr.skip()),
        params.get("method", gr.skip()),
        params.get("level", gr.skip()),
        gr.Tabs(selected="decompress"),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui():
    with gr.Blocks(title="SpectroVQ") as demo:
        gr.Markdown(
            "# SpectroVQ\n"
            "Compress and denoise MS/MS spectra with a residual vector-quantized autoencoder. "
            "Fewer quantizers → more compression and more denoising; more quantizers → higher fidelity. "
            "Pretrained weights: [Zenodo](https://zenodo.org/records/20043624)."
        )

        with gr.Sidebar(label="Sessions", open=True, width=380):
            gr.Markdown(
                "### Sessions\nPast runs in `gui_outputs/`. Click one for details."
            )
            ss_refresh = gr.Button("↻ Refresh", size="sm")
            ss_table = gr.Dataframe(
                headers=SESSION_HEADERS,
                interactive=False,
                wrap=True,
                max_height=360,
                show_label=False,
                column_widths=["32%", "33%", "35%"],
            )
            ss_jobs, ss_targets = gr.State([]), gr.State({})
            ss_job = gr.State(None)  # the selected session folder
            ss_details = gr.Markdown()
            ss_to_explore = gr.Button(
                "Open output in Explore", size="sm", visible=False
            )
            ss_to_decompress = gr.Button("Send to Decompress", size="sm", visible=False)
            ss_to_ckpt = gr.Button("Use as model checkpoint", size="sm", visible=False)
            ss_log = gr.Textbox(
                label="Log", lines=8, max_lines=8, visible=False, autoscroll=True
            )

        with gr.Accordion("Model", open=True):
            with gr.Row():
                with gr.Column(scale=4):
                    ckpt, _ = host_file_picker(
                        "Checkpoint (.ckpt) path on the server",
                        "*.ckpt",
                        "/path/to/SpectroVQ.ckpt",
                        value=initial_ckpt,  # re-read on every page load
                    )
                with gr.Column(scale=2):
                    yaml_cfg, _ = host_file_picker(
                        "Model YAML (optional)",
                        "*.yaml",
                        "modelParams.yaml",
                        value=initial_yaml,
                    )
                with gr.Column(scale=1):
                    device = gr.Dropdown(
                        available_devices(), value="auto", label="Device"
                    )
                    load_btn = gr.Button("Load model")
            status = gr.Markdown(initial_model_status)  # re-run on every page load
            load_btn.click(model_status, [ckpt, yaml_cfg, device], status)
        model_inputs = [ckpt, yaml_cfg, device]

        with gr.Tabs() as tabs:
            # ---------------- Explore ----------------
            with gr.Tab("Explore", id="explore"):
                gr.Markdown(
                    "Reconstruct one spectrum with a chosen number of quantizers and see how similarity "
                    "grows as quantizers are added."
                )
                ex_path = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=1):
                        ex_file, ex_on_pick = host_file_picker(
                            "MGF path on the server", "*.mgf", "/path/to/spectra.mgf"
                        )
                        ex_info = gr.Markdown()
                        ex_idx = gr.Slider(
                            0, 1, value=0, step=1, label="Spectrum index"
                        )
                        ex_q = gr.Slider(1, 12, value=3, step=1, label="Quantizers")
                        ex_btn = gr.Button("Reconstruct", variant="primary")
                    with gr.Column(scale=3):
                        ex_plot = gr.Plot(label="Input vs reconstruction")
                        with gr.Row():
                            ex_curve = gr.Plot(label="Similarity vs quantizers")
                            ex_summary = gr.Markdown()
                ex_file.submit(explore_load, ex_file, [ex_idx, ex_info, ex_path])
                ex_file.blur(explore_load, ex_file, [ex_idx, ex_info, ex_path])
                ex_on_pick(explore_load, [ex_idx, ex_info, ex_path])
                ex_args = [ex_path, ex_idx, ex_q, *model_inputs]
                ex_outs = [ex_plot, ex_curve, ex_summary]
                ex_btn.click(explore_run, ex_args, ex_outs)
                ex_idx.release(explore_run, ex_args, ex_outs)
                ex_q.release(explore_run, ex_args, ex_outs)

            # ---------------- Denoise ----------------
            with gr.Tab("Denoise", id="denoise"):
                dn_in, dn_out = gr.State(None), gr.State(None)
                with gr.Row():
                    with gr.Column(scale=1):
                        dn_file, _ = host_file_picker(
                            "Input MGF path on the server", "*.mgf"
                        )
                        dn_q = gr.Slider(
                            1,
                            12,
                            value=3,
                            step=1,
                            label="Quantizers",
                            info="Fewer = stronger denoising",
                        )
                        dn_retain = gr.Checkbox(
                            True, label="Keep original peaks outside 150–1500 m/z"
                        )
                        dn_thr = gr.Checkbox(
                            True, label="Drop peaks below 1% of base peak"
                        )
                        dn_comp = gr.Checkbox(
                            True, label="Reconstruct chimeric spectra"
                        )
                        dn_outfile = gr.Textbox(
                            label="Export path on the server",
                            placeholder="empty = <session folder>/<name>_denoised.mgf",
                        )
                        dn_btn = gr.Button("Denoise", variant="primary")
                    with gr.Column(scale=2):
                        dn_log = gr.Textbox(
                            label="Log", lines=14, max_lines=14, autoscroll=True
                        )
                        dn_idx = gr.Slider(
                            0, 1, value=0, step=1, label="Preview spectrum index"
                        )
                        dn_plot = gr.Plot(label="Original vs denoised")
                dn_btn.click(
                    with_session_list(denoise_run),
                    [
                        dn_file,
                        dn_outfile,
                        dn_q,
                        dn_retain,
                        dn_thr,
                        dn_comp,
                        *model_inputs,
                    ],
                    [dn_log, dn_idx, dn_in, dn_out, ss_table, ss_jobs],
                    concurrency_id="pipeline",
                ).then(denoise_preview, [dn_in, dn_out, dn_idx], dn_plot)
                dn_idx.release(denoise_preview, [dn_in, dn_out, dn_idx], dn_plot)

            # ---------------- Compress ----------------
            with gr.Tab("Compress", id="compress"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cp_file, _ = host_file_picker(
                            "Input MGF path on the server", "*.mgf"
                        )
                        cp_q = gr.Slider(
                            1,
                            12,
                            value=4,
                            step=1,
                            label="Quantizers",
                            info="Must match at decompression (CLI uses 4)",
                        )
                        with gr.Row():
                            cp_method = gr.Dropdown(
                                ["gzip", "zlib", "zstd"], value="gzip", label="Method"
                            )
                            cp_level = gr.Number(6, precision=0, label="Level")
                        cp_bs = gr.Number(32, precision=0, label="Batch size")
                        cp_outdir = gr.Textbox(
                            label="Export folder on the server",
                            placeholder="empty = the session folder",
                        )
                        cp_btn = gr.Button("Compress", variant="primary")
                    with gr.Column(scale=2):
                        cp_log = gr.Textbox(
                            label="Log", lines=12, max_lines=12, autoscroll=True
                        )
                        cp_summary = gr.Markdown()
                cp_btn.click(
                    with_session_list(compress_run),
                    [
                        cp_file,
                        cp_outdir,
                        cp_q,
                        cp_method,
                        cp_level,
                        cp_bs,
                        *model_inputs,
                    ],
                    [cp_log, cp_summary, ss_table, ss_jobs],
                    concurrency_id="pipeline",
                )

            # ---------------- Decompress ----------------
            with gr.Tab("Decompress", id="decompress"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dc_files, _ = host_file_picker(
                            ".vqms2 path on the server", "*.vqms2"
                        )
                        gr.Markdown(
                            "The `.parquet` (and `_compounded.pkl`, if any) must sit next to it "
                            "with the same name."
                        )
                        dc_q = gr.Slider(
                            1,
                            12,
                            value=4,
                            step=1,
                            label="Quantizers (same as compression)",
                        )
                        with gr.Row():
                            dc_method = gr.Dropdown(
                                ["gzip", "zlib", "zstd"], value="gzip", label="Method"
                            )
                            dc_level = gr.Number(6, precision=0, label="Level")
                        dc_bs = gr.Number(32, precision=0, label="Batch size")
                        dc_raw = gr.Checkbox(
                            False, label="Also write raw (non-chimeric) reconstructions"
                        )
                        dc_outfile = gr.Textbox(
                            label="Export path on the server",
                            placeholder="empty = <session folder>/<name>_decompressed.mgf",
                        )
                        dc_btn = gr.Button("Decompress", variant="primary")
                    with gr.Column(scale=2):
                        dc_log = gr.Textbox(
                            label="Log", lines=14, max_lines=14, autoscroll=True
                        )
                dc_btn.click(
                    with_session_list(decompress_run),
                    [
                        dc_files,
                        dc_outfile,
                        dc_q,
                        dc_method,
                        dc_level,
                        dc_bs,
                        dc_raw,
                        *model_inputs,
                    ],
                    [dc_log, ss_table, ss_jobs],
                    concurrency_id="pipeline",
                )

            # ---------------- Train ----------------
            with gr.Tab("Train", id="train"):
                gr.Markdown(
                    "Runs `train.py` in a separate process. The library is a `.feather` consensus library "
                    "with columns `PrecursorMZ`, `CompoundName`, `Comment`, `m/z`, `intensity`, `Charge`. "
                    "Checkpoints are monitored on `val_mean_loss` with early stopping."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        tr_data, _ = host_file_picker(
                            "Library (.feather) path on the server",
                            "*.feather",
                            "library.feather",
                        )
                        tr_out = gr.Textbox(
                            label="Output directory",
                            placeholder="empty = gui_outputs/<timestamp>_train",
                        )
                        with gr.Row():
                            tr_bs = gr.Number(32, precision=0, label="Batch size")
                            tr_lr = gr.Number(1e-4, label="Learning rate")
                        with gr.Row():
                            tr_ep = gr.Number(50, precision=0, label="Max epochs")
                            tr_nw = gr.Number(
                                4, precision=0, label="Data loader workers"
                            )
                        with gr.Row():
                            tr_btn = gr.Button("Start training", variant="primary")
                            tr_stop = gr.Button("Stop", variant="stop")
                        tr_stop_msg = gr.Markdown()
                    with gr.Column(scale=2):
                        tr_log = gr.Textbox(
                            label="Log", lines=20, max_lines=20, autoscroll=True
                        )
                        tr_ckpts = gr.Markdown()
                tr_btn.click(
                    with_session_list(train_run),
                    [tr_data, tr_out, tr_bs, tr_lr, tr_ep, tr_nw],
                    [tr_log, tr_ckpts, ss_table, ss_jobs],
                    concurrency_limit=1,
                )
                tr_stop.click(train_stop, None, tr_stop_msg)

        # ---------------- Sessions sidebar ----------------
        light = {"queue": False, "show_progress": "hidden"}

        # Where each kind of run lives: its tab, log box, inputs (component -> job.json
        # setting) and how to show its outputs once done.
        def file_listing(title):
            def show(outs):
                return f"**{title}:**\n\n" + (
                    "\n".join(f"- `{p}`" for p in outs) or "_None written._"
                )

            return show

        session_tabs = {
            "denoise": (
                "denoise",
                dn_log,
                {dn_file: "input", dn_q: "quantizers"},
                None,
                None,
            ),
            "compress": (
                "compress",
                cp_log,
                {
                    cp_file: "input",
                    cp_q: "quantizers",
                    cp_method: "method",
                    cp_level: "level",
                },
                cp_summary,
                file_listing("Compressed files"),
            ),
            "decompress": (
                "decompress",
                dc_log,
                {
                    dc_files: "input",
                    dc_q: "quantizers",
                    dc_method: "method",
                    dc_level: "level",
                },
                None,
                None,
            ),
            "train": (
                "train",
                tr_log,
                {tr_data: "input"},
                tr_ckpts,
                file_listing("Checkpoints"),
            ),
        }
        watch_outputs = [tabs, ss_table, ss_jobs, ss_details, ss_log]
        for _, log_box, inputs, result_box, _ in session_tabs.values():
            watch_outputs += [log_box, *inputs] + ([result_box] if result_box else [])

        def watch_session(job):
            """Put a running session back in its tab and stream its log there until it ends."""
            meta = read_job(job) if job else {}
            view = session_tabs.get(meta.get("kind"))
            if view is None or not job_running(job, meta):
                yield {c: gr.skip() for c in watch_outputs}
                return
            tab, log_box, inputs, result_box, show = view
            params = meta.get("params", {})
            yield {
                tabs: gr.Tabs(selected=tab),
                **{c: params[k] for c, k in inputs.items() if k in params},
            }
            log_file = Path(job) / JOB_LOG
            while job_running(job, read_job(job)):
                buf = _ACTIVE_JOBS.get(job)  # live buffer if the run is in this process
                if buf is not None or log_file.exists():
                    text = buf.get() if buf is not None else log_file.read_text()
                    yield {log_box: text, ss_log: gr.update(value=text, visible=True)}
                time.sleep(1)
            meta = read_job(job)
            details, log, _ = session_view(job)
            outs = [p for p in meta.get("outputs", []) if os.path.isfile(p)]
            if meta.get("status") == "done":
                end = "✅ Done." + "".join(f"\nSaved on the server: {p}" for p in outs)
            else:
                end = f"❌ {meta.get('error')}"
            rows, jobs = list_sessions()
            yield {
                log_box: f"{log}\n{end}",
                **({result_box: show(outs)} if result_box else {}),
                ss_table: gr.update(value=rows or None),
                ss_jobs: jobs,
                ss_details: details,
                ss_log: gr.update(value=log, visible=bool(log)),
            }

        demo.load(refresh_sessions, None, [ss_table, ss_jobs], **light)
        ss_refresh.click(refresh_sessions, None, [ss_table, ss_jobs], **light)
        ss_table.select(
            select_session,
            ss_jobs,
            [
                ss_details,
                ss_log,
                ss_targets,
                ss_to_explore,
                ss_to_decompress,
                ss_to_ckpt,
                ss_job,
            ],
            **light,
        ).then(
            watch_session,
            ss_job,
            watch_outputs,
            concurrency_limit=None,
            show_progress="hidden",
        )
        ss_to_explore.click(
            lambda t: (t["mgf"], gr.Tabs(selected="explore")),
            ss_targets,
            [ex_file, tabs],
            **light,
        ).then(explore_load, ex_file, [ex_idx, ex_info, ex_path])
        ss_to_decompress.click(
            send_to_decompress,
            ss_targets,
            [dc_files, dc_q, dc_method, dc_level, tabs],
            **light,
        )
        ss_to_ckpt.click(lambda t: t["ckpt"], ss_targets, ckpt, **light)
    return demo


def main():
    parser = argparse.ArgumentParser(description="SpectroVQ browser GUI")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind; keep 127.0.0.1 and use an SSH tunnel, or 0.0.0.0 on a trusted network",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Pre-fill the checkpoint path (default: the last model that loaded, saved in gui_config.json)",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Starting folder of the server file browsers (default: the SpectroVQ project folder)",
    )
    args = parser.parse_args()
    if args.ckpt:
        os.environ["SPECTROVQ_CKPT"] = args.ckpt
    global BROWSE_ROOT
    BROWSE_ROOT = os.path.abspath(os.path.expanduser(args.root))
    OUTPUT_ROOT.mkdir(exist_ok=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        # Long paths in the sessions sidebar have few break points; let them wrap anywhere.
        css=".sidebar code { overflow-wrap: anywhere; }",
    )


if __name__ == "__main__":
    main()
