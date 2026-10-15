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
written there too (under gui_outputs/). Nothing is uploaded from the browser.

Run on the server, then reach it e.g. through an SSH tunnel (ssh -L 7860:localhost:7860 server):
    python app.py [--host 127.0.0.1] [--port 7860] [--root /data] [--ckpt model.ckpt]
"""

import argparse
import contextlib
import io
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


def new_job_dir(prefix):
    OUTPUT_ROOT.mkdir(exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f"{datetime.now():%Y%m%d-%H%M%S}_{prefix}_", dir=OUTPUT_ROOT
        )
    )


def host_path(path, what="file"):
    """Validate a path on the host machine typed or picked in the GUI."""
    path = os.path.expanduser((path or "").strip())
    if not path:
        raise gr.Error(f"Choose a {what} on the server first.")
    if not os.path.isfile(path):
        raise gr.Error(f"Not found on the server: {path}")
    return os.path.abspath(path)


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
    start = browse_folder(value)
    with gr.Accordion(f"Browse {label}", open=False):
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


def run_logged(fn):
    """Run fn() in a thread, yielding (done, log_text, result, error) while it runs."""
    buf = _LogBuffer()
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

    t = threading.Thread(target=target, daemon=True)
    t.start()
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


def denoise_run(mgf_file, n_q, retain, threshold, compounded, ckpt, yaml_path, device):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device)
    job = new_job_dir("denoise")
    out_path = str(job / f"{Path(in_path).stem}_denoised.mgf")

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

    yield "", None, gr.update(), None, None
    for done, log, result, err in run_logged(work):
        if not done:
            yield log, None, gr.update(), None, None
            continue
        if err is not None:
            yield log + f"\n❌ {err}", None, gr.update(), None, None
            return
        n_in, n_out = count_spectra(in_path), count_spectra(result)
        log += (
            f"\n✅ Done. {n_in} input spectra → {n_out} output spectra "
            f"({n_out - n_in} extra chimeric spectra appended at the end).\n"
            f"Saved on the server: {result}"
        )
        yield (
            log,
            result,
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


def compress_run(mgf_file, n_q, method, level, batch_size, ckpt, yaml_path, device):
    in_path = host_path(mgf_file, "MGF file")
    model = load_model(ckpt, yaml_path, device)
    job = new_job_dir("compress")
    stem = str(job / Path(in_path).stem)

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

    yield "", None, ""
    for done, log, result, err in run_logged(work):
        if not done:
            yield log, None, ""
            continue
        if err is not None:
            yield log + f"\n❌ {err}", None, ""
            return
        original = os.path.getsize(in_path)
        total = sum(os.path.getsize(p) for p in result)
        rows = "\n".join(
            f"| `{Path(p).name}` | {human_size(os.path.getsize(p))} |" for p in result
        )
        summary = (
            f"| File | Size |\n|---|---|\n| original `{Path(in_path).name}` | {human_size(original)} |\n"
            f"{rows}\n| **total compressed** | **{human_size(total)}** "
            f"(ratio **{original / max(total, 1):.1f}×**) |\n\n"
            f"Decompress with **{int(n_q)} quantizers, {method}, level {int(level)}** — "
            f"the bitstream has no header, so these must match. Keep all files together.\n\n"
            f"Saved on the server in `{job}`."
        )
        yield log + "\n✅ Done.", result, summary


def decompress_run(
    vq_file, n_q, method, level, batch_size, stored_raw, ckpt, yaml_path, device
):
    vq = host_path(vq_file, ".vqms2 file")
    if not vq.endswith(".vqms2"):
        raise gr.Error(f"Expected a .vqms2 file, got {vq}")
    # The decompressor finds its side files next to the .vqms2 by swapping the extension.
    base = vq[: -len(".vqms2")]
    if not os.path.exists(base + ".parquet"):
        raise gr.Error(f"Metadata file not found next to it: {base}.parquet")
    model = load_model(ckpt, yaml_path, device)
    job = new_job_dir("decompress")
    out_path = str(job / f"{Path(vq).stem}_decompressed.mgf")

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

    yield "", None
    for done, log, result, err in run_logged(work):
        if not done:
            yield log, None
            continue
        if err is not None:
            yield (
                log
                + (
                    f"\n❌ {err}\n\nIf this is a decoding error, check that the quantizer count, method "
                    f"and level match the ones used for compression."
                ),
                None,
            )
            return
        yield (
            log
            + f"\n✅ Done. {count_spectra(result)} spectra written.\nSaved on the server: {result}",
            result,
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
    output_dir = os.path.abspath(
        os.path.expanduser((output_dir or "").strip()) or str(new_job_dir("train"))
    )
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
    buf.write("$ " + " ".join(cmd) + "\n")

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
    yield buf.get() + f"\n{status}", f"**Checkpoints in `{output_dir}`:**\n\n{listing}"


def train_stop():
    proc = _TRAIN["proc"]
    if proc is None or proc.poll() is not None:
        return "No training run in progress."
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    return "Sent stop signal to the training process."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui():
    default_ckpt = os.environ.get("SPECTROVQ_CKPT", "")
    with gr.Blocks(title="SpectroVQ") as demo:
        gr.Markdown(
            "# SpectroVQ\n"
            "Compress and denoise MS/MS spectra with a residual vector-quantized autoencoder. "
            "Fewer quantizers → more compression and more denoising; more quantizers → higher fidelity. "
            "Pretrained weights: [Zenodo](https://zenodo.org/records/20043624)."
        )

        with gr.Accordion("Model", open=True):
            with gr.Row():
                with gr.Column(scale=4):
                    ckpt, _ = host_file_picker(
                        "Checkpoint (.ckpt) path on the server",
                        "*.ckpt",
                        "/path/to/SpectroVQ.ckpt",
                        value=default_ckpt,
                    )
                with gr.Column(scale=2):
                    yaml_cfg, _ = host_file_picker(
                        "Model YAML (optional)", "*.yaml", "modelParams.yaml"
                    )
                with gr.Column(scale=1):
                    device = gr.Dropdown(
                        available_devices(), value="auto", label="Device"
                    )
                    load_btn = gr.Button("Load model")
            status = gr.Markdown(
                "_No model loaded yet. It is loaded automatically on first use._"
            )
            load_btn.click(model_status, [ckpt, yaml_cfg, device], status)
        model_inputs = [ckpt, yaml_cfg, device]

        with gr.Tabs():
            # ---------------- Explore ----------------
            with gr.Tab("Explore"):
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
            with gr.Tab("Denoise"):
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
                        dn_btn = gr.Button("Denoise", variant="primary")
                        dn_result = gr.File(label="Download denoised MGF")
                    with gr.Column(scale=2):
                        dn_log = gr.Textbox(
                            label="Log", lines=14, max_lines=14, autoscroll=True
                        )
                        dn_idx = gr.Slider(
                            0, 1, value=0, step=1, label="Preview spectrum index"
                        )
                        dn_plot = gr.Plot(label="Original vs denoised")
                dn_btn.click(
                    denoise_run,
                    [dn_file, dn_q, dn_retain, dn_thr, dn_comp, *model_inputs],
                    [dn_log, dn_result, dn_idx, dn_in, dn_out],
                    concurrency_id="pipeline",
                ).then(denoise_preview, [dn_in, dn_out, dn_idx], dn_plot)
                dn_idx.release(denoise_preview, [dn_in, dn_out, dn_idx], dn_plot)

            # ---------------- Compress ----------------
            with gr.Tab("Compress"):
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
                        cp_btn = gr.Button("Compress", variant="primary")
                    with gr.Column(scale=2):
                        cp_log = gr.Textbox(
                            label="Log", lines=12, max_lines=12, autoscroll=True
                        )
                        cp_summary = gr.Markdown()
                        cp_result = gr.File(
                            label="Download compressed files", file_count="multiple"
                        )
                cp_btn.click(
                    compress_run,
                    [cp_file, cp_q, cp_method, cp_level, cp_bs, *model_inputs],
                    [cp_log, cp_result, cp_summary],
                    concurrency_id="pipeline",
                )

            # ---------------- Decompress ----------------
            with gr.Tab("Decompress"):
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
                        dc_btn = gr.Button("Decompress", variant="primary")
                    with gr.Column(scale=2):
                        dc_log = gr.Textbox(
                            label="Log", lines=14, max_lines=14, autoscroll=True
                        )
                        dc_result = gr.File(label="Download decompressed MGF")
                dc_btn.click(
                    decompress_run,
                    [dc_files, dc_q, dc_method, dc_level, dc_bs, dc_raw, *model_inputs],
                    [dc_log, dc_result],
                    concurrency_id="pipeline",
                )

            # ---------------- Train ----------------
            with gr.Tab("Train"):
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
                    train_run,
                    [tr_data, tr_out, tr_bs, tr_lr, tr_ep, tr_nw],
                    [tr_log, tr_ckpts],
                    concurrency_limit=1,
                )
                tr_stop.click(train_stop, None, tr_stop_msg)
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
    parser.add_argument("--ckpt", default=None, help="Pre-fill the checkpoint path")
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
        allowed_paths=[str(OUTPUT_ROOT)],
    )


if __name__ == "__main__":
    main()
