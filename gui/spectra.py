"""Spectrum binning, reconstruction and plotting helpers."""

import os
import re
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyteomics import mgf, mzml

from gui.config import BIN_SIZE, MZ_RANGE, SPECTRA_DIMENSION
from utils.Compression.preprocessing import rescaleSpectra
from utils.SpectrumProcessing.SpectraLoading import massSpectrumToVector


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


_MZML_SPECTRUM_ID = re.compile(r'<spectrum\s[^>]*?\bid="([^"]*)"')
_MZML_MS_LEVEL = re.compile(r'accession="MS:1000511"[^>]*?\bvalue="(\d+)"')


def is_mzml(path):
    return os.path.splitext(path)[1].lower() == ".mzml"


@lru_cache(maxsize=4)
def _mzml_ms2_ids(path, _mtime):
    """Ids of the MS2 spectra of an mzML. Only MS2 is supported, so MS1 scans are not offered."""
    ids, current = [], None
    with open(path, errors="ignore") as f:
        for line in f:
            if "<spectrum " in line:
                match = _MZML_SPECTRUM_ID.search(line)
                current = match.group(1) if match else None
            elif current is not None and "MS:1000511" in line:
                match = _MZML_MS_LEVEL.search(line)
                if match and int(match.group(1)) == 2:
                    ids.append(current)
                current = None
            elif "<chromatogramList" in line:
                break
    return ids


def mzml_ms2_ids(path):
    return _mzml_ms2_ids(path, os.path.getmtime(path))


def read_spectrum(path, i):
    """Spectrum i of an MGF or the MS2 scans of an mzML, as (m/z, intensity, title)."""
    if is_mzml(path):
        ids = mzml_ms2_ids(path)
        spectrum_id = ids[min(i, len(ids) - 1)]
        with mzml.PreIndexedMzML(path) as reader:
            spec = reader.get_by_id(spectrum_id)
        return spec["m/z array"], spec["intensity array"], spectrum_id
    with mgf.IndexedMGF(path) as reader:
        i = min(i, len(reader) - 1)
        spec = reader[i]
    return spec["m/z array"], spec["intensity array"], spectrum_title(spec, i)


def count_spectra(path):
    if is_mzml(path):
        return len(mzml_ms2_ids(path))
    n = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("BEGIN IONS"):
                n += 1
    return n
