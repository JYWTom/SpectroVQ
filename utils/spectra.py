"""Reading single spectra, binning them and reconstructing them with the model.

Only MS2 is supported: an mzML's MS1 scans are never offered or read.
"""

import os
import re
from functools import lru_cache

import numpy as np
import torch
from pyteomics import mgf, mzml

from utils.Compression.preprocessing import rescaleSpectra
from utils.config import BIN_SIZE, CODE_BITS, MZ_RANGE, SPECTRA_DIMENSION
from utils.SpectrumProcessing.SpectraLoading import massSpectrumToVector

RAW_PEAK_BYTES = 8  # a raw peak: float32 m/z + float32 intensity


# --- Binning and reconstruction


def spectrum_to_vector(mz, intensity):
    """The binned spectrum the model takes (a tensor), and its maximum intensity."""
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


def vector_peaks(vec):
    """(m/z, intensity) of the non-empty bins of a binned spectrum, at the bin centres."""
    idx = np.nonzero(vec > 0)[0]
    return idx * BIN_SIZE + MZ_RANGE[0] + BIN_SIZE / 2, vec[idx]


def reconstruct_batch(model, vecs, n_q):
    """Reconstructions of a (B, D) batch of binned spectra with n_q quantizers, as a numpy array.

    Same postprocessing as MGFDenoiser / SpectrumBatchDecoder: rescale, square, drop < 1e-3.
    """
    device = next(model.parameters()).device
    x = torch.sqrt(vecs).unsqueeze(1).to(device)
    with torch.no_grad():
        out, _, _ = model.forwardWithnumQuantizers(x, n_q)
        out = torch.square(rescaleSpectra(out, dim=2))
        out = torch.where(out < 1e-3, torch.zeros_like(out), out)
    return out.squeeze(1).float().cpu().numpy()


def reconstruct_vector(model, vec, n_q):
    """Reconstruction of one binned spectrum with n_q quantizers."""
    return reconstruct_batch(model, vec.view(1, -1), n_q)[0]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosines(a, b):
    """Row-wise cosine similarity of two (n, d) arrays; 0 where either row is empty."""
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def code_bytes(model, n_q):
    """Size of one spectrum's codes with n_q quantizers (before the entropy coder)."""
    return n_q * model.quantizedlen * CODE_BITS / 8


def peaks_in_range(mz, upper_inclusive=False):
    mz = np.asarray(mz)
    upper = mz <= MZ_RANGE[1] if upper_inclusive else mz < MZ_RANGE[1]
    return int(((mz >= MZ_RANGE[0]) & upper).sum())


# --- Reading MGF and mzML

_MZML_SPECTRUM_ID = re.compile(r'<spectrum\s[^>]*?\bid="([^"]*)"')
_MZML_MS_LEVEL = re.compile(r'accession="MS:1000511"[^>]*?\bvalue="(\d+)"')


def is_mzml(path):
    return os.path.splitext(path)[1].lower() == ".mzml"


def spectrum_title(spec, i):
    params = spec.get("params", {})
    return str(params.get("title", f"spectrum {i}"))


@lru_cache(maxsize=4)
def _mzml_ms2_ids(path, _mtime):
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
    """Ids of the MS2 spectra of an mzML, in file order."""
    return _mzml_ms2_ids(path, os.path.getmtime(path))


def read_spectrum(path, i):
    """Spectrum i (clamped to the last one) of an MGF or of the MS2 scans of an mzML, as
    (m/z, intensity, title)."""
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
    """Number of spectra of an MGF, or of MS2 spectra of an mzML."""
    if is_mzml(path):
        return len(mzml_ms2_ids(path))
    n = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("BEGIN IONS"):
                n += 1
    return n


def iter_ms2(path):
    """(m/z, intensity) of every MS2 spectrum of an MGF or mzML, in file order."""
    if is_mzml(path):
        with mzml.read(path) as reader:
            for spec in reader:
                if int(spec.get("ms level", 2)) == 2:
                    yield spec["m/z array"], spec["intensity array"]
    else:
        with mgf.read(path, use_index=False) as reader:
            for spec in reader:
                yield spec["m/z array"], spec["intensity array"]
