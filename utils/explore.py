"""Reconstructing one spectrum with 1..n_q quantizers and measuring how close it gets."""

from typing import NamedTuple

import numpy as np

from utils.config import MZ_RANGE
from utils.errors import UserError
from utils.spectra import (
    RAW_PEAK_BYTES,
    code_bytes,
    cosine,
    peaks_in_range,
    read_spectrum,
    reconstruct_vector,
    spectrum_to_vector,
    vector_peaks,
)


class Exploration(NamedTuple):
    title: str
    quantizers: int  # the setting reconstructed (clamped to what the model has)
    input_peaks: tuple  # (m/z, intensity) of the binned input
    reconstruction: tuple  # (m/z, intensity) of its reconstruction
    similarities: list  # cosine to the input with 1, 2, ... model.n_q quantizers
    peaks: int  # in the raw input
    peaks_in_range: int  # of those, inside MZ_RANGE
    code_bytes: float  # size of the codes with `quantizers` quantizers
    raw_bytes: int  # size of the raw peak list

    @property
    def similarity(self):
        return self.similarities[self.quantizers - 1]


def explore_spectrum(model, path, index, quantizers):
    """Reconstruct spectrum index of path with quantizers, and compare every quantizer count."""
    max_q = int(model.n_q)
    n_q = int(min(max(1, quantizers), max_q))
    mz, it, title = read_spectrum(path, int(index))
    mz, it = np.asarray(mz), np.asarray(it)
    vec, _ = spectrum_to_vector(mz, it)
    if float(vec.sum()) == 0:
        raise UserError(
            f"Spectrum {int(index)} has no peaks in the {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z range."
        )
    target = vec.cpu().numpy()
    recon = reconstruct_vector(model, vec, n_q)
    sims = [
        cosine(target, reconstruct_vector(model, vec, q)) for q in range(1, max_q + 1)
    ]
    return Exploration(
        title=title,
        quantizers=n_q,
        input_peaks=vector_peaks(target),
        reconstruction=vector_peaks(recon),
        similarities=sims,
        peaks=len(mz),
        peaks_in_range=peaks_in_range(mz),
        code_bytes=code_bytes(model, n_q),
        raw_bytes=len(mz) * RAW_PEAK_BYTES,
    )
