"""Benchmark: SpectroVQ vs a TopN-peaks baseline, per-spectrum compression ratio vs cosine.

For every spectrum (up to a limit) and every setting:
    SpectroVQ, n_q quantizers: n_q x quantizedlen codes x CODE_BITS bits, reconstructed by the model
    TopN peaks, N:             the N most intense peaks inside MZ_RANGE, stored as float32 pairs
Compression ratio = raw peak list size (all peaks as float32 m/z + intensity pairs) / that size.
Similarity = cosine between the binned input and the binned reconstruction, the same space as
utils.explore. Both are per-spectrum nominal sizes, before the entropy coder and side files.

The MATCHED metric follows the lab's plot_similarity_vs_compression.py instead (see evaluate()),
so its plots can be compared with those figures. Its residual second pass is a different
compression scheme and is not reproduced.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from utils.config import MZ_RANGE
from utils.errors import UserError
from utils.spectra import (
    RAW_PEAK_BYTES,
    code_bytes,
    cosines,
    iter_ms2,
    peaks_in_range,
    reconstruct_batch,
    spectrum_to_vector,
)

STANDARD = "Binned cosine, fixed TopN"
MATCHED = "Square-root cosine, ratio-matched TopN"
METRICS = (STANDARD, MATCHED)
OUTPUT_SUFFIXES = (".csv", ".svg", ".png")
CSV_HEADER = [
    "spectrum_index",
    "method",
    "setting",
    "peaks",
    "compression_ratio",
    "cosine",
]
GOOD_COSINE = 0.8  # the summary's "Cosine >= 0.8" column


def parse_ints(text, what, lo=1, hi=None):
    """Sorted distinct whole numbers from comma- or space-separated text, in [lo, hi]."""
    try:
        values = sorted({int(v) for v in str(text).replace(",", " ").split()})
    except ValueError:
        raise UserError(f"{what}: enter whole numbers separated by commas.") from None
    if not values or values[0] < lo or (hi is not None and values[-1] > hi):
        bound = f"{lo}–{hi}" if hi is not None else f"≥ {lo}"
        raise UserError(f"{what}: values must be {bound}.")
    return values


def spectrovq_batch(model, vecs, n_qs):
    """Reconstructions of a (B, D) batch for each n_q."""
    return {q: reconstruct_batch(model, vecs, q) for q in n_qs}


def top_n_vector(mz, it, n):
    in_range = (mz >= MZ_RANGE[0]) & (mz < MZ_RANGE[1])
    mz, it = mz[in_range], it[in_range]
    if len(it) > n:
        keep = np.argpartition(it, -n)[-n:]
        mz, it = mz[keep], it[keep]
    return spectrum_to_vector(mz, it)[0].numpy()


def matched_top_n(sqrt_target, fold):
    """Top-k bins of a square-rooted binned spectrum, with k set by SpectroVQ's ratio.

    k = (bins above 1e-3 normalized intensity) / fold, as in the lab's
    plot_similarity_vs_compression.py; k = 0 gives an empty spectrum (cosine 0).
    """
    k = int(np.count_nonzero(sqrt_target * sqrt_target > 1e-3) / fold)
    if k <= 0:
        return np.zeros_like(sqrt_target)
    cutoff = np.partition(sqrt_target, -k)[-k]
    return np.where(sqrt_target > cutoff, sqrt_target, 0.0)


def evaluate(path, model, n_qs, top_ns, limit, batch_size, metric=STANDARD):
    """One row per (spectrum, method, setting): index, method, setting, peaks, ratio, cosine.

    metric STANDARD: cosine on binned intensities, ratio over all peaks, TopN with fixed N.
    metric MATCHED:  cosine on square-rooted binned intensities, ratio over the peaks inside
                     MZ_RANGE, and TopN given the same ratio as SpectroVQ for each spectrum (its
                     setting is then the quantizer count it is matched to).
    """
    sizes = {q: code_bytes(model, q) for q in n_qs}
    rows, batch, skipped, seen = [], [], 0, 0

    def flush_matched():
        vecs = torch.stack([b[3] for b in batch])
        targets = np.sqrt(vecs.numpy())
        for q, rec in spectrovq_batch(model, vecs, n_qs).items():
            sims = cosines(targets, np.sqrt(rec))
            for (i, _, (mz, _), _), t, s in zip(batch, targets, sims):
                n_in = peaks_in_range(mz, upper_inclusive=True)
                fold = n_in * RAW_PEAK_BYTES / sizes[q]
                top = matched_top_n(t, fold)
                rows.append((i, "SpectroVQ", q, n_in, fold, float(s)))
                rows.append(
                    (
                        i,
                        "TopN Peaks",
                        q,
                        n_in,
                        fold,
                        float(cosines(t[None], top[None])[0]),
                    )
                )
        batch.clear()

    def flush():
        if metric == MATCHED:
            return flush_matched()
        vecs = torch.stack([b[3] for b in batch])
        targets = vecs.numpy()
        for q, rec in spectrovq_batch(model, vecs, n_qs).items():
            sims = cosines(targets, rec)
            for (i, n_peaks, _, _), s in zip(batch, sims):
                ratio = n_peaks * RAW_PEAK_BYTES / sizes[q]
                rows.append((i, "SpectroVQ", q, n_peaks, ratio, float(s)))
        for n in top_ns:
            tops = np.stack([top_n_vector(mz, it, n) for _, _, (mz, it), _ in batch])
            sims = cosines(targets, tops)
            for (i, n_peaks, _, _), s in zip(batch, sims):
                ratio = n_peaks / max(min(n, n_peaks), 1)
                rows.append((i, "TopN Peaks", n, n_peaks, ratio, float(s)))
        batch.clear()

    for i, (mz, it) in enumerate(iter_ms2(path)):
        if limit and seen >= limit:
            break
        mz = np.asarray(mz, dtype=np.float32)
        it = np.asarray(it, dtype=np.float32)
        vec, _ = spectrum_to_vector(mz, it)
        if float(vec.sum()) == 0:
            skipped += 1
            continue
        batch.append((i, len(mz), (mz, it), vec.float().cpu()))
        seen += 1
        if len(batch) >= batch_size:
            flush()
            print(f"Evaluated {seen} spectra")
    if batch:
        flush()
    print(
        f"Evaluated {seen} spectra"
        + (
            f", skipped {skipped} with no peaks in {MZ_RANGE[0]}–{MZ_RANGE[1]} m/z"
            if skipped
            else ""
        )
    )
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)


def evaluation_stem(input_path):
    return f"{Path(input_path).stem}_evaluation"


def run_evaluation(
    path, model, n_qs, top_ns, limit, batch_size, metric, layout, output_base
):
    """evaluate() and save its rows (output_base.csv) and plot (.svg, .png).

    Returns (rows, figure, files written)."""
    from utils.plots import evaluation_plot  # matplotlib only when plotting

    rows = evaluate(path, model, n_qs, top_ns, limit, max(int(batch_size), 1), metric)
    if not rows:
        raise RuntimeError("No MS2 spectra with peaks in range were found.")
    write_csv(rows, output_base + ".csv")
    fig = evaluation_plot(rows, layout)
    fig.savefig(output_base + ".svg")
    fig.savefig(output_base + ".png", dpi=150)
    return rows, fig, [output_base + s for s in OUTPUT_SUFFIXES]


def summary(rows, model, matched=False):
    """Per (method, setting), SpectroVQ first: {method, setting, name, size, ratio (median),
    cosine (median), good (share with cosine >= GOOD_COSINE)}."""
    settings = sorted(
        {(r[1], r[2]) for r in rows}, key=lambda k: (k[0] != "SpectroVQ", k[1])
    )
    out = []
    for method, setting in settings:
        sel = [r for r in rows if r[1] == method and r[2] == setting]
        sims = np.array([r[5] for r in sel])
        if method == "SpectroVQ":
            name, size = f"{setting} quantizer(s)", f"{code_bytes(model, setting):g} B"
        elif matched:
            name = f"matched to {setting} quantizer(s)"
            size = f"≈ {code_bytes(model, setting):g} B"
        else:
            name, size = f"top {setting} peaks", f"≤ {setting * RAW_PEAK_BYTES} B"
        out.append(
            {
                "method": method,
                "setting": setting,
                "name": name,
                "size": size,
                "ratio": float(np.median([r[4] for r in sel])),
                "cosine": float(np.median(sims)),
                "good": float((sims >= GOOD_COSINE).mean()),
            }
        )
    return out
