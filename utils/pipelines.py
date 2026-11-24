"""The file-level runs: compress, decompress and denoise a spectra file with a loaded model.

main.py, denoise.py and the GUI's tabs all run them through these functions; they take plain
settings and return the paths they wrote. The check_* functions refuse a bad request before a
session is created for it (UserError).
"""

import os
from pathlib import Path

from utils.config import codec_levels
from utils.errors import UserError
from utils.mgfHandler.compressor import SpectrumCompressor
from utils.mgfHandler.decompressor import SpectrumDecompressor
from utils.mgfHandler.denoiser import MGFDenoiser
from utils.paths import refuse_existing
from utils.spectra import count_spectra

ARCHIVE = ".vqms2"
METADATA = ".parquet"
LEGACY_COMPOUNDED = "_compounded.pkl"  # older archives, from before footer indices
COMPRESSED_SUFFIXES = (ARCHIVE, METADATA)


# --- Compress


def compressed_stem(input_path, folder):
    """Where compressing input_path into folder writes: <folder>/<input stem> + suffixes."""
    return str(Path(folder) / Path(input_path).stem)


def check_compress(input_path, folder):
    """Refuse to overwrite an earlier compression of the same file in folder."""
    refuse_existing(folder, Path(input_path).stem, COMPRESSED_SUFFIXES)


def compress(
    model,
    input_path,
    output_stem,
    quantizers,
    method=None,
    level=None,
    batch_size=128,
    verbose=1,
    num_workers=None,
):
    """Compress input_path into output_stem.vqms2 + .parquet; returns the files written.

    num_workers None keeps the compressor's default; method None means no entropy coder.
    """
    compressor = SpectrumCompressor(
        inputFilePath=input_path,
        OutputFileName=output_stem,
        model=model,
        batch_size=int(batch_size),
        quantizer=int(quantizers),
        **codec_levels(method, level),
    )
    extra = {} if num_workers is None else {"num_workers": num_workers}
    compressor.CompressAll(verbose=verbose, debug=False, **extra)
    stem = output_stem.removesuffix(ARCHIVE)  # the compressor accepts either form
    return [p for p in (stem + s for s in COMPRESSED_SUFFIXES) if os.path.exists(p)]


def compression_summary(input_path, outputs):
    """{original: (path, size), files: [(path, size)], total, ratio} of a compression."""
    original = os.path.getsize(input_path)
    files = [(p, os.path.getsize(p)) for p in outputs]
    total = sum(size for _, size in files)
    return {
        "original": (input_path, original),
        "files": files,
        "total": total,
        "ratio": original / max(total, 1),
    }


# --- Decompress


def archive_base(archive):
    """The archive's path without .vqms2: its side files are found by swapping the suffix."""
    if not archive.endswith(ARCHIVE):
        raise UserError(f"Expected a {ARCHIVE} file, got {archive}")
    return archive[: -len(ARCHIVE)]


def check_decompress(archive, out_format="mgf", requested_output="", stored_raw=False):
    """Refuse a decompression that cannot work: not an archive, no metadata next to it, or raw
    reconstructions asked for in mzML."""
    base = archive_base(archive)
    if not os.path.exists(base + METADATA):
        raise UserError(f"Metadata file not found next to it: {base}{METADATA}")
    requested = (requested_output or "").strip().lower()
    if stored_raw and (out_format.lower() == "mzml" or requested.endswith(".mzml")):
        raise UserError("Raw reconstructions can only be written to MGF.")


def side_files(archive):
    """The side files a decompression of archive reads."""
    base = archive_base(archive)
    files = [base + METADATA]
    if os.path.exists(base + LEGACY_COMPOUNDED):
        files.append(base + LEGACY_COMPOUNDED)
    return files


def decompressed_name(archive, out_format):
    return f"{Path(archive).stem}_decompressed.{out_format}"


def decompress(
    model,
    archive,
    output_path,
    quantizers,
    method=None,
    level=None,
    batch_size=128,
    stored_raw=False,
    verbose=1,
):
    """Decompress archive (and its side files) into output_path; returns output_path.

    quantizers, method and level must be the ones it was compressed with.
    """
    SpectrumDecompressor(
        inputFileName=archive,
        outputFilePath=output_path,
        model=model,
        batch_size=int(batch_size),
        quantizer=int(quantizers),
        stored_raw=stored_raw,
        **codec_levels(method, level),
    ).DecompressAll(verbose=verbose)
    return output_path


# --- Denoise


def denoised_name(input_path):
    return f"{Path(input_path).stem}_denoised.mgf"


def denoise(
    model,
    input_path,
    output_path,
    quantizers,
    retain_original_peaks=True,
    one_percent_threshold=True,
    compounded_spectra=True,
    num_workers=0,
):
    """Denoise an MGF into output_path; returns output_path."""
    MGFDenoiser(model=model, inputMGF=input_path, outputMGF=output_path).denoiseMGF(
        numQuantSingle=int(quantizers),
        retainOriginalPeaks=retain_original_peaks,
        OnePercentThreshold=one_percent_threshold,
        compoundedSpectra=compounded_spectra,
        num_workers=num_workers,
    )
    return output_path


def denoise_counts(input_path, output_path):
    """(input spectra, output spectra): chimeric spectra add extra ones at the end."""
    return count_spectra(input_path), count_spectra(output_path)
