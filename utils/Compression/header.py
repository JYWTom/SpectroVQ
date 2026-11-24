"""Settings a .vqms2 can only be decoded with, stored in its .parquet side file.

The packed code stream carries no header, so decoding it with a different
quantizer count, codec or model code length either fails late or silently
produces wrong spectra. The compressor records those settings as parquet
schema metadata and the decompressor checks them before reading any codes.
Files written before this existed have no entry and are not checked.
"""

from __future__ import annotations

import json
import os
import pickle

import pyarrow.parquet as pq

from ..config import CODE_BITS  # re-exported for callers

KEY = b"spectrovq"
FORMAT_VERSION = 1
MAX_CODEBOOK_SIZE = 1 << CODE_BITS


def codec_name(gzip_level=None, zlib_level=None, zstd_level=None) -> str:
    """The codec selected by the mutually exclusive *_compression_level arguments."""
    chosen = [
        name
        for name, level in (
            ("gzip", gzip_level),
            ("zlib", zlib_level),
            ("zstd", zstd_level),
        )
        if level is not None
    ]
    if len(chosen) > 1:
        raise ValueError("Cannot use multiple compression methods simultaneously")
    return chosen[0] if chosen else "none"


def check_codebook(model) -> None:
    size = getattr(model, "codebook_size", None)
    if size is not None and size > MAX_CODEBOOK_SIZE:
        raise ValueError(
            f"codebook_size {size} does not fit the {CODE_BITS}-bit codes .vqms2 "
            f"stores; use a model with codebook_size <= {MAX_CODEBOOK_SIZE}"
        )


def check_quantizer(model, quantizer: int) -> None:
    """Raise before any encoding if the model has fewer quantizers than requested."""
    n_q = getattr(model, "n_q", None)
    if not 1 <= int(quantizer) <= (n_q or int(quantizer)):
        raise ValueError(
            f"quantizer must be between 1 and the model's n_q ({n_q}), got {quantizer}"
        )


def make(quantizer: int, codec: str, quantizedlen: int) -> dict[bytes, bytes]:
    settings = {
        "format": FORMAT_VERSION,
        "quantizer": int(quantizer),
        "codec": codec,
        "quantizedlen": int(quantizedlen),
        "bits": CODE_BITS,
    }
    return {KEY: json.dumps(settings).encode()}


def read(parquet_path: str) -> dict | None:
    metadata = pq.read_schema(parquet_path).metadata or {}
    raw = metadata.get(KEY)
    return json.loads(raw) if raw is not None else None


def check(parquet_path: str, quantizer: int, codec: str, quantizedlen: int) -> None:
    """Raise if the given decoding settings differ from the recorded ones."""
    stored = read(parquet_path)
    if stored is None:
        return
    given = {
        "quantizer": int(quantizer),
        "codec": codec,
        "quantizedlen": int(quantizedlen),
    }
    wrong = {k: v for k, v in given.items() if stored.get(k) != v}
    if not wrong:
        return
    hints = {
        "quantizer": f"--quantizer {stored.get('quantizer')}",
        "codec": f"--compression_method {stored.get('codec')}"
        if stored.get("codec") != "none"
        else "no --compression_method",
        "quantizedlen": f"a model whose code length is {stored.get('quantizedlen')} "
        "(the same YAML config)",
    }
    details = ", ".join(f"{k} {given[k]} (recorded: {stored.get(k)})" for k in wrong)
    raise ValueError(
        f"{parquet_path} was compressed with different settings: {details}. "
        f"Decompress with {' and '.join(hints[k] for k in wrong)}."
    )


# Indices of the spectra whose residual was encoded a second time ("compounded"); their
# extra codes follow the main ones in the .vqms2. Stored in the parquet footer, so the
# count always matches the archive. Older archives kept them in <name>_compounded.pkl.
COMPOUNDED_KEY = b"spectrovq.compounded"
LEGACY_COMPOUNDED_SUFFIX = "_compounded.pkl"


def compounded_entry(indices) -> dict[bytes, bytes]:
    return {COMPOUNDED_KEY: json.dumps([int(i) for i in indices]).encode()}


def read_compounded(footer: dict, stem: str) -> list[int]:
    """The compounded indices of an archive (stem: its path without extension).

    The footer entry wins, so a stale _compounded.pkl next to a newer archive is ignored.
    A legacy pickle is read without allowing any class or function, so it can hold only
    plain lists and ints and cannot run code.
    """
    if COMPOUNDED_KEY in footer:
        return [int(i) for i in json.loads(footer[COMPOUNDED_KEY])]
    path = stem + LEGACY_COMPOUNDED_SUFFIX
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        try:
            indices = _PlainUnpickler(f).load()
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            raise ValueError(
                f"{path} does not hold a plain list of indices: {e}"
            ) from e
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError(f"{path} does not hold a plain list of indices")
    return indices


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
