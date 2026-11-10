"""Settings a .vqms2 can only be decoded with, stored in its .parquet side file.

The packed code stream carries no header, so decoding it with a different
quantizer count, codec or model code length either fails late or silently
produces wrong spectra. The compressor records those settings as parquet
schema metadata and the decompressor checks them before reading any codes.
Files written before this existed have no entry and are not checked.
"""

from __future__ import annotations

import json

import pyarrow.parquet as pq

KEY = b"spectrovq"
FORMAT_VERSION = 1
CODE_BITS = 10  # BitPacker and the zstd packer both store 10-bit codes
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
