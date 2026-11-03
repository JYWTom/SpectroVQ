"""Extension -> reader/writer lookup."""

from __future__ import annotations

import os

from .base import SpectrumReader, SpectrumWriter

_READERS: dict[str, type[SpectrumReader]] = {}
_WRITERS: dict[str, type[SpectrumWriter]] = {}


def _ext(path: str) -> str:
    return os.path.splitext(os.fspath(path))[1].lower()


def register_reader(cls: type[SpectrumReader]) -> type[SpectrumReader]:
    """Class decorator: register ``cls`` for each extension in ``cls.extensions``."""
    for ext in cls.extensions:
        _READERS[ext.lower()] = cls
    return cls


def register_writer(cls: type[SpectrumWriter]) -> type[SpectrumWriter]:
    for ext in cls.extensions:
        _WRITERS[ext.lower()] = cls
    return cls


def _lookup(table, path, kind):
    ext = _ext(path)
    try:
        return table[ext]
    except KeyError:
        raise ValueError(
            f"Unsupported {kind} format '{ext or path}'. "
            f"Supported: {', '.join(sorted(table)) or 'none'}"
        ) from None


def get_reader(path: str) -> SpectrumReader:
    return _lookup(_READERS, path, "input")(path)


def get_writer(path: str) -> SpectrumWriter:
    return _lookup(_WRITERS, path, "output")(path)


def supported_input_extensions():
    return sorted(_READERS)


def supported_output_extensions():
    return sorted(_WRITERS)
