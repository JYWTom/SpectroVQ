# Importing a format module registers it.
from . import mgf, mzml  # noqa: F401
from .base import Spectrum, SpectrumReader, SpectrumWriter
from .registry import (
    get_reader,
    get_writer,
    register_reader,
    register_writer,
    supported_input_extensions,
    supported_output_extensions,
)

__all__ = [
    "Spectrum",
    "SpectrumReader",
    "SpectrumWriter",
    "get_reader",
    "get_writer",
    "register_reader",
    "register_writer",
    "supported_input_extensions",
    "supported_output_extensions",
]
