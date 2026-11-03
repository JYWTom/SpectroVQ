"""MGF reader/writer (pyteomics)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from pyteomics import mgf

from .base import Spectrum, SpectrumReader, SpectrumWriter
from .registry import register_reader, register_writer


@register_reader
class MGFReader(SpectrumReader):
    extensions = (".mgf",)

    def __iter__(self) -> Iterator[Spectrum]:
        with mgf.read(self.path) as reader:
            for s in reader:
                yield Spectrum(s["m/z array"], s["intensity array"], s["params"])


@register_writer
class MGFWriter(SpectrumWriter):
    extensions = (".mgf",)

    def write(self, spectra: Iterable[Spectrum]) -> None:
        mgf.write(
            (
                {
                    "m/z array": s.mz,
                    "intensity array": np.asarray(s.intensity),
                    "params": s.params,
                }
                for s in spectra
            ),
            self.path,
        )
