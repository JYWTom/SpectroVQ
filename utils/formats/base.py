"""Format-neutral spectrum record and reader/writer interfaces.

A file format only has to know how to turn a file into an iterable of
``Spectrum`` objects (reader) and an iterable of ``Spectrum`` objects into a
file (writer). Everything downstream (VQ encoding, bit packing, parquet
metadata) works on ``Spectrum`` and is independent of the file format.
"""

from __future__ import annotations

import itertools
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class Spectrum:
    """One spectrum.

    ``params`` uses MGF-style keys (the canonical metadata layout stored in
    the .parquet side file): ``title``, ``pepmass`` (m/z, intensity) tuple,
    ``charge``, ``rtinseconds``, ``scans``, ... Format specific extras may be
    added and are round-tripped as-is.

    ``mzml`` holds the source mzML metadata (see ``mzml_metadata``). It is kept
    apart from ``params`` so it never reaches formats that cannot represent it.
    """

    mz: np.ndarray
    intensity: np.ndarray
    params: dict = field(default_factory=dict)
    mzml: dict = field(default_factory=dict)


@dataclass
class RunMetadata:
    """File-level metadata captured from the source file.

    ``header`` is the source mzML text up to the first spectrum. ``lossy_params``
    describes a lossy reconstruction, recorded in the output's processing history.
    """

    header: str
    lossy_params: dict[str, str] = field(default_factory=dict)


class SpectrumReader(ABC):
    """Reads spectra from a file. Subclasses implement ``__iter__``."""

    extensions: ClassVar[tuple[str, ...]] = ()

    def __init__(self, path: str):
        self.path = os.fspath(path)

    @abstractmethod
    def __iter__(self) -> Iterator[Spectrum]: ...

    def read_all(self, limit: int | None = None) -> list[Spectrum]:
        """Load (up to ``limit``) spectra into memory."""
        return list(itertools.islice(iter(self), limit))

    def run_metadata(self) -> RunMetadata | None:
        """File-level metadata, for formats that have any."""
        return None


class SpectrumWriter(ABC):
    """Writes spectra to a file. Subclasses implement ``write``."""

    extensions: ClassVar[tuple[str, ...]] = ()

    def __init__(self, path: str):
        self.path = os.fspath(path)

    @abstractmethod
    def write(
        self, spectra: Iterable[Spectrum], run: RunMetadata | None = None
    ) -> None: ...
