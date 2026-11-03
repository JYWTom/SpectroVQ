"""mzML reader (pyteomics).

Spectra are mapped to the same MGF-style ``params`` the rest of the pipeline
expects (``title``, ``rtinseconds``, ``pepmass``, ``charge``), so a spectrum
read from mzML and the same spectrum read from an MGF converted from it are
interchangeable downstream.

The compressor stores a precursor per spectrum, so only MSn (level >= 2)
scans are usable. MS1 scans have no precursor and are skipped.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
from pyteomics import mzml

from .base import Spectrum, SpectrumReader
from .registry import register_reader


def _scan_number(native_id: str, index: int) -> int:
    """Scan number from a native id such as ``... scan=3091`` (falls back to index+1)."""
    for part in native_id.split():
        if part.startswith("scan="):
            return int(part[5:])
    return index + 1


def _retention_seconds(scan: dict) -> float | None:
    rt = scan.get("scan start time")
    if rt is None:
        return None
    unit = getattr(rt, "unit_info", None) or "minute"
    return float(rt) * (1.0 if unit == "second" else 60.0)


class MzMLReader(SpectrumReader):
    extensions = (".mzml",)

    def __init__(self, path: str, ms_level: int = 2):
        super().__init__(path)
        if ms_level < 2:
            raise ValueError("ms_level must be >= 2: MS1 scans have no precursor")
        self.ms_level = ms_level

    def __iter__(self) -> Iterator[Spectrum]:
        run = os.path.splitext(os.path.basename(self.path))[0]
        with mzml.read(self.path) as reader:
            for s in reader:
                if s.get("ms level") != self.ms_level:
                    continue
                yield self._convert(s, run)

    def _convert(self, s: dict, run: str) -> Spectrum:
        scan_no = _scan_number(s["id"], s["index"])
        params: dict = {}

        precursors = s.get("precursorList", {}).get("precursor", [])
        ion = (
            precursors[0].get("selectedIonList", {}).get("selectedIon", [{}])[0]
            if precursors
            else {}
        )
        charge = ion.get("charge state")
        title = f"{run}.{scan_no}.{scan_no}"
        if charge is not None:
            title += f".{int(charge)}"
        params["title"] = title

        rt = _retention_seconds(s.get("scanList", {}).get("scan", [{}])[0])
        if rt is not None:
            params["rtinseconds"] = rt
        if "selected ion m/z" in ion:
            params["pepmass"] = (ion["selected ion m/z"], ion.get("peak intensity"))
        if charge is not None:
            params["charge"] = [int(charge)]

        return Spectrum(
            np.asarray(s["m/z array"], dtype=np.float64),
            np.asarray(s["intensity array"], dtype=np.float64),
            params,
        )


register_reader(MzMLReader)
