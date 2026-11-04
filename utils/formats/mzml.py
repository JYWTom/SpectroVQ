"""mzML reader (pyteomics) and writer.

Spectra are mapped to the same MGF-style ``params`` the rest of the pipeline
expects (``title``, ``rtinseconds``, ``pepmass``, ``charge``), so a spectrum
read from mzML and the same spectrum read from an MGF converted from it are
interchangeable downstream. The full mzML metadata travels separately in
``Spectrum.mzml``.

The compressor stores a precursor per spectrum, so only MSn (level >= 2)
scans are usable. MS1 scans have no precursor and are skipped, which means an
mzML written here never contains MS1 scans or chromatograms.
"""

from __future__ import annotations

import base64
import hashlib
import os
import warnings
import zlib
from collections.abc import Iterable, Iterator
from xml.sax.saxutils import quoteattr

import numpy as np
from lxml import etree
from pyteomics import mzml

from .base import RunMetadata, Spectrum, SpectrumReader, SpectrumWriter
from .mzml_metadata import (
    BASE_PEAK_INTENSITY,
    BASE_PEAK_MZ,
    DERIVED_ACCESSIONS,
    HIGHEST_MZ,
    LOWEST_MZ,
    TOTAL_ION_CURRENT,
    ParamKey,
    flatten_spectrum,
    read_header,
)
from .registry import register_reader, register_writer

MZML_NS = "http://psi.hupo.org/ms/mzml"
SOFTWARE_ID = "SpectroVQ"
SOFTWARE_VERSION = "0.1.0"
# mzML terms for content this writer never outputs: MS1 scans and the TIC.
_DROPPED_FILE_CONTENT = ("MS:1000579", "MS:1000235")
# XSD child order, for elements with more than one kind of child.
_CHILD_ORDER = {
    "scanList": 0,
    "precursorList": 1,
    "productList": 2,
    "isolationWindow": 0,
    "selectedIonList": 1,
    "activation": 2,
}
_INDEXED_OPEN = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    f'<indexedmzML xmlns="{MZML_NS}" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    f'xsi:schemaLocation="{MZML_NS} '
    'http://psidev.info/files/ms/mzML/xsd/mzML1.1.2_idx.xsd">\n'
)


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


def _spectrum_elements(path: str) -> Iterator:
    """Yield raw ``<spectrum>`` elements, releasing each one after use."""
    for _, element in etree.iterparse(path, tag="{*}spectrum", huge_tree=True):
        yield element
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]


@register_reader
class MzMLReader(SpectrumReader):
    extensions = (".mzml",)

    def __init__(self, path: str, ms_level: int = 2):
        super().__init__(path)
        if ms_level < 2:
            raise ValueError("ms_level must be >= 2: MS1 scans have no precursor")
        self.ms_level = ms_level

    def __iter__(self) -> Iterator[Spectrum]:
        run = os.path.splitext(os.path.basename(self.path))[0]
        # pyteomics decodes the arrays; its parsed dicts drop accessions and
        # element structure, so the metadata comes from the raw elements.
        with mzml.read(self.path) as reader:
            for s, element in zip(reader, _spectrum_elements(self.path), strict=True):
                if s.get("ms level") != self.ms_level:
                    continue
                spectrum = self._convert(s, run)
                spectrum.mzml = flatten_spectrum(element)
                yield spectrum

    def run_metadata(self) -> RunMetadata:
        return RunMetadata(read_header(self.path))

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


class _Node:
    """One element of a spectrum being rebuilt from its flattened values."""

    __slots__ = ("attrs", "children", "params")

    def __init__(self):
        self.attrs: dict[str, str] = {}
        self.params: list[tuple[ParamKey, str]] = []
        self.children: dict[tuple[str, int], _Node] = {}


def _build_tree(values: dict[ParamKey, str]) -> _Node:
    root = _Node()
    for key, value in values.items():
        node = root
        if key.path:
            for segment in key.path.split("/"):
                tag, _, index = segment.rpartition(".")
                node = node.children.setdefault((tag, int(index)), _Node())
        if key.kind == "attr":
            node.attrs[key.name] = value
        elif key.kind != "node":
            node.params.append((key, value))
    return root


def _render_param(key: ParamKey, value: str, derived: dict[str, float]) -> str | None:
    if key.kind == "ref":
        return f"<referenceableParamGroupRef ref={quoteattr(key.name)}/>"
    unit = ""
    if key.unit_accession:
        unit = (
            f" unitCvRef={quoteattr(key.unit_cv_ref)}"
            f" unitAccession={quoteattr(key.unit_accession)}"
            f" unitName={quoteattr(key.unit_name)}"
        )
    if key.kind == "user":
        kind = f" type={quoteattr(key.xsd_type)}" if key.xsd_type else ""
        return f"<userParam name={quoteattr(key.name)} value={quoteattr(value)}{kind}{unit}/>"
    if key.accession in DERIVED_ACCESSIONS and key.path == "":
        if key.accession not in derived:  # no peaks left to compute it from
            return None
        value = repr(derived[key.accession])
    return (
        f"<cvParam cvRef={quoteattr(key.cv_ref)} accession={quoteattr(key.accession)}"
        f" name={quoteattr(key.name)} value={quoteattr(value)}{unit}/>"
    )


def _param_rank(key: ParamKey) -> int:
    return {"ref": 0, "cv": 1, "user": 2}[key.kind]


def _render(
    tag: str,
    node: _Node,
    local_ids: set[str],
    source_file: str | None,
    derived: dict[str, float],
    extra_attrs: str = "",
    tail: str = "",
) -> str:
    attrs = dict(node.attrs)
    # A parent scan that is not in this file (MS1 never is) has to be an external
    # reference: the mzML schema requires spectrumRef to resolve within the file.
    target = attrs.get("spectrumRef")
    if tag in ("precursor", "scan") and target is not None and target not in local_ids:
        del attrs["spectrumRef"]
        if source_file is None:
            warnings.warn(f"No sourceFile to reference, dropping parent scan {target}")
        else:
            attrs["sourceFileRef"] = source_file
            attrs["externalSpectrumID"] = target
    if tag.endswith("List"):
        attrs = {"count": str(len(node.children)), **attrs}
    rendered_attrs = "".join(f" {k}={quoteattr(v)}" for k, v in attrs.items())

    parts = []
    for key, value in sorted(node.params, key=lambda p: _param_rank(p[0])):
        param = _render_param(key, value, derived)
        if param is not None:
            parts.append(param)
    children = sorted(
        node.children.items(), key=lambda c: (_CHILD_ORDER.get(c[0][0], 0), c[0][1])
    )
    for (child_tag, _), child in children:
        parts.append(_render(child_tag, child, local_ids, source_file, derived))
    parts.append(tail)

    body = "\n".join(part for part in parts if part)
    open_tag = f"<{tag}{extra_attrs}{rendered_attrs}"
    return f"{open_tag}>\n{body}\n</{tag}>\n" if body else f"{open_tag}/>\n"


def _binary_array(data: np.ndarray, precision: tuple[str, str], kind: str) -> str:
    encoded = base64.b64encode(zlib.compress(data.tobytes())).decode("ascii")
    return (
        f'<binaryDataArray encodedLength="{len(encoded)}">\n'
        f'<cvParam cvRef="MS" accession="{precision[0]}" name="{precision[1]}" value=""/>\n'
        '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>\n'
        f"{kind}\n"
        f"<binary>{encoded}</binary>\n"
        "</binaryDataArray>"
    )


def _derived_values(mz: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    if len(mz) == 0:
        return {}
    base_peak = int(np.argmax(intensity))
    return {
        BASE_PEAK_MZ: float(mz[base_peak]),
        BASE_PEAK_INTENSITY: float(intensity[base_peak]),
        TOTAL_ION_CURRENT: float(intensity.sum(dtype=np.float64)),
        LOWEST_MZ: float(mz.min()),
        HIGHEST_MZ: float(mz.max()),
    }


def _intensity_unit(values: dict[ParamKey, str]) -> str:
    for key in values:
        if key.accession == BASE_PEAK_INTENSITY and key.unit_accession:
            return (
                f" unitCvRef={quoteattr(key.unit_cv_ref)}"
                f" unitAccession={quoteattr(key.unit_accession)}"
                f" unitName={quoteattr(key.unit_name)}"
            )
    return ' unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"'


def _spectrum_xml(
    index: int, spectrum: Spectrum, local_ids: set[str], source_file: str | None
) -> str:
    mz = np.asarray(spectrum.mz, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f4")
    arrays = (
        '<binaryDataArrayList count="2">\n'
        + _binary_array(
            mz,
            ("MS:1000523", "64-bit float"),
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""'
            ' unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>',
        )
        + "\n"
        + _binary_array(
            intensity,
            ("MS:1000521", "32-bit float"),
            '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""'
            f"{_intensity_unit(spectrum.mzml)}/>",
        )
        + "\n</binaryDataArrayList>"
    )
    return _render(
        "spectrum",
        _build_tree(spectrum.mzml),
        local_ids,
        source_file,
        _derived_values(mz, intensity),
        extra_attrs=f' index="{index}" defaultArrayLength="{len(mz)}"',
        tail=arrays,
    )


def _add_processing(root, ms_levels: list[str], lossy_params: dict[str, str]) -> None:
    """Record the MS1 removal and any lossy reconstruction in the processing history."""
    ns = {"m": MZML_NS}
    software_list = root.find("m:softwareList", ns)
    processing_list = root.find("m:dataProcessingList", ns)
    if software_list is None or processing_list is None:
        raise ValueError("mzML header has no softwareList or dataProcessingList")

    if software_list.find(f"m:software[@id='{SOFTWARE_ID}']", ns) is None:
        software = etree.SubElement(
            software_list,
            f"{{{MZML_NS}}}software",
            id=SOFTWARE_ID,
            version=SOFTWARE_VERSION,
        )
        etree.SubElement(
            software,
            f"{{{MZML_NS}}}cvParam",
            cvRef="MS",
            accession="MS:1000799",
            name="custom unreleased software tool",
            value="SpectroVQ",
        )
        software_list.set("count", str(len(software_list)))

    # "data filtering" is the closest PSI-MS term: the CV has none for VQ or lossy
    # compression, and the mzML semantic rules require a CV term on each method.
    steps = [{"ms levels": ",".join(ms_levels)}]
    if lossy_params:
        steps.append({"lossy vector-quantized reconstruction": "", **lossy_params})
    for processing in processing_list.findall("m:dataProcessing", ns):
        order = 1 + max(
            (
                int(m.get("order", -1))
                for m in processing.findall("m:processingMethod", ns)
            ),
            default=-1,
        )
        for offset, step in enumerate(steps):
            method = etree.SubElement(
                processing,
                f"{{{MZML_NS}}}processingMethod",
                order=str(order + offset),
                softwareRef=SOFTWARE_ID,
            )
            etree.SubElement(
                method,
                f"{{{MZML_NS}}}cvParam",
                cvRef="MS",
                accession="MS:1001486",
                name="data filtering",
                value="",
            )
            for name, value in step.items():
                etree.SubElement(
                    method, f"{{{MZML_NS}}}userParam", name=name, value=value
                )


def _edit_header(
    run: RunMetadata, count: int, ms_levels: list[str]
) -> tuple[str, str | None]:
    """Make the stored header describe the written file: returns (text, source file id)."""
    root = etree.fromstring(f"{run.header}</spectrumList></run></mzML>".encode())
    ns = {"m": MZML_NS}
    for accession in _DROPPED_FILE_CONTENT:
        for term in root.findall(
            f"m:fileDescription/m:fileContent/m:cvParam[@accession='{accession}']", ns
        ):
            term.getparent().remove(term)
    _add_processing(root, ms_levels, run.lossy_params)

    run_element = root.find("m:run", ns)
    source_file = run_element.get("defaultSourceFileRef")
    if source_file is None:
        first = root.find("m:fileDescription/m:sourceFileList/m:sourceFile", ns)
        source_file = None if first is None else first.get("id")
    spectrum_list = run_element.find("m:spectrumList", ns)
    spectrum_list.set("count", str(count))

    etree.indent(root, space="  ")
    text = etree.tostring(root, encoding="unicode")
    # Keep the serialized text up to and including the spectrumList open tag.
    end = text.index("<spectrumList")
    end = text.index(">", end)
    return text[:end].rstrip("/") + ">\n", source_file


@register_writer
class MzMLWriter(SpectrumWriter):
    """Writes MS2-only indexedmzML using metadata captured from a source mzML."""

    extensions = (".mzml",)

    def write(
        self, spectra: Iterable[Spectrum], run: RunMetadata | None = None
    ) -> None:
        if run is None:
            raise ValueError(
                "mzML output needs metadata captured from an mzML input; "
                "this source has none. Write MGF instead."
            )
        spectra = list(spectra)
        ids = [s.mzml.get(ParamKey("", "attr", "id")) for s in spectra]
        if None in ids:
            raise ValueError("mzML output needs per-spectrum mzML metadata")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate spectrum ids; mzML requires them to be unique")
        local_ids = set(ids)
        ms_levels = sorted(
            {
                v
                for s in spectra
                for k, v in s.mzml.items()
                if k.accession == "MS:1000511"
            }
        )
        header, source_file = _edit_header(run, len(spectra), ms_levels)

        digest = hashlib.sha1()
        position = 0

        def emit(handle, text: str) -> None:
            nonlocal position
            data = text.encode("utf-8")
            handle.write(data)
            digest.update(data)
            position += len(data)

        offsets = []
        with open(self.path, "wb") as f:
            emit(f, _INDEXED_OPEN + header)
            for index, spectrum in enumerate(spectra):
                offsets.append((ids[index], position))
                emit(f, _spectrum_xml(index, spectrum, local_ids, source_file))
            emit(f, "</spectrumList>\n</run>\n</mzML>\n")
            index_offset = position
            entries = "\n".join(
                f"<offset idRef={quoteattr(spectrum_id)}>{offset}</offset>"
                for spectrum_id, offset in offsets
            )
            emit(
                f,
                '<indexList count="1">\n<index name="spectrum">\n'
                f"{entries}\n</index>\n</indexList>\n"
                f"<indexListOffset>{index_offset}</indexListOffset>\n<fileChecksum>",
            )
            # The checksum covers everything up to and including <fileChecksum>.
            f.write(f"{digest.hexdigest()}</fileChecksum>\n</indexedmzML>\n".encode())
