"""mzML reader and writer (lxml; zlib or uncompressed binary arrays, no numpress).

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


# Accessions the reader needs, read straight from the raw elements.
_MS_LEVEL = "MS:1000511"
_SCAN_START_TIME = "MS:1000016"
_SELECTED_ION_MZ = ("MS:1000744", "MS:1000040")  # the second is mzML 1.0's "m/z"
_CHARGE_STATE = "MS:1000041"
_PEAK_INTENSITY = "MS:1000042"
_MZ_ARRAY = "MS:1000514"
_INTENSITY_ARRAY = "MS:1000515"
_DTYPES = {
    "MS:1000521": "<f4",  # 32-bit float
    "MS:1000523": "<f8",  # 64-bit float
    "MS:1000519": "<i4",  # 32-bit integer
    "MS:1000522": "<i8",  # 64-bit integer
}
_NO_COMPRESSION, _ZLIB = "MS:1000576", "MS:1000574"
# Seconds per time unit, by unit accession and by unit name; minutes when none is given.
_SECONDS = {
    "UO:0000010": 1.0,
    "UO:0000031": 60.0,
    "UO:0000028": 1e-3,
    "UO:0000032": 3600.0,
    "second": 1.0,
    "minute": 60.0,
    "millisecond": 1e-3,
    "hour": 3600.0,
}

# accession -> (value, unit accession, unit name)
_CvParams = dict[str, tuple[str, str, str]]


def _param_groups(header: str) -> dict[str, list]:
    """referenceableParamGroup id -> its cvParam elements, from the run header."""
    try:
        root = etree.fromstring(f"{header}</spectrumList></run></mzML>".encode())
    except etree.XMLSyntaxError as e:
        # Not silently empty: params kept in groups (ms level, array encoding) would be
        # missed, skipping spectra or decoding their peaks wrongly.
        raise ValueError(f"Cannot parse the mzML header: {e}") from e
    return {
        group.get("id"): [c for c in group if _is(c, "cvParam")]
        for group in root.iterfind(".//{*}referenceableParamGroup")
    }


def _is(element, tag: str) -> bool:
    return isinstance(element.tag, str) and etree.QName(element).localname == tag


def _cv_params(element, groups: dict[str, list]) -> _CvParams:
    """The cvParams of an element, including those of the param groups it refers to
    (its own take precedence)."""
    found: _CvParams = {}
    for child in element:
        if _is(child, "referenceableParamGroupRef"):
            for param in groups.get(child.get("ref"), ()):
                found.setdefault(param.get("accession", ""), _cv_value(param))
        elif _is(child, "cvParam"):
            found[child.get("accession", "")] = _cv_value(child)
    return found


def _cv_value(param) -> tuple[str, str, str]:
    return (
        param.get("value", ""),
        param.get("unitAccession", ""),
        param.get("unitName", ""),
    )


def _retention_seconds(scan, groups) -> float | None:
    if scan is None:
        return None
    rt = _cv_params(scan, groups).get(_SCAN_START_TIME)
    if rt is None:
        return None
    value, unit_accession, unit_name = rt
    seconds = _number(value)
    if seconds is None:
        return None
    return seconds * (_SECONDS.get(unit_accession) or _SECONDS.get(unit_name, 60.0))


def _number(value: str) -> float | None:
    """A cvParam value as a float; None when it is empty or not a number."""
    try:
        return float(value)
    except ValueError:
        return None


def _decode_array(array, groups, spectrum_id: str) -> tuple[str, np.ndarray] | None:
    """(m/z or intensity accession, values) of one binaryDataArray; None for any other
    array (ion mobility, ...), which is never decoded."""
    params = _cv_params(array, groups)
    kind = next((a for a in (_MZ_ARRAY, _INTENSITY_ARRAY) if a in params), None)
    if kind is None:
        return None
    dtype = next((d for a, d in _DTYPES.items() if a in params), "<f8")
    # Numpress and zstd are not supported; read as raw values they would be garbage.
    names = " ".join(_param_names(array, groups)).lower()
    if "numpress" in names or "zstd" in names:
        raise ValueError(
            f"Spectrum {spectrum_id!r} uses an unsupported binary encoding; convert "
            "the file with zlib or no compression (e.g. msconvert --zlib)"
        )
    data = base64.b64decode(array.findtext("{*}binary") or "")
    # Otherwise uncompressed, as when MS:1000576 is given or the term is left out.
    # An empty <binary/> is an empty array, also when flagged as zlib.
    if data and _ZLIB in params:
        data = zlib.decompress(data)
    return kind, np.frombuffer(data, dtype=dtype).astype(np.float64)


def _param_names(element, groups) -> list[str]:
    names = []
    for child in element:
        if _is(child, "referenceableParamGroupRef"):
            names += [p.get("name", "") for p in groups.get(child.get("ref"), ())]
        elif _is(child, "cvParam"):
            names.append(child.get("name", ""))
    return names


def _spectrum_elements(path: str) -> Iterator:
    """Yield raw ``<spectrum>`` elements, releasing each one after use."""
    # huge_tree for large <binary> nodes; no entity expansion, which mzML never needs.
    for _, element in etree.iterparse(
        path, tag="{*}spectrum", huge_tree=True, resolve_entities=False
    ):
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
        # One pass over the raw elements gives both the peaks and the full metadata
        # (pyteomics would need a second parse: its dicts drop accessions and structure).
        groups = _param_groups(read_header(self.path))
        for element in _spectrum_elements(self.path):
            level = _cv_params(element, groups).get(_MS_LEVEL)
            if (
                level is None
                or not level[0].isdigit()
                or int(level[0]) != self.ms_level
            ):
                continue
            spectrum = self._convert(element, groups, run)
            spectrum.mzml = flatten_spectrum(element)
            yield spectrum

    def run_metadata(self) -> RunMetadata:
        return RunMetadata(read_header(self.path))

    def _convert(self, element, groups: dict[str, list], run: str) -> Spectrum:
        spectrum_id = element.get("id", "")
        scan_no = _scan_number(spectrum_id, int(element.get("index", 0)))
        params: dict = {}

        ion = element.find(
            "{*}precursorList/{*}precursor/{*}selectedIonList/{*}selectedIon"
        )
        ion_params = _cv_params(ion, groups) if ion is not None else {}
        # Empty or non-numeric values (an undetermined charge) count as absent
        charge = _number(ion_params.get(_CHARGE_STATE, ("",))[0])
        charge = int(charge) if charge is not None else None
        title = f"{run}.{scan_no}.{scan_no}"
        if charge is not None:
            title += f".{charge}"
        params["title"] = title

        rt = _retention_seconds(element.find("{*}scanList/{*}scan"), groups)
        if rt is not None:
            params["rtinseconds"] = rt
        mz = next(
            (
                _number(ion_params[a][0])
                for a in _SELECTED_ION_MZ
                if a in ion_params and _number(ion_params[a][0]) is not None
            ),
            None,
        )
        if mz is not None:
            intensity = _number(ion_params.get(_PEAK_INTENSITY, ("",))[0])
            params["pepmass"] = (mz, intensity)
        if charge is not None:
            params["charge"] = [charge]

        decoded = (
            _decode_array(array, groups, spectrum_id)
            for array in element.iterfind("{*}binaryDataArrayList/{*}binaryDataArray")
        )
        arrays = dict(pair for pair in decoded if pair is not None)
        empty = np.empty(0, dtype=np.float64)
        return Spectrum(
            arrays.get(_MZ_ARRAY, empty),
            arrays.get(_INTENSITY_ARRAY, empty),
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
