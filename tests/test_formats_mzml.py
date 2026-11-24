import base64
import hashlib
import os
import re
import struct

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyteomics import mzml

from utils.formats import (
    RunMetadata,
    Spectrum,
    get_reader,
    get_writer,
    supported_input_extensions,
)
from utils.formats.mzml import MzMLReader
from utils.formats.mzml_metadata import (
    COLUMN_PREFIX,
    ParamKey,
    decode_row,
    encode_columns,
    load_definitions,
)
from utils.mgfHandler.decompressor import SpectrumDecompressor


def _b64(values):
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode()


def _spectrum(index, scan, level, mz, it, precursor=""):
    return f"""
    <spectrum index="{index}" id="controllerType=0 controllerNumber=1 scan={scan}" defaultArrayLength="{len(mz)}">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}"/>
      <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="1.5" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
      <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="1.5"/>
      <scanList count="1">
        <cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>
        <scan>
          <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="2.0" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
          <cvParam cvRef="MS" accession="MS:1000512" name="filter string" value="FTMS &amp; ms2 &lt;test&gt;"/>
          <userParam name="vendor extra" value="7" type="xsd:float"/>
        </scan>
      </scanList>
      {precursor}
      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="0">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
          <binary>{_b64(mz)}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="0">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
          <binary>{_b64(it)}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>"""


PRECURSOR = """<precursorList count="1"><precursor spectrumRef="controllerType=0 controllerNumber=1 scan=1">
        <selectedIonList count="1"><selectedIon>
          <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="371.5"/>
          <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>
          <cvParam cvRef="MS" accession="MS:1000042" name="peak intensity" value="1000.0"/>
        </selectedIon></selectedIonList></precursor></precursorList>"""

MZML = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <cvList count="1"><cv id="MS" fullName="MS" version="4.1.0" URI="x"/></cvList>
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>
      <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
    </fileContent>
    <sourceFileList count="1"><sourceFile id="RAW1" name="r.raw" location="file:///x"/></sourceFileList>
  </fileDescription>
  <softwareList count="1"><software id="pwiz" version="3"/></softwareList>
  <dataProcessingList count="1"><dataProcessing id="dp">
    <processingMethod order="0" softwareRef="pwiz"><cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/></processingMethod>
  </dataProcessing></dataProcessingList>
  <run id="r" defaultSourceFileRef="RAW1">
    <spectrumList count="2" defaultDataProcessingRef="dp">
      {_spectrum(0, 1, 1, [100.0, 200.0], [1.0, 2.0])}
      {_spectrum(1, 2, 2, [110.5, 220.25, 330.0], [5.0, 6.0, 7.0], PRECURSOR)}
    </spectrumList>
  </run>
</mzML>"""


@pytest.fixture
def mzml_path(tmp_path):
    p = tmp_path / "run1.mzML"
    p.write_text(MZML)
    return str(p)


def test_registered():
    assert ".mzml" in supported_input_extensions()


def test_reads_ms2_only_with_mgf_style_params(mzml_path):
    spectra = get_reader(mzml_path).read_all()
    assert len(spectra) == 1  # MS1 scan skipped
    s = spectra[0]
    np.testing.assert_allclose(s.mz, [110.5, 220.25, 330.0])
    np.testing.assert_allclose(s.intensity, [5.0, 6.0, 7.0])
    assert s.params["title"] == "run1.2.2.2"
    assert s.params["rtinseconds"] == pytest.approx(120.0)  # 2 min
    assert s.params["pepmass"] == (371.5, 1000.0)
    assert s.params["charge"] == [2]


def test_ms1_rejected(mzml_path):
    with pytest.raises(ValueError, match="ms_level"):
        MzMLReader(mzml_path, ms_level=1)


def _through_parquet(spectra, tmp_path):
    """Store flattened metadata the way the compressor does and read it back."""
    columns, definitions = encode_columns([s.mzml for s in spectra])
    path = tmp_path / "meta.parquet"
    pq.write_table(pa.table(columns), path)
    rows = pd.read_parquet(path)
    parsed = load_definitions(definitions)
    return [decode_row(rows.iloc[i], parsed) for i in range(len(rows))]


def _written(mzml_path, tmp_path, run=None):
    reader = get_reader(mzml_path)
    spectra = reader.read_all()
    for s, values in zip(spectra, _through_parquet(spectra, tmp_path), strict=True):
        s.mzml = values
    out = tmp_path / "out.mzML"
    get_writer(out).write(spectra, run or reader.run_metadata())
    return out


def test_reader_captures_metadata_and_header(mzml_path):
    reader = get_reader(mzml_path)
    values = reader.read_all()[0].mzml
    by_name = {key.name: value for key, value in values.items()}
    assert by_name["id"] == "controllerType=0 controllerNumber=1 scan=2"
    assert by_name["filter string"] == "FTMS & ms2 <test>"
    assert by_name["vendor extra"] == "7"
    assert by_name["base peak m/z"] == ""  # recomputed on output, never stored
    assert "<softwareList" in reader.run_metadata().header


def test_encode_picks_compact_types():
    def key(name):
        return ParamKey("", "attr", name)

    rows = [
        {key("id"): "scan=9", key("n"): "1.50", key("s"): "a", key("flag"): ""},
        {key("id"): "scan=10", key("n"): "2e3", key("s"): "007"},
    ]
    columns, definitions = encode_columns(rows)
    types = {d[3].name: d[1] for d in load_definitions(definitions)}
    assert types == {"id": "int", "n": "float", "s": "str", "flag": "flag"}
    assert columns[f"{COLUMN_PREFIX}id"].to_pylist() == [9, 10]


def test_values_with_leading_zeros_stay_exact():
    key = ParamKey("", "attr", "x")
    columns, definitions = encode_columns([{key: "007"}, {key: "12"}])
    row = {f"{COLUMN_PREFIX}x": columns[f"{COLUMN_PREFIX}x"][0].as_py()}
    assert decode_row(row, load_definitions(definitions)) == {key: "007"}


def test_round_trip_preserves_spectra_and_metadata(mzml_path, tmp_path):
    out = _written(mzml_path, tmp_path)
    original = get_reader(mzml_path).read_all()[0]
    rebuilt = get_reader(out).read_all()[0]
    np.testing.assert_allclose(rebuilt.mz, original.mz)
    np.testing.assert_allclose(rebuilt.intensity, original.intensity)
    assert rebuilt.params["pepmass"] == original.params["pepmass"]
    assert rebuilt.params["charge"] == original.params["charge"]
    assert rebuilt.params["rtinseconds"] == original.params["rtinseconds"]

    with mzml.read(str(out)) as reader:
        s = next(iter(reader))
    assert s["index"] == 0
    assert s["scanList"]["scan"][0]["filter string"] == "FTMS & ms2 <test>"
    assert s["scanList"]["scan"][0]["vendor extra"] == 7


def test_output_is_ms2_only_with_valid_parent_reference(mzml_path, tmp_path):
    text = _written(mzml_path, tmp_path).read_text()
    assert text.count("<spectrum ") == 1
    assert '<spectrumList count="1"' in text
    assert "MS:1000579" not in text
    assert "spectrumRef" not in text  # the schema forbids dangling references
    assert 'sourceFileRef="RAW1"' in text
    assert 'externalSpectrumID="controllerType=0 controllerNumber=1 scan=1"' in text


def test_local_parent_keeps_spectrum_ref(tmp_path):
    def attr(path, name):
        return ParamKey(path, "attr", name)

    parent = "scan=1"
    spectra = [
        Spectrum(
            np.array([100.0]),
            np.array([1.0]),
            mzml={
                attr("", "id"): "scan=2",
                ParamKey("", "cv", "ms level", "MS:1000511", "MS"): "3",
                attr("precursorList.0/precursor.0", "spectrumRef"): "scan=3",
            },
        ),
        Spectrum(
            np.array([100.0]),
            np.array([1.0]),
            mzml={
                attr("", "id"): "scan=3",
                attr("precursorList.0/precursor.0", "spectrumRef"): parent,
            },
        ),
    ]
    out = tmp_path / "out.mzML"
    header = MZML[MZML.index("<mzML") : MZML.index("<spectrum ")]
    get_writer(out).write(spectra, RunMetadata(header))
    text = out.read_text()
    assert 'spectrumRef="scan=3"' in text
    assert 'externalSpectrumID="scan=1"' in text


def test_derived_values_recomputed_from_written_peaks(mzml_path, tmp_path):
    with mzml.read(str(_written(mzml_path, tmp_path))) as reader:
        s = next(iter(reader))
    assert s["base peak m/z"] == pytest.approx(330.0)
    assert s["total ion current"] == pytest.approx(18.0)


def test_processing_history_records_filtering_and_lossy_step(mzml_path, tmp_path):
    reader = get_reader(mzml_path)
    run = RunMetadata(
        reader.run_metadata().header, {"model sha256": "abc", "quantizers": "4"}
    )
    text = _written(mzml_path, tmp_path, run).read_text()
    assert '<software id="SpectroVQ"' in text
    assert '<softwareList count="2"' in text
    assert '<processingMethod order="1" softwareRef="SpectroVQ"' in text
    assert '<userParam name="ms levels" value="2"' in text
    assert '<processingMethod order="2" softwareRef="SpectroVQ"' in text
    assert '<userParam name="model sha256" value="abc"' in text


def test_index_and_checksum_match_file(mzml_path, tmp_path):
    data = _written(mzml_path, tmp_path).read_bytes()
    with mzml.MzML(str(tmp_path / "out.mzML"), use_index=True) as indexed:
        assert (
            indexed.get_by_id("controllerType=0 controllerNumber=1 scan=2")["ms level"]
            == 2
        )
    offset = int(re.search(rb'<offset idRef="[^"]*">(\d+)<', data).group(1))
    assert data[offset:].startswith(b"<spectrum ")
    list_offset = int(re.search(rb"<indexListOffset>(\d+)<", data).group(1))
    assert data[list_offset:].startswith(b"<indexList")
    end = data.index(b"<fileChecksum>") + len(b"<fileChecksum>")
    checksum = re.search(rb"<fileChecksum>(\w+)<", data).group(1).decode()
    assert hashlib.sha1(data[:end]).hexdigest() == checksum


def test_writer_needs_mzml_source(tmp_path):
    spectrum = Spectrum(np.array([100.0]), np.array([1.0]), {"title": "a"})
    with pytest.raises(ValueError, match="mzML input"):
        get_writer(tmp_path / "out.mzML").write([spectrum])
    with pytest.raises(ValueError, match="per-spectrum"):
        get_writer(tmp_path / "out.mzML").write([spectrum], RunMetadata(MZML))


def test_writer_rejects_duplicate_ids(mzml_path, tmp_path):
    reader = get_reader(mzml_path)
    s = reader.read_all()[0]
    with pytest.raises(ValueError, match="unique"):
        get_writer(tmp_path / "out.mzML").write([s, s], reader.run_metadata())


def test_metadata_does_not_reach_mgf(mzml_path, tmp_path):
    decompressor = SpectrumDecompressor.__new__(SpectrumDecompressor)
    decompressor.writer = get_writer(tmp_path / "out.mgf")
    spectra = get_reader(mzml_path).read_all()
    columns, definitions = encode_columns([s.mzml for s in spectra])
    decompressor.mzmlDefinitions = load_definitions(definitions)
    row = pd.Series(
        {
            "title": "a",
            "pepmz": 371.5,
            "pepit": 1000.0,
            **{name: array[0].as_py() for name, array in columns.items()},
        }
    )
    spectrum = decompressor.formatMGFformat([100.0], [1.0], row)
    decompressor.writer.write([spectrum])
    assert spectrum.mzml == {}
    assert sorted(spectrum.params) == ["pepmass", "title"]
    assert "FILTER" not in (tmp_path / "out.mgf").read_text().upper()


REAL = os.path.join(os.path.dirname(__file__), "..", "LFQ_Orbitrap_DDA_Ecoli_02.mzML")


@pytest.mark.skipif(not os.path.exists(REAL), reason="reference mzML not present")
def test_real_file_first_spectrum_matches_mgf_convention():
    import itertools

    s = next(iter(get_reader(REAL)))
    assert s.params["title"] == "LFQ_Orbitrap_DDA_Ecoli_02.3091.3091.2"
    assert s.params["charge"] == [2]
    assert len(s.mz) == len(s.intensity) == 30
    assert (
        list(itertools.islice(get_reader(REAL), 2))[1]
        .params["title"]
        .startswith("LFQ_Orbitrap_DDA_Ecoli_02.")
    )
