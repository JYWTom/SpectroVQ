import base64
import os
import struct

import numpy as np
import pytest

from utils.formats import get_reader, get_writer, supported_input_extensions
from utils.formats.mzml import MzMLReader


def _b64(values):
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode()


def _spectrum(index, scan, level, mz, it, precursor=""):
    return f"""
    <spectrum index="{index}" id="controllerType=0 controllerNumber=1 scan={scan}" defaultArrayLength="{len(mz)}">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}"/>
      <scanList count="1">
        <cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>
        <scan>
          <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="2.0" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
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


PRECURSOR = """<precursorList count="1"><precursor>
        <selectedIonList count="1"><selectedIon>
          <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="371.5"/>
          <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>
          <cvParam cvRef="MS" accession="MS:1000042" name="peak intensity" value="1000.0"/>
        </selectedIon></selectedIonList></precursor></precursorList>"""

MZML = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <cvList count="1"><cv id="MS" fullName="MS" version="4.1.0" URI="x"/></cvList>
  <run id="r">
    <spectrumList count="2">
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


def test_no_writer():
    with pytest.raises(ValueError, match="Unsupported output"):
        get_writer("out.mzML")


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
