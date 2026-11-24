"""Decoding settings recorded in the .parquet side file are checked on decompress."""

import pyarrow.parquet as pq
import pytest
import torch

from tests.test_compress_roundtrip import CODECS, N_Q, _spectra
from utils.Compression import header
from utils.formats import get_reader, get_writer
from utils.mgfHandler.compressor import SpectrumCompressor
from utils.mgfHandler.decompressor import SpectrumDecompressor
from utils.model.getModel import getModel


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return getModel().eval()


@pytest.fixture
def packed(tmp_path, model):
    src = tmp_path / "in.mgf"
    get_writer(str(src)).write(_spectra())
    stem = str(tmp_path / "packed")
    SpectrumCompressor(
        model,
        str(src),
        OutputFileName=stem,
        batch_size=2,
        quantizer=N_Q,
        **CODECS["zstd"],
    ).CompressAll(num_workers=0)
    return stem


def _decompressor(stem, tmp_path, model, quantizer=N_Q, codec="zstd"):
    return SpectrumDecompressor(
        model,
        stem + ".vqms2",
        str(tmp_path / "out.mgf"),
        batch_size=2,
        quantizer=quantizer,
        **CODECS[codec],
    )


def test_settings_are_recorded(packed, model):
    assert header.read(packed + ".parquet") == {
        "format": header.FORMAT_VERSION,
        "quantizer": N_Q,
        "codec": "zstd",
        "quantizedlen": model.quantizedlen,
        "bits": 10,
    }


def test_smaller_quantizer_is_rejected(packed, tmp_path, model):
    """Without the header this decoded misaligned codes with no error."""
    with pytest.raises(ValueError, match=f"--quantizer {N_Q}"):
        _decompressor(packed, tmp_path, model, quantizer=N_Q - 2)


def test_wrong_codec_is_rejected(packed, tmp_path, model):
    with pytest.raises(ValueError, match="--compression_method zstd"):
        _decompressor(packed, tmp_path, model, codec="zlib")


def test_files_without_header_still_decode(packed, tmp_path, model):
    path = packed + ".parquet"
    table = pq.read_table(path)
    meta = {k: v for k, v in table.schema.metadata.items() if k != header.KEY}
    pq.write_table(table.replace_schema_metadata(meta), path)
    assert header.read(path) is None
    _decompressor(packed, tmp_path, model).DecompressAll()
    assert len(get_reader(str(tmp_path / "out.mgf")).read_all()) == len(_spectra())


def test_codebook_too_large_for_10_bits_is_rejected(tmp_path, model, monkeypatch):
    src = tmp_path / "in.mgf"
    get_writer(str(src)).write(_spectra())
    monkeypatch.setattr(model, "codebook_size", 2048)
    with pytest.raises(ValueError, match="codebook_size 2048"):
        SpectrumCompressor(model, str(src), OutputFileName=str(tmp_path / "p"))
