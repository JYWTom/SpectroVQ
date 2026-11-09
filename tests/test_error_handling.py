"""Corrupt or mismatched inputs raise clear errors instead of producing bad output."""

import io
import zlib
from pathlib import Path

import pytest
import torch

from tests.test_compress_roundtrip import N_Q, _spectra
from utils.Compression.compress import ZlibCompressedFile
from utils.formats import get_writer
from utils.mgfHandler.compressor import SpectrumCompressor
from utils.mgfHandler.decompressor import SpectrumDecompressor
from utils.model.getModel import DEFAULTPARAMS, getModel
from utils.model.modelStream import SpectroVQ


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return getModel().eval()


def _pack(tmp_path, model, **codec):
    src = tmp_path / "in.mgf"
    get_writer(str(src)).write(_spectra())
    stem = str(tmp_path / "packed")
    SpectrumCompressor(
        model, str(src), OutputFileName=stem, batch_size=2, quantizer=N_Q, **codec
    ).CompressAll(num_workers=0)
    return stem + ".vqms2"


def _unpack(path, tmp_path, model, quantizer=N_Q, **codec):
    SpectrumDecompressor(
        model,
        path,
        str(tmp_path / "out.mgf"),
        batch_size=2,
        quantizer=quantizer,
        **codec,
    ).DecompressAll()


def test_truncated_file_raises(tmp_path, model):
    path = _pack(tmp_path, model, zlib_compression_level=None)
    with open(path, "r+b") as f:
        f.truncate(f.seek(0, 2) // 2)
    with pytest.raises(ValueError, match="ended before all codes were read"):
        _unpack(path, tmp_path, model, zlib_compression_level=None)


def test_quantizer_mismatch_raises(tmp_path, model):
    path = _pack(tmp_path, model, zlib_compression_level=None)
    with pytest.raises(ValueError, match="--quantizer"):
        _unpack(path, tmp_path, model, quantizer=N_Q + 4, zlib_compression_level=None)


def test_corrupt_zlib_stream_raises(tmp_path, model):
    path = _pack(tmp_path, model, zlib_compression_level=6)
    data = bytearray(Path(path).read_bytes())
    for i in range(2, len(data) - 4):  # keep the 2-byte header, garble the body
        data[i] ^= 0xA5
    Path(path).write_bytes(bytes(data))
    with pytest.raises(zlib.error):
        _unpack(path, tmp_path, model, zlib_compression_level=6)


def test_zlib_file_roundtrip():
    buf = io.BytesIO()
    w = ZlibCompressedFile(buf, compression_level=6, mode="w")
    payload = bytes(range(256)) * 50
    w.write(payload)
    w.flush()
    r = ZlibCompressedFile(io.BytesIO(buf.getvalue()), mode="r")
    assert r.read(100) + r.read() == payload


def test_qinco_is_rejected():
    with pytest.raises(NotImplementedError, match="QINCo"):
        SpectroVQ(**{**DEFAULTPARAMS, "useQINCo": True})
