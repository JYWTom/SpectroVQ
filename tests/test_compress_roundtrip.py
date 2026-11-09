"""Compress -> decompress round trip, and the encoder's result for each flag combination.

Uses the default (untrained) model on CPU: reconstructions are meaningless, but
everything the container stores verbatim (metadata, out-of-range peaks,
spectrum count and order) must come back unchanged.
"""

import numpy as np
import pytest
import torch

from utils.Compression.torchprocessor import EncodeResult, SpectrumTorchBatchEncoder
from utils.formats import Spectrum, get_reader, get_writer
from utils.mgfHandler.compressor import SpectrumCompressor
from utils.mgfHandler.decompressor import SpectrumDecompressor
from utils.model.getModel import getModel

N_Q = 4
N_SPECTRA = 5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return getModel().eval()


def _spectra():
    rng = np.random.default_rng(0)
    spectra = []
    for i in range(N_SPECTRA):
        in_range = np.sort(rng.uniform(150, 1500, 40))
        out_of_range = np.array(
            [100.0 + i, 1600.0 + i]
        )  # stored verbatim in the parquet
        mz = np.sort(np.concatenate([in_range, out_of_range]))
        spectra.append(
            Spectrum(
                mz,
                rng.uniform(1, 1000, mz.size),
                {
                    "title": f"run.{i}.{i}.2",
                    "rtinseconds": 10.0 * i,
                    "pepmass": (400.0 + i, 1000.0),
                    "charge": [2],
                },
            )
        )
    return spectra


def _out_of_range(s):
    keep = (s.mz < 150) | (s.mz >= 1500)
    return s.mz[keep], s.intensity[keep]


def test_compress_decompress_roundtrip(tmp_path, model):
    src = tmp_path / "in.mgf"
    get_writer(str(src)).write(_spectra())
    stem = str(tmp_path / "packed")

    SpectrumCompressor(
        model, str(src), OutputFileName=stem, batch_size=2, quantizer=N_Q
    ).CompressAll(num_workers=0)
    out = tmp_path / "out.mgf"
    SpectrumDecompressor(
        model, stem + ".vqms2", str(out), batch_size=2, quantizer=N_Q
    ).DecompressAll()

    back = get_reader(str(out)).read_all()
    assert [s.params["title"] for s in back] == [s.params["title"] for s in _spectra()]
    for a, b in zip(_spectra(), back):
        assert b.params["pepmass"][0] == pytest.approx(a.params["pepmass"][0])
        mz_a, it_a = _out_of_range(a)
        mz_b, it_b = _out_of_range(b)
        np.testing.assert_allclose(mz_b, mz_a, atol=1e-3)
        np.testing.assert_allclose(it_b, it_a, rtol=1e-6)


@pytest.mark.parametrize("compounded", [False, True])
@pytest.mark.parametrize("out_of_range", [False, True])
def test_encoder_result_fields(model, compounded, out_of_range):
    spectra = _spectra()
    enc = SpectrumTorchBatchEncoder(
        [s.mz for s in spectra],
        [s.intensity for s in spectra],
        model,
        compounded=compounded,
    )
    res = enc.getReconstructIndices(
        outputOutOfRange=out_of_range, quantizer=N_Q, batch_size=2, num_workers=0
    )
    assert isinstance(res, EncodeResult)
    assert res.codes.shape[:2] == (N_Q, N_SPECTRA)  # N B D
    assert len(res.max_vals) == N_SPECTRA
    assert len(res.compounded_codes) == len(res.compounded_idx)
    if not compounded:
        assert res.compounded_codes == []
    assert all(0 <= i < N_SPECTRA for i in res.compounded_idx)
    if out_of_range:
        assert len(res.extended_mz) == len(res.extended_it) == N_SPECTRA
        for s, mz in zip(spectra, res.extended_mz):
            np.testing.assert_array_equal(mz, _out_of_range(s)[0])
    else:
        assert res.extended_mz == res.extended_it == []
