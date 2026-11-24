"""MGFDenoiser end to end with the default (untrained) model on CPU."""

import numpy as np
import pytest
import torch

from utils.formats import Spectrum, get_reader, get_writer
from utils.mgfHandler.denoiser import MGFDenoiser
from utils.model.getModel import getModel

N_SPECTRA = 6


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return getModel().eval()


@pytest.fixture
def src(tmp_path):
    rng = np.random.default_rng(0)
    spectra = []
    for i in range(N_SPECTRA):
        mz = np.sort(rng.uniform(150, 1500, 60))
        params = {"title": f"r.{i}.{i}.2", "pepmass": (500.0 + i, 1.0), "charge": [2]}
        if i % 2:  # some spectra carry no retention time
            params["rtinseconds"] = 10.0 * i
        spectra.append(Spectrum(mz, rng.uniform(1, 1000, mz.size), params))
    path = tmp_path / "in.MGF"  # upper-case extension
    get_writer(str(path)).write(spectra)
    return path


def _denoise(model, src, out, compounded):
    MGFDenoiser(model=model, inputMGF=str(src), outputMGF=str(out)).denoiseMGF(
        numQuantSingle=2, compoundedSpectra=compounded
    )
    return get_reader(str(out)).read_all()


@pytest.fixture
def sparse_model(model, monkeypatch):
    """The untrained model never flags a spectrum as compounded; a sparse
    reconstruction that misses most input peaks flags every one."""

    def sparse(x, n_q):
        out = torch.zeros_like(x)
        out[..., [2000, 4000, 6000]] = 1.0
        return out, None, None

    monkeypatch.setattr(model, "forwardWithnumQuantizers", sparse)
    return model


def test_compounded_off_writes_one_spectrum_each(tmp_path, sparse_model, src):
    back = _denoise(sparse_model, src, tmp_path / "out.mgf", compounded=False)
    assert [s.params["title"] for s in back] == [
        f"r.{i}.{i}.2" for i in range(N_SPECTRA)
    ]


def test_compounded_on_adds_one_raw_copy_each(tmp_path, sparse_model, src):
    back = _denoise(sparse_model, src, tmp_path / "out.mgf", compounded=True)
    titles = [s.params["title"] for s in back]
    assert titles[:N_SPECTRA] == [f"r.{i}.{i}.2" for i in range(N_SPECTRA)]
    raw = back[N_SPECTRA:]
    assert len(raw) == N_SPECTRA  # includes spectra without a retention time
    assert all("_raw_" in s.params["title"] for s in raw)


def test_missing_input_raises(tmp_path, model):
    with pytest.raises(FileNotFoundError):
        MGFDenoiser(model=model, inputMGF=str(tmp_path / "nope.mgf"), outputMGF="x.mgf")
