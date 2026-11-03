import numpy as np
import pytest

from utils.formats import Spectrum, get_reader, get_writer


def _spectra():
    return [
        Spectrum(
            np.array([100.5, 200.25]),
            np.array([10.0, 20.0]),
            {
                "title": "a.1.1.2",
                "rtinseconds": 1.5,
                "pepmass": (371.3, 1000.0),
                "charge": [2],
            },
        ),
        Spectrum(
            np.array([150.0]),
            np.array([5.0]),
            {
                "title": "a.2.2.3",
                "rtinseconds": 2.5,
                "pepmass": (454.5, None),
                "charge": [3],
            },
        ),
    ]


def test_mgf_roundtrip(tmp_path):
    path = str(tmp_path / "t.MGF")  # case-insensitive extension
    get_writer(path).write(_spectra())
    back = get_reader(path).read_all()
    assert len(back) == 2
    for a, b in zip(_spectra(), back):
        np.testing.assert_allclose(a.mz, b.mz)
        np.testing.assert_allclose(a.intensity, b.intensity)
        assert a.params["title"] == b.params["title"]
        assert a.params["pepmass"][0] == pytest.approx(b.params["pepmass"][0])


def test_read_all_limit(tmp_path):
    path = str(tmp_path / "t.mgf")
    get_writer(path).write(_spectra())
    assert len(get_reader(path).read_all(limit=1)) == 1


def test_unknown_extension():
    with pytest.raises(ValueError, match="Supported"):
        get_reader("x.unknown")
    with pytest.raises(ValueError, match="Supported"):
        get_writer("x.unknown")
