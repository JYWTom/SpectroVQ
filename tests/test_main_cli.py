"""main.py: arguments reach the compressor, and bad arguments fail before the model loads."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from utils.formats import Spectrum, get_reader, get_writer
from utils.model.getModel import getModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _main(*args):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    torch.manual_seed(0)
    # Lightning layout: ApplyWeights strips the leading "model." component
    state = {"model." + k: v for k, v in getModel().state_dict().items()}
    torch.save({"state_dict": state}, d / "w.ckpt")
    rng = np.random.default_rng(0)
    spectra = [
        Spectrum(
            np.sort(rng.uniform(150, 1500, 30)),
            rng.uniform(1, 1000, 30),
            {"title": f"r.{i}.{i}.2", "pepmass": (500.0 + i, 1.0), "charge": [2]},
        )
        for i in range(3)
    ]
    get_writer(str(d / "in.mgf")).write(spectra)
    return d


def _base(d):
    return ["-w", str(d / "w.ckpt"), "--accelerator", "cpu"]


def test_quantizer_and_codec_are_used(files):
    d = files
    stem = str(d / "packed")
    common = [*_base(d), "-q", "3", "-cM", "zstd", "-cL", "3"]
    r = _main("-m", "-i", str(d / "in.mgf"), "-o", stem, *common)
    assert r.returncode == 0, r.stderr
    # 10-bit codes, zstd: count header holds n_spectra * n_q * quantizedlen
    with open(stem + ".vqms2", "rb") as f:
        count = int.from_bytes(f.read(4), "little")
    quantizedlen = getModel().quantizedlen
    assert count % (3 * quantizedlen) == 0 and count >= 3 * 3 * quantizedlen

    r = _main("-d", "-i", stem + ".vqms2", "-o", str(d / "out.mgf"), *common)
    assert r.returncode == 0, r.stderr
    assert len(get_reader(str(d / "out.mgf")).read_all()) == 3


@pytest.mark.parametrize(
    "extra, message",
    [
        (["-cM", "zstd"], "must be given together"),
        (["-cL", "3"], "must be given together"),
        (["-cM", "lzma", "-cL", "3"], "invalid choice"),
        (["-q", "0"], "at least 1"),
    ],
)
def test_bad_arguments_fail_fast(files, extra, message):
    d = files
    r = _main("-m", "-i", str(d / "in.mgf"), "-o", str(d / "x"), *_base(d), *extra)
    assert r.returncode == 2
    assert message in r.stderr


def test_mode_is_required(files):
    d = files
    r = _main("-i", str(d / "in.mgf"), "-o", str(d / "x"), *_base(d))
    assert r.returncode == 2
    assert "one of the arguments --compress/-m --decompress/-d is required" in r.stderr


def test_output_with_vqms2_extension_is_not_doubled(files):
    """The README passes --output output.vqms2."""
    d = files
    r = _main("-m", "-i", str(d / "in.mgf"), "-o", str(d / "readme.vqms2"), *_base(d))
    assert r.returncode == 0, r.stderr
    assert (d / "readme.vqms2").exists() and (d / "readme.parquet").exists()
    assert not (d / "readme.vqms2.vqms2").exists()
