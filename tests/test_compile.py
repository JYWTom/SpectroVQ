"""torch.compile on real kernels. Needs CUDA: Inductor generates different code on other devices.

Errors are not suppressed here, because compile_model's fallback to eager would hide a failure.
"""

import pytest
import torch
from torch import _dynamo

from utils.model.getModel import getModel
from utils.model.modules.conv import SConvTranspose1d

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def loud_compile():
    previous = _dynamo.config.suppress_errors
    _dynamo.config.suppress_errors = False
    yield
    _dynamo.config.suppress_errors = previous


def _model():
    torch.manual_seed(0)
    return getModel().to("cuda").train()


def _spectra(batch=32):
    return torch.rand(batch, 1, 13500, device="cuda") ** 4


@cuda
def test_compiled_model_trains(loud_compile):
    """Without the copy in SConvTranspose1d this faults: CUDA error: misaligned address."""
    model = _model()
    model.encoder.compile(dynamic=False)
    model.decoder.compile(dynamic=False)
    x = _spectra()
    for _ in range(2):
        out, quantize_loss, _ = model(x, use_vq=True)
        (out.mean() + quantize_loss.mean()).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


@cuda
def test_compiled_decoder_gradients_match_eager(loud_compile):
    z = torch.rand(32, 256, 27, device="cuda")

    def gradients(model):
        model.decoder(z).mean().backward()
        return {
            name: p.grad.clone()
            for name, p in model.decoder.named_parameters()
            if p.grad is not None
        }

    eager = gradients(_model())
    compiled_model = _model()
    compiled_model.decoder.compile(dynamic=False)
    compiled = gradients(compiled_model)
    assert eager.keys() == compiled.keys()
    for name, grad in eager.items():
        scale = grad.abs().max().clamp(min=1e-12)
        difference = (grad - compiled[name]).abs().max() / scale
        assert difference < 1e-2, f"{name}: relative difference {difference}"


@cuda
def test_compiled_state_dict_keys_unchanged(loud_compile):
    eager = _model().state_dict().keys()
    model = _model()
    model.encoder.compile(dynamic=False)
    model.decoder.compile(dynamic=False)
    assert model.state_dict().keys() == eager


def test_eager_trim_stays_a_view():
    """The copy is only for compiled training: eager runs don't pay for it."""
    layer = SConvTranspose1d(4, 2, 4, stride=2)
    out = layer(torch.rand(1, 4, 10))
    assert out.requires_grad
    assert out._base is not None
