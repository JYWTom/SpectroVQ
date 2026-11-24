"""Loading a trained model for inference: weights, device and torch.compile, cached.

The GUI, main.py and denoise.py all load their model here.
"""

import os
import threading

from utils.compile import COMPILE_MODES, compile_model
from utils.device import resolve_device
from utils.errors import UserError
from utils.model.getModel import ApplyWeights, getModel

_CACHE = {}
_LOCK = threading.Lock()


def load_model(
    ckpt_path, yaml_path=None, device="auto", compile_mode="none", cache=True
):
    """The model in eval mode on device, compiled with compile_mode (CUDA only).

    With cache, the last model loaded is kept and returned again until the checkpoint (or its
    modification time), YAML, device or compile mode changes.
    """
    ckpt_path = os.path.expanduser((ckpt_path or "").strip())
    yaml_path = os.path.expanduser((yaml_path or "").strip()) or None
    if not ckpt_path:
        raise UserError("Set the model checkpoint path first.")
    if not os.path.exists(ckpt_path):
        raise UserError(f"Checkpoint not found: {ckpt_path}")
    if yaml_path is not None and not os.path.exists(yaml_path):
        raise UserError(f"YAML config not found: {yaml_path}")
    if compile_mode not in COMPILE_MODES:
        raise UserError(f"torch.compile must be one of {', '.join(COMPILE_MODES)}")
    device = resolve_device(device).type
    if not cache:
        return _load(ckpt_path, yaml_path, device, compile_mode)
    key = (
        os.path.abspath(ckpt_path),
        os.path.getmtime(ckpt_path),
        yaml_path,
        device,
        compile_mode,
    )
    with _LOCK:
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = _load(ckpt_path, yaml_path, device, compile_mode)
        return _CACHE[key]


def _load(ckpt_path, yaml_path, device, compile_mode):
    model = ApplyWeights(getModel(yaml_path), ckpt_path)
    model.to(device)
    model.eval()
    compile_model(model, compile_mode, device)
    return model


def describe(model):
    """What a loaded model is: {device, parameters (millions), quantizers, codes, compiled}."""
    return {
        "device": next(model.parameters()).device.type,
        "parameters": sum(p.numel() for p in model.parameters()) / 1e6,
        "quantizers": int(model.n_q),
        "codes": int(model.quantizedlen),  # per quantizer per spectrum
        "compiled": model.encoder._compiled_call_impl is not None,
    }
