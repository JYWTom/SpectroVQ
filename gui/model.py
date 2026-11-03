"""Model loading, caching and device selection."""

import os
import threading

import gradio as gr

from gui.config import read_config, save_config
from utils.compile import COMPILE_MODES, compile_model
from utils.device import resolve_device
from utils.model.getModel import ApplyWeights, getModel

_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()


def initial_ckpt():
    """Checkpoint shown when a page opens: --ckpt wins, else the last one that loaded."""
    return os.environ.get("SPECTROVQ_CKPT") or read_config().get("ckpt", "")


def initial_yaml():
    return read_config().get("yaml", "")


def initial_model_status():
    ckpt = initial_ckpt()
    if not ckpt:
        return "_No model loaded yet. It is loaded automatically on first use._"
    source = "from --ckpt" if os.environ.get("SPECTROVQ_CKPT") else "last used"
    return (
        f"_Checkpoint ({source}): `{ckpt}`. It is loaded automatically on first use._"
    )


def load_model(ckpt_path, yaml_path, device):
    ckpt_path = os.path.expanduser((ckpt_path or "").strip())
    yaml_path = os.path.expanduser((yaml_path or "").strip()) or None
    if not ckpt_path:
        raise gr.Error("Set the model checkpoint path in the Model panel first.")
    if not os.path.exists(ckpt_path):
        raise gr.Error(f"Checkpoint not found: {ckpt_path}")
    if yaml_path is not None and not os.path.exists(yaml_path):
        raise gr.Error(f"YAML config not found: {yaml_path}")
    device = resolve_device(device).type
    # Opt in with SPECTROVQ_COMPILE=default|max-autotune (CUDA only).
    compile_mode = os.environ.get("SPECTROVQ_COMPILE", "none")
    if compile_mode not in COMPILE_MODES:
        raise gr.Error(
            f"SPECTROVQ_COMPILE must be one of {', '.join(COMPILE_MODES)}, got {compile_mode!r}"
        )
    key = (
        os.path.abspath(ckpt_path),
        os.path.getmtime(ckpt_path),
        yaml_path,
        device,
        compile_mode,
    )
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()
            model = ApplyWeights(getModel(yaml_path), ckpt_path)
            model.to(device)
            model.eval()
            compile_model(model, compile_mode, device)
            _MODEL_CACHE[key] = model
            save_config(
                ckpt=os.path.abspath(ckpt_path),
                yaml=os.path.abspath(yaml_path) if yaml_path else "",
            )
        return _MODEL_CACHE[key]


def model_status(ckpt_path, yaml_path, device):
    try:
        model = load_model(ckpt_path, yaml_path, device)
    except gr.Error as e:
        return f"⚠️ {e.message}"
    except Exception as e:  # noqa: BLE001
        return f"⚠️ Failed to load model: {e}"
    n_q = model.n_q
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return (
        f"✅ Loaded on **{resolve_device(device).type}** · {n_params:.1f}M parameters · "
        f"{n_q} quantizers · {model.quantizedlen} codes per quantizer per spectrum"
    )
