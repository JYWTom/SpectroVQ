"""The "Model" panel: checkpoint, optional YAML config, device and torch.compile mode, shared
by every tab. Each run uses the panel's values when it starts, and its session records them."""

import gradio as gr

from gui.components.file_picker import host_file_picker
from gui.model import (
    initial_ckpt,
    initial_model_status,
    initial_yaml,
    model_status,
)
from utils.device import available_accelerators
from utils.config import COMPILE_MODES, DEFAULT_ACCELERATOR, DEFAULT_COMPILE


def build():
    """Returns the (checkpoint, yaml, device, compile) inputs the model-using tabs pass to
    load_model."""
    with gr.Accordion("Model", open=True):
        with gr.Row():
            with gr.Column(scale=4):
                ckpt, _ = host_file_picker(
                    "Checkpoint (.ckpt) path on the server",
                    "*.ckpt",
                    "/path/to/SpectroVQ.ckpt",
                    value=initial_ckpt,  # re-read on every page load
                )
            with gr.Column(scale=2):
                yaml_cfg, _ = host_file_picker(
                    "Model YAML (optional)",
                    "*.yaml",
                    "modelParams.yaml",
                    value=initial_yaml,
                )
            with gr.Column(scale=1):
                device = gr.Dropdown(
                    available_accelerators(), value=DEFAULT_ACCELERATOR, label="Device"
                )
                compile_mode = gr.Dropdown(
                    list(COMPILE_MODES),
                    value=DEFAULT_COMPILE,
                    label="torch.compile",
                    info="CUDA only; per run",
                )
                load_btn = gr.Button("Load model")
        status = gr.Markdown(initial_model_status)  # re-run on every page load
        load_btn.click(model_status, [ckpt, yaml_cfg, device, compile_mode], status)
    return ckpt, yaml_cfg, device, compile_mode
