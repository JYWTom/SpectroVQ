"""Assembles the page: header, sessions sidebar, model panel and tabs."""

import gradio as gr

from gui.components import model_panel, sessions_sidebar
from gui.tabs import compress, decompress, denoise, explore, train


def build_ui():
    with gr.Blocks(title="SpectroVQ") as demo:
        gr.Markdown(
            "# SpectroVQ\n"
            "Compress and denoise MS/MS spectra with a residual vector-quantized autoencoder. "
            "Fewer quantizers → more compression and more denoising; more quantizers → higher fidelity. "
            "Pretrained weights: [Zenodo](https://zenodo.org/records/20043624)."
        )

        sidebar = sessions_sidebar.build()
        model_inputs = model_panel.build()
        ckpt = model_inputs[0]

        with gr.Tabs() as tabs:
            ex = explore.build(model_inputs)
            dn = denoise.build(model_inputs, sidebar.outputs)
            cp = compress.build(model_inputs, sidebar.outputs)
            dc = decompress.build(model_inputs, sidebar.outputs)
            tr = train.build(sidebar.outputs)

        # Session kind -> how its tab shows it; each kind's tab id is the kind itself.
        views = {
            "denoise": dn.session,
            "compress": cp.session,
            "decompress": dc.session,
            "train": tr.session,
        }
        sessions_sidebar.wire(demo, sidebar, tabs, views, ex, dc, ckpt)
    return demo
