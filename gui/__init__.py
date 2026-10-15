"""
Browser GUI for SpectroVQ (Gradio).

Tabs (gui/tabs/):
    Explore    - reconstruct a single spectrum with 1..n_q quantizers and compare it to the input
    Denoise    - MGF -> denoised MGF (MGFDenoiser), with a before/after viewer
    Compress   - MGF -> .vqms2 + .parquet (+ _compounded.pkl) (MGFCompressor)
    Decompress - .vqms2 + side files -> MGF (MGFDecompressor)
    Train      - runs train.py as a subprocess and streams its log

Shared widgets (file picker, model panel, sessions sidebar) live in gui/components/; gui/app.py
assembles the page and gui.py at the project root is the entry point.

The GUI is a remote front end for the machine it runs on (e.g. a GPU server): every input is a
path on that machine, picked by typing it or with the "Browse ..." file browser, and outputs are
written to an export path on that machine (default: the run's gui_outputs/ folder). Nothing is
uploaded from or downloaded to the browser.

Run on the server, then reach it e.g. through an SSH tunnel (ssh -L 7860:localhost:7860 server):
    python gui.py [--host 127.0.0.1] [--port 7860] [--root /data] [--ckpt model.ckpt]
"""
