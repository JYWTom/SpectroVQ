"""
Browser GUI for SpectroVQ (Gradio).

Tabs (gui/tabs/):
    Explore    - reconstruct a single spectrum with 1..n_q quantizers and compare it to the input
    Denoise    - MGF -> denoised MGF, with a before/after viewer
    Compress   - MGF or mzML -> .vqms2 + .parquet
    Decompress - .vqms2 + side files -> MGF or mzML
    Evaluate   - SpectroVQ vs a TopN-peaks baseline: per-spectrum compression ratio vs cosine
    Train      - runs train.py --background and follows its session

Shared widgets (file picker, model panel, sessions sidebar) live in gui/components/; gui/app.py
assembles the page and gui.py at the project root is the entry point.

This package is Gradio only: layout, event wiring and formatting results as text. Everything a
tab does lives in utils/ (pipelines, explore, evaluate, training, jobs, sessions, spectra,
plots, paths, settings, model.loader), where main.py, train.py and denoise.py use it too. The
library raises utils.errors.UserError for a bad request; gui/errors.py shows it as a gr.Error.

The GUI is a remote front end for the machine it runs on (e.g. a GPU server): every input is a
path on that machine, picked by typing it or with the "Browse ..." file browser, and outputs are
written to an export path on that machine (default: the run's folder in ~/.spectrovq/gui_outputs/). Nothing is
uploaded from or downloaded to the browser.

Several people can run their own GUI from one shared install (e.g. a lab GPU server): each
user's sessions and settings live in ~/.spectrovq (--data-dir), and every server needs a token,
like Jupyter (gui/auth.py), because anyone on the machine can reach its 127.0.0.1 ports.

Run on the server, then reach it e.g. through an SSH tunnel (ssh -L PORT:localhost:PORT server).
Without --port Gradio uses $GRADIO_SERVER_PORT or 7860, moving to 7861, 7862, ... if that is
taken. The URL to open, with its token, is printed at startup; `python gui.py --list` shows it again:
    python gui.py [--host 127.0.0.1] [--port 7860] [--root /data] [--ckpt model.ckpt]
                  [--token TOKEN | --no-token] [--data-dir ~/.spectrovq]
"""
