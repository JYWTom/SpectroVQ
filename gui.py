"""
Browser GUI for SpectroVQ (Gradio). The code lives in the gui/ package; see gui/__init__.py.

Run on the server, then reach it e.g. through an SSH tunnel (ssh -L 7860:localhost:7860 server):
    python gui.py [--host 127.0.0.1] [--port 7860] [--root /data] [--ckpt model.ckpt]
"""

import argparse
import os

from gui import config
from gui.app import build_ui


def main():
    parser = argparse.ArgumentParser(description="SpectroVQ browser GUI")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind; keep 127.0.0.1 and use an SSH tunnel, or 0.0.0.0 on a trusted network",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Pre-fill the checkpoint path (default: the last model that loaded, saved in gui_config.json)",
    )
    parser.add_argument(
        "--root",
        default=str(config.ROOT),
        help="Starting folder of the server file browsers (default: the SpectroVQ project folder)",
    )
    args = parser.parse_args()
    if args.ckpt:
        os.environ["SPECTROVQ_CKPT"] = args.ckpt
    config.BROWSE_ROOT = os.path.abspath(os.path.expanduser(args.root))
    config.OUTPUT_ROOT.mkdir(exist_ok=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        # Long paths in the sessions sidebar have few break points; let them wrap anywhere.
        css=".sidebar code { overflow-wrap: anywhere; }",
    )


if __name__ == "__main__":
    main()
