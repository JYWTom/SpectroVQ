"""
Browser GUI for SpectroVQ (Gradio). The code lives in the gui/ package; see gui/__init__.py.

Run on the server, then reach it e.g. through an SSH tunnel (ssh -L PORT:localhost:PORT server).
Without --port Gradio uses $GRADIO_SERVER_PORT or 7860, moving to 7861, 7862, ... if that is
taken. Like Jupyter, each launch makes a random token and prints the URL to open
(http://127.0.0.1:PORT/?token=...); `python gui.py --list` shows it again:
    python gui.py [--host 127.0.0.1] [--port 7860] [--root /data] [--ckpt model.ckpt]
                  [--token TOKEN | --no-token] [--data-dir ~/.spectrovq]
"""

import argparse
import json
import os
import signal
import sys

from starlette.middleware import Middleware

from gui import config
from gui.auth import TokenAuth, make_token


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # alive, but someone else's
    return True


def list_servers():
    """Print this user's running GUI servers and their token URLs, like `jupyter server list`."""
    servers = []
    for f in sorted(config.RUNTIME_DIR.glob("gui-*.json")):
        try:
            info = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if process_alive(info.get("pid", -1)):
            servers.append(info)
        else:
            f.unlink(missing_ok=True)  # left behind by a server that was killed
    if not servers:
        print("No SpectroVQ GUI servers running.")
    for info in servers:
        print(f"{info['url']}  (pid {info['pid']})")


def write_runtime_file(port, url):
    path = config.RUNTIME_DIR / f"gui-{port}.json"
    # Created owner-only before the token is written, so other users can never read it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "pid": os.getpid(), "port": port}, f)
    return path


def main():
    parser = argparse.ArgumentParser(description="SpectroVQ browser GUI")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind; keep 127.0.0.1 and use an SSH tunnel, or 0.0.0.0 on a trusted network",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve on (default: Gradio picks $GRADIO_SERVER_PORT or 7860, then the next free port)",
    )
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
        default=str(config.BROWSE_ROOT),
        help="Starting folder of the server file browsers (default: your home folder)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SPECTROVQ_DATA_DIR", str(config.DATA_DIR)),
        help="Where your sessions (gui_outputs/) and settings are kept (default: ~/.spectrovq)",
    )
    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument(
        "--token",
        default=os.environ.get("SPECTROVQ_TOKEN"),
        help="Token needed to open the GUI (default: $SPECTROVQ_TOKEN, else a new random one)",
    )
    token_group.add_argument(
        "--no-token",
        action="store_true",
        help="Let anyone who can reach the port use the GUI (unsafe on a shared server)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List your running GUI servers and exit"
    )
    args = parser.parse_args()

    config.set_data_dir(args.data_dir)
    if args.list:
        return list_servers()
    if args.ckpt:
        os.environ["SPECTROVQ_CKPT"] = args.ckpt
    config.BROWSE_ROOT = os.path.abspath(os.path.expanduser(args.root))

    token = None if args.no_token else (args.token or make_token())
    middleware = [Middleware(TokenAuth, token=token)] if token else []

    from gui.app import build_ui  # imports torch; keep --list fast

    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True,
        app_kwargs={"middleware": middleware},
        # Long paths in the sessions sidebar have few break points; let them wrap anywhere.
        css=".sidebar code { overflow-wrap: anywhere; }",
    )

    port = demo.server_port
    url = f"http://127.0.0.1:{port}/" + (f"?token={token}" if token else "")
    runtime_file = write_runtime_file(port, url)
    print()
    if token:
        print("To open the GUI, forward the port and open this URL (it includes your token):")
        print(f"    ssh -L {port}:localhost:{port} <server>")
        print(f"    {url}")
        if demo.share_url:
            print(f"Share link (also needs the token): {demo.share_url}?token={token}")
        print("Run `python gui.py --list` on the server to see this again.")
    else:
        print("WARNING: --no-token: anyone who can reach this port can use the GUI.")
    print(flush=True)  # show the URL now even when stdout goes to a log file (nohup ... &)
    # `kill` should also remove the runtime file, not just Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        demo.block_thread()
    finally:
        runtime_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
