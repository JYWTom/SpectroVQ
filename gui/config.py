"""Paths, binning constants and the server-side settings file."""

import json
import os
from pathlib import Path

# The SpectroVQ project folder.
ROOT = Path(__file__).resolve().parent.parent

# Per-user state, so several people can run the GUI from one shared install; set with --data-dir.
DATA_DIR = Path.home() / ".spectrovq"
OUTPUT_ROOT = DATA_DIR / "gui_outputs"
# Server-side GUI settings that outlive restarts: the last model that loaded successfully.
CONFIG_PATH = DATA_DIR / "gui_config.json"
# Running servers' URLs and tokens, for `gui.py --list`.
RUNTIME_DIR = DATA_DIR / "runtime"
# Starting folder of the "Browse ..." file browsers; set with --root. '..' and typed paths go anywhere.
BROWSE_ROOT = str(Path.home())

# Binning parameters; must match train.py, the compressor/decompressor and the denoiser.
BIN_SIZE = 0.1
SPECTRA_DIMENSION = 13500
MZ_RANGE = [150, 1500]
CODE_BITS = 10


def set_data_dir(path):
    """Point all per-user state at path, creating it readable by its owner only."""
    global DATA_DIR, OUTPUT_ROOT, CONFIG_PATH, RUNTIME_DIR
    DATA_DIR = Path(path).expanduser().resolve()
    OUTPUT_ROOT = DATA_DIR / "gui_outputs"
    CONFIG_PATH = DATA_DIR / "gui_config.json"
    RUNTIME_DIR = DATA_DIR / "runtime"
    for d in (DATA_DIR, OUTPUT_ROOT, RUNTIME_DIR):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)


def read_config():
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_config(**updates):
    config = read_config()
    if all(config.get(k) == v for k, v in updates.items()):
        return
    config.update(updates)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    os.replace(tmp, CONFIG_PATH)  # atomic, so a crash never leaves half a file
