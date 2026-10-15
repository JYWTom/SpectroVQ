"""Paths, binning constants and the server-side settings file."""

import json
import os
from pathlib import Path

# The SpectroVQ project folder.
ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = ROOT / "gui_outputs"
# Starting folder of the "Browse ..." file browsers; set with --root. '..' and typed paths go anywhere.
BROWSE_ROOT = str(ROOT)

# Binning parameters; must match train.py, the compressor/decompressor and the denoiser.
BIN_SIZE = 0.1
SPECTRA_DIMENSION = 13500
MZ_RANGE = [150, 1500]
CODE_BITS = 10

# Server-side GUI settings that outlive restarts: the last model that loaded successfully.
CONFIG_PATH = ROOT / "gui_config.json"


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
