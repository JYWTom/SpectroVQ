"""This GUI server's state: where its data dir is and what its command line set.

The data dir's layout and settings file are utils.settings; every other setting is utils.config.
"""

from pathlib import Path

from utils.sessions import default_data_dir
from utils.settings import data_dir

_DIR = data_dir(default_data_dir())
DATA_DIR = (
    _DIR.root
)  # per-user state, so several people can share one install; --data-dir
OUTPUT_ROOT = _DIR.sessions
CONFIG_PATH = _DIR.settings
RUNTIME_DIR = _DIR.runtime
# Starting folder of the "Browse ..." file browsers; set with --root. '..' and typed paths go anywhere.
BROWSE_ROOT = str(Path.home())
# The checkpoint given with --ckpt, shown when a page opens; "" shows the last one that loaded.
CKPT = ""


def set_data_dir(path):
    """Point all per-user state at path, creating it readable by its owner only."""
    global DATA_DIR, OUTPUT_ROOT, CONFIG_PATH, RUNTIME_DIR
    DATA_DIR, OUTPUT_ROOT, CONFIG_PATH, RUNTIME_DIR = data_dir(path, create=True)
