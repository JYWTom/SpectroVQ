"""Settings shared by the command line (main.py, train.py, denoise.py) and the GUI.

Every default, choice and fixed constant lives here once, so the argument parsers and the GUI's
widgets cannot drift apart. Plain Python with no imports, so the argument parsers and --list
stay fast and need no torch.
"""

# --- Binning: fixed by the trained model; the trainer, codec, denoiser and GUI all use these.
BIN_SIZE = 0.1
SPECTRA_DIMENSION = 13500
MZ_RANGE = [150, 1500]
CODE_BITS = 10  # BitPacker and the zstd packer both store 10-bit codes

# --- Hardware
ACCELERATORS = ("auto", "cpu", "cuda", "mps")
DEFAULT_ACCELERATOR = "auto"
# torch.compile modes, applied by utils.compile and recorded in each session (params["compile"]).
COMPILE_MODES = ("none", "default", "max-autotune")
DEFAULT_COMPILE = "none"

# --- Compress / decompress (main.py, the Compress and Decompress tabs)
CODECS = ("gzip", "zlib", "zstd")  # entropy coders for the packed codes
NO_CODEC = (
    "none"  # the GUI's name for no entropy coder; on the command line, omit the method
)
CODEC_LEVEL_RANGES = {"gzip": (1, 9), "zlib": (1, 9), "zstd": (1, 22)}
COMPRESSION_METHOD = None  # no entropy coder
COMPRESSION_LEVEL = None  # required together with a method
QUANTIZERS = 4  # must match between compression and decompression
MIN_QUANTIZERS = 1
MAX_QUANTIZERS = 12  # the GUI sliders' upper end
CODEC_BATCH_SIZE = 128
GUI_CODEC = (
    "gzip"  # the coder and level the GUI's Compress and Decompress tabs pre-select
)
GUI_CODEC_LEVEL = 6
STORED_RAW = False  # decompression: also write raw (non-chimeric) reconstructions


def codec_levels(method, level):
    """The *_compression_level keyword arguments of the compressor and decompressor: only the
    chosen coder gets the level, the others stay None (disabled). method None or NO_CODEC means
    no entropy coder."""
    level = int(level) if level is not None else None
    return {
        f"{name}_compression_level": (level if name == method else None)
        for name in CODECS
    }


def codec_description(method, level):
    """How a run's entropy coder is named in logs and summaries."""
    if method in (None, NO_CODEC):
        return "no entropy coder"
    return f"{method} level {int(level)}"


# --- Denoise (denoise.py, the Denoise tab)
DENOISE_QUANTIZERS = 3  # fewer = stronger denoising
RETAIN_ORIGINAL_PEAKS = True  # keep original peaks outside MZ_RANGE
ONE_PERCENT_THRESHOLD = True  # drop peaks below 1% of the base peak
COMPOUNDED_SPECTRA = True  # reconstruct chimeric spectra
DENOISE_NUM_WORKERS = 0

# --- Train (train.py, the Train tab)
TRAIN_DATA_PATH = "library.feather"
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50
NUM_WORKERS = 4
CHECKPOINT_MINUTES = 30

# --- Explore and Evaluate (GUI only)
EXPLORE_QUANTIZERS = 3
EVAL_QUANTIZERS = (1, 2, 4, 8)
EVAL_TOP_N = (10, 20, 50, 100)
EVAL_MAX_SPECTRA = 5000  # 0 = whole file
EVAL_BATCH_SIZE = 64
