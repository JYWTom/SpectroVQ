# %%
import argparse
import os

import torch

from utils import config
from utils.config import ACCELERATORS, CODECS, CODEC_LEVEL_RANGES
from utils.device import resolve_device
from utils.formats import supported_input_extensions, supported_output_extensions
from utils import pipelines
from utils.model import loader
from utils.session_cli import (
    add_background_argument,
    add_compile_argument,
    add_list_argument,
    run_session,
)

# Set up command line argument parser for compression/decompression modes
argparser = argparse.ArgumentParser(
    description="Compress or Decompress mass spectrometry files"
)
mode = argparser.add_mutually_exclusive_group(required=True)
mode.add_argument("--compress", "-m", action="store_true", help="Mode: Compress")
mode.add_argument("--decompress", "-d", action="store_true", help="Mode: Decompress")
argparser.add_argument(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Input file path (.mgf or .mzML, or .vqms2 when decompressing)",
)
argparser.add_argument(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Compress: output path (.vqms2 added if missing; the .parquet side file "
    "is written next to it). Decompress: .mgf or .mzML "
    "(.mzML needs an archive compressed from mzML)",
)
argparser.add_argument(
    "--weights", "-w", type=str, required=True, help="Path to model weights"
)
argparser.add_argument(
    "--compression_method",
    "-cM",
    type=str,
    choices=CODECS,
    default=config.COMPRESSION_METHOD,
    help="Entropy coder applied to the packed codes; omit for none. "
    "Decompression must use the same method.",
)
argparser.add_argument(
    "--compression_level",
    "-cL",
    type=int,
    default=config.COMPRESSION_LEVEL,
    help="Level for --compression_method ("
    + ", ".join(f"{c} {lo}-{hi}" for c, (lo, hi) in CODEC_LEVEL_RANGES.items())
    + ")",
)
argparser.add_argument(
    "--batch_size",
    "-b",
    type=int,
    default=config.CODEC_BATCH_SIZE,
    help=f"Spectra per batch (default: {config.CODEC_BATCH_SIZE})",
)
argparser.add_argument(
    "--quantizer",
    "-q",
    type=int,
    default=config.QUANTIZERS,
    help="Number of quantizers; must match between compression and decompression "
    f"(default: {config.QUANTIZERS})",
)
argparser.add_argument(
    "--stored_raw",
    "-sR",
    action="store_true",
    default=config.STORED_RAW,
    help="Store raw reconstructed spectra",
)
argparser.add_argument(
    "--accelerator",
    type=str,
    default=config.DEFAULT_ACCELERATOR,
    choices=ACCELERATORS,
    help="Lightning accelerator: auto, cpu, cuda, mps",
)
add_compile_argument(argparser, "torch.compile the encoder and decoder (CUDA only)")
argparser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
add_background_argument(argparser)
add_list_argument(argparser)


def _session_params(arguments):
    """The settings recorded for the run's session, named as the GUI's tabs name theirs
    (run_session adds the compile mode)."""
    params = {
        "input": os.path.abspath(arguments.input),
        "output": os.path.abspath(arguments.output),
        "quantizers": arguments.quantizer,
    }
    if arguments.compression_method is not None:
        params.update(
            method=arguments.compression_method, level=arguments.compression_level
        )
    params["batch_size"] = arguments.batch_size
    if not arguments.compress:
        params["stored_raw"] = arguments.stored_raw
    return params


def _outputs(arguments):
    """The files a run has written, for its session record."""
    out = os.path.abspath(arguments.output)
    if arguments.compress:
        stem = out.removesuffix(".vqms2")
        out = [stem + ".vqms2", stem + ".parquet"]
    else:
        out = [out]
    return [p for p in out if os.path.exists(p)]


def main(argv=None):
    arguments = argparser.parse_args(argv)
    input_file_path = arguments.input
    output_file_path = arguments.output

    # Validate everything before the (slow) model load
    if (arguments.compression_method is None) != (arguments.compression_level is None):
        argparser.error(
            "--compression_method and --compression_level must be given together"
        )
    if arguments.quantizer < config.MIN_QUANTIZERS:
        argparser.error(f"--quantizer must be at least {config.MIN_QUANTIZERS}")
    if not os.path.exists(input_file_path):
        argparser.error(f"Input file not found: {input_file_path}")
    if not os.path.exists(arguments.weights):
        argparser.error(f"Weights file not found: {arguments.weights}")
    if arguments.compress:
        if (
            os.path.splitext(input_file_path)[1].lower()
            not in supported_input_extensions()
        ):
            argparser.error(
                f"Input file must be one of: {', '.join(supported_input_extensions())}"
            )
    else:
        if not input_file_path.endswith(".vqms2"):
            argparser.error("Input file must be a .vqms2")
        if (
            os.path.splitext(output_file_path)[1].lower()
            not in supported_output_extensions()
        ):
            argparser.error(
                f"Output file must be one of: {', '.join(supported_output_extensions())}"
            )

    # After validation, so a bad argument shows in the terminal, not only in the log.
    with run_session(
        arguments,
        __file__,
        argv,
        kind="compress" if arguments.compress else "decompress",
        params=_session_params(arguments),
        outputs=lambda: _outputs(arguments),
    ):
        _run(arguments)


def _run(arguments):
    device = resolve_device(arguments.accelerator)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # The trained model, with its weights, in eval mode on the selected device
    model = loader.load_model(
        arguments.weights, None, arguments.accelerator, arguments.compile, cache=False
    )

    verboseInt = 2 if arguments.verbose else 0
    if arguments.compress:
        if arguments.verbose:
            print("Compressing...")
        pipelines.compress(
            model,
            arguments.input,
            arguments.output,
            arguments.quantizer,
            arguments.compression_method,
            arguments.compression_level,
            arguments.batch_size,
            verbose=verboseInt,
        )
    else:
        if arguments.verbose:
            print("Decompressing...")
        pipelines.decompress(
            model,
            arguments.input,
            arguments.output,
            arguments.quantizer,
            arguments.compression_method,
            arguments.compression_level,
            arguments.batch_size,
            arguments.stored_raw,
            verbose=verboseInt,
        )


# DataLoader workers re-import this module on spawn platforms (macOS, Windows)
if __name__ == "__main__":
    main()
