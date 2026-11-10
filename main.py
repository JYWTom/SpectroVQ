# %%
import argparse
import os

import torch

from utils.compile import COMPILE_MODES, compile_model
from utils.device import ACCELERATORS, resolve_device
from utils.formats import supported_input_extensions, supported_output_extensions
from utils.mgfHandler.compressor import SpectrumCompressor
from utils.mgfHandler.decompressor import SpectrumDecompressor
from utils.model import getModel

CODECS = ("gzip", "zlib", "zstd")

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
    help="Compress: output path (.vqms2 added if missing; .parquet and "
    "_compounded.pkl are written next to it). Decompress: .mgf or .mzML "
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
    help="Entropy coder applied to the packed codes; omit for none. "
    "Decompression must use the same method.",
)
argparser.add_argument(
    "--compression_level",
    "-cL",
    type=int,
    help="Level for --compression_method (gzip/zlib 1-9, zstd 1-22)",
)
argparser.add_argument(
    "--batch_size", "-b", type=int, help="Batch size for compression"
)
argparser.add_argument(
    "--quantizer",
    "-q",
    type=int,
    default=4,
    help="Number of quantizers; must match between compression and decompression (default: 4)",
)
argparser.add_argument(
    "--stored_raw", "-sR", action="store_true", help="Store raw reconstructed spectra"
)
argparser.add_argument(
    "--accelerator",
    type=str,
    default="auto",
    choices=ACCELERATORS,
    help="Lightning accelerator: auto, cpu, cuda, mps",
)
argparser.add_argument(
    "--compile",
    choices=COMPILE_MODES,
    default="none",
    help="torch.compile the encoder and decoder (CUDA only)",
)
argparser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")


def main(argv=None):
    arguments = argparser.parse_args(argv)
    input_file_path = arguments.input
    output_file_path = arguments.output

    # Validate everything before the (slow) model load
    if (arguments.compression_method is None) != (arguments.compression_level is None):
        argparser.error(
            "--compression_method and --compression_level must be given together"
        )
    if arguments.quantizer < 1:
        argparser.error("--quantizer must be at least 1")
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

    # Only the selected backend receives the level; the others stay None (disabled).
    codec_levels = {f"{name}_compression_level": None for name in CODECS}
    if arguments.compression_method is not None:
        codec_levels[f"{arguments.compression_method}_compression_level"] = (
            arguments.compression_level
        )

    device = resolve_device(arguments.accelerator)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # Load the neural network model for spectrum processing
    SpectraStream = getModel.getModel()
    # Apply pre-trained weights to the model
    SpectraStream = getModel.ApplyWeights(SpectraStream, arguments.weights)
    # Move model to the Lightning-selected device and set to evaluation mode
    SpectraStream.to(device)
    SpectraStream.eval()
    compile_model(SpectraStream, arguments.compile, device)

    verboseInt = 2 if arguments.verbose else 0
    if arguments.compress:
        compressor = SpectrumCompressor(
            inputFilePath=input_file_path,
            OutputFileName=output_file_path,
            model=SpectraStream,
            batch_size=arguments.batch_size,
            quantizer=arguments.quantizer,
            **codec_levels,
        )
        if arguments.verbose:
            print("Compressing...")
        compressor.CompressAll(verbose=verboseInt, debug=False)
    else:
        decompressor = SpectrumDecompressor(
            inputFileName=input_file_path,
            outputFilePath=output_file_path,
            model=SpectraStream,
            batch_size=arguments.batch_size,
            quantizer=arguments.quantizer,
            stored_raw=arguments.stored_raw,
            **codec_levels,
        )
        if arguments.verbose:
            print("Decompressing...")
        decompressor.DecompressAll(verbose=verboseInt)


# DataLoader workers re-import this module on spawn platforms (macOS, Windows)
if __name__ == "__main__":
    main()
