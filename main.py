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

# Set up command line argument parser for compression/decompression modes
argparser = argparse.ArgumentParser(
    description="Compress or Decompress mass spectrometry files"
)
argparser.add_argument("--compress", "-m", action="store_true", help="Mode: Compress")
argparser.add_argument(
    "--decompress", "-d", action="store_true", help="Mode: Decompress"
)
argparser.add_argument(
    "--input",
    "-i",
    type=str,
    help="Input file path (.mgf, or .vqms2 when decompressing)",
)
argparser.add_argument("--output", "-o", type=str, help="Output file name")
argparser.add_argument("--weights", "-w", type=str, help="Path to model weights")
argparser.add_argument(
    "--compression_method",
    "-cM",
    type=str,
    help="Compression/Decompress Method. Available Compression Method: [gzip,zlib,zstd]",
)
argparser.add_argument("--compression_level", "-cL", type=str, help="Compression Level")
argparser.add_argument(
    "--batch_size", "-b", type=int, help="Batch size for compression"
)
argparser.add_argument(
    "--quantizer", "-q", type=int, help="Quantizer level for compression"
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
# %%
arguments = argparser.parse_args()
compress = arguments.compress
decompress = arguments.decompress
input_file_path = arguments.input
output_file_path = arguments.output
weights_path = arguments.weights
compression_method = arguments.compression_method
compression_level = arguments.compression_level
batch_size = arguments.batch_size
stored_raw = arguments.stored_raw
verbose = arguments.verbose
device = resolve_device(arguments.accelerator)

# Double check the input
if input_file_path is None:
    raise ValueError("Input file path is required")
if output_file_path is None:
    raise ValueError("Output file path is required")
if weights_path is None:
    raise ValueError("Weights path is required")


# Load
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
# Load the neural network model for spectrum processing
SpectraStream = getModel.getModel()
# Apply pre-trained weights to the model
SpectraStream = getModel.ApplyWeights(SpectraStream, weights_path)
# Move model to the Lightning-selected device and set to evaluation mode
SpectraStream.to(device)
SpectraStream.eval()
compile_model(SpectraStream, arguments.compile, device)

level = int(compression_level) if compression_level is not None else None
# Only the selected backend receives the level; the others stay None (disabled).
codec_levels = {
    "gzip_compression_level": None,
    "zlib_compression_level": None,
    "zstd_compression_level": None,
}
if compression_method in ("gzip", "zlib", "zstd"):
    codec_levels[f"{compression_method}_compression_level"] = level

if compress + decompress > 1:
    raise ValueError("Mode must be either compress or decompress, not both")
elif compress:
    if os.path.splitext(input_file_path)[1].lower() not in supported_input_extensions():
        raise ValueError(
            f"Input file must be one of: {', '.join(supported_input_extensions())}"
        )
    compressor = SpectrumCompressor(
        inputFilePath=input_file_path,
        OutputFileName=output_file_path,
        model=SpectraStream,
        batch_size=batch_size,
        quantizer=4,
        **codec_levels,
    )
    if verbose:
        print("Compressing...")
    verboseInt = 2 if verbose else 0
    compressor.CompressAll(verbose=verboseInt, debug=False)
elif decompress:
    if not input_file_path.endswith(".vqms2"):
        raise ValueError("Input file must be a .vqms2")
    if (
        os.path.splitext(output_file_path)[1].lower()
        not in supported_output_extensions()
    ):
        raise ValueError(
            f"Output file must be one of: {', '.join(supported_output_extensions())}"
        )
    decompressor = SpectrumDecompressor(
        inputFileName=input_file_path,
        outputFilePath=output_file_path,
        model=SpectraStream,
        batch_size=batch_size,
        quantizer=4,
        stored_raw=stored_raw,
        **codec_levels,
    )
    if verbose:
        print("Decompressing...")
    verboseInt = 2 if verbose else 0
    decompressor.DecompressAll(verbose=verboseInt)
# %%
