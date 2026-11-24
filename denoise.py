#!/usr/bin/env python3
"""
Denoise spectra using MGFDenoiser from SpectroVQ.

This script loads a trained SpectroVQ model and uses it to denoise MGF files.
"""

import argparse
import os
import sys
import traceback

# Add utils to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils import config
from utils.config import ACCELERATORS
from utils import pipelines
from utils.model import loader


def load_model(model_path, device="auto", yaml_config=None):
    """Load a trained SpectroVQ model from checkpoint (utils.model.loader)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return loader.load_model(model_path, yaml_config, device, cache=False)


def main():
    parser = argparse.ArgumentParser(description='Denoise MGF spectra using SpectroVQ')
    
    # Required arguments
    parser.add_argument('--input_mgf', type=str, required=True,
                        help='Path to input MGF file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.ckpt or .pt file)')
    parser.add_argument('--output_mgf', type=str, required=True,
                        help='Path to output denoised MGF file')
    
    # Optional arguments
    parser.add_argument(
        "--yaml_config",
        type=str,
        default=None,
        help="Path to YAML configuration file for model parameters",
    )
    parser.add_argument(
        "--accelerator",
        "--device",
        dest="accelerator",
        type=str,
        default=config.DEFAULT_ACCELERATOR,
        choices=ACCELERATORS,
        help="Lightning accelerator: auto, cpu, cuda, mps (--device is a deprecated alias)",
    )
    parser.add_argument(
        "--num_quantizers",
        type=int,
        default=config.DENOISE_QUANTIZERS,
        help="Number of quantizers to use for denoising",
    )
    parser.add_argument(
        "--retain_original_peaks",
        action="store_true",
        default=config.RETAIN_ORIGINAL_PEAKS,
        help="Retain original peaks outside the m/z range",
    )
    parser.add_argument(
        "--no_retain_original_peaks",
        dest="retain_original_peaks",
        action="store_false",
        help="Do not retain original peaks outside the m/z range",
    )
    parser.add_argument(
        "--one_percent_threshold",
        action="store_true",
        default=config.ONE_PERCENT_THRESHOLD,
        help="Apply 1%% intensity threshold",
    )
    parser.add_argument(
        "--no_one_percent_threshold",
        dest="one_percent_threshold",
        action="store_false",
        help="Do not apply 1%% intensity threshold",
    )
    parser.add_argument(
        "--compounded_spectra",
        action="store_true",
        default=config.COMPOUNDED_SPECTRA,
        help="Enable raw spectrum reconstruction",
    )
    parser.add_argument(
        "--no_compounded_spectra",
        dest="compounded_spectra",
        action="store_false",
        help="Disable raw spectrum reconstruction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.DENOISE_NUM_WORKERS,
        help="Number of workers for data loading",
    )

    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_mgf):
        print(f"Error: Input MGF file not found: {args.input_mgf}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_mgf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        # Load model
        print(f"Loading model from {args.model_path}...")
        if args.yaml_config:
            print(f"Using configuration from {args.yaml_config}")
        model = load_model(args.model_path, args.accelerator, args.yaml_config)
        print(f"Model loaded successfully on {next(model.parameters()).device}")

        # Run denoising
        print(f"Starting denoising process...")
        print(f"Input: {args.input_mgf}")
        print(f"Output: {args.output_mgf}")
        print(f"Using {args.num_quantizers} quantizers")

        pipelines.denoise(
            model,
            args.input_mgf,
            args.output_mgf,
            args.num_quantizers,
            retain_original_peaks=args.retain_original_peaks,
            one_percent_threshold=args.one_percent_threshold,
            compounded_spectra=args.compounded_spectra,
            num_workers=args.num_workers,
        )
        
        print("Denoising completed successfully!")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error during denoising: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
