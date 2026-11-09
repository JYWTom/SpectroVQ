from ..Compression import compress
from ..Compression import torchprocessor
from ..formats import get_reader
import os
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pickle
import warnings


DEFAULTMGFPARAMSKEYS = [
    "title",
    "rtinseconds",
    "pepmass",
    "charge",
    "scans",
    "seq",
    "modifications",
]


class SpectrumCompressor:
    def __init__(
        self,
        model,
        inputFilePath,
        OutputFileName=None,
        batch_size=None,
        quantizer=6,
        gzip_compression_level=None,
        zlib_compression_level=6,
        zstd_compression_level=None,
        outputOutOfRange=True,
    ):
        """
        Initialize the SpectrumCompressor for compressing mass spectrometry data.

        Args:
            model: Neural network model for encoding spectra
            inputFilePath (str): Path to the input spectra file (any registered format, e.g. .mgf)
            OutputFileName (str, optional): Output path without extension (a trailing .vqms2 is
                accepted); writes <name>.vqms2, <name>.parquet and, if needed, <name>_compounded.pkl.
                Defaults to the input filename.
            batch_size (int, optional): Number of spectra to process in each batch. Defaults to 128.
            quantizer (int): Number of quantization levels for vector quantization. Defaults to 6.
            gzip_compression_level (int, optional): Compression level for gzip. Defaults to None.
            zlib_compression_level (int): Compression level for zlib. Defaults to 6.
            outputOutOfRange (bool): Whether to include peaks outside the range of 150-1500Th. Defaults to True.
        """
        self.model = model
        self.inputFilePath = inputFilePath
        self.OutputFileName = OutputFileName
        if self.OutputFileName is None:
            self.OutputFileName = os.path.basename(inputFilePath).split(".")[0]
            warnings.warn(
                f"Output file name is not defined; using input name {self.OutputFileName} as output name",
                stacklevel=2,
            )
        # The extensions are appended below; accept a name that already has one
        if self.OutputFileName.endswith(".vqms2"):
            self.OutputFileName = self.OutputFileName[: -len(".vqms2")]
        self.reader = get_reader(inputFilePath)
        self.device = next(self.model.parameters()).device
        self.packer = compress.BitPacker(
            10,
            open(self.OutputFileName + ".vqms2", "wb"),
            gzip_compression=gzip_compression_level,
            zlib_compression=zlib_compression_level,
            zstd_compression=zstd_compression_level,
        )
        self.gzip_compression_level = gzip_compression_level
        self.parquet_compression_level = (
            zlib_compression_level
            if zlib_compression_level is not None
            else gzip_compression_level
        )
        self.quantizer = quantizer
        self.metadataFileName = self.OutputFileName + ".parquet"
        self.MaxValList = []
        self.ExtendedMzList, self.ExtendedIntensityList = [], []
        self.outputOutOfRange = outputOutOfRange

        self.batch_size = 128 if batch_size is None else batch_size

        self.compounded_codes_list = []
        self.compounded_idx_list = []

    def ProcessMetaData(self):
        """
        Process and save metadata from compressed spectra to a parquet file.

        Extracts relevant metadata from the parameter list, processes pepmass and charge data,
        adds maximum values, and saves to a compressed parquet file.
        """
        df_meta = pd.DataFrame(self.paramList)
        # df_meta['pepmass'] = df_meta['pepmass'].astype(str)
        df_meta["pepmz"] = df_meta["pepmass"].apply(lambda x: x[0])
        df_meta["pepit"] = df_meta["pepmass"].apply(
            lambda x: x[1] if len(x) > 1 else np.nan
        )
        if "charge" in df_meta.columns:
            df_meta["charge"] = df_meta["charge"].apply(lambda x: x[0])
        df_meta["MaxVal"] = self.MaxValList
        if self.outputOutOfRange:
            df_meta["ExtendedMZ"] = self.ExtendedMzList
            df_meta["ExtendedIT"] = self.ExtendedIntensityList
        df_meta.pop("pepmass")
        df_meta.to_parquet(
            self.metadataFileName,
            compression="gzip",
            compression_level=self.parquet_compression_level,
        )

    # Mode 1: Load all spectra into MEMORY, then perform compression
    def LoadAll(self, debug: bool = False):
        """
        Load all spectra from the input file into memory.

        Reads all spectra via the format reader and stores m/z arrays, intensity arrays,
        and parameters for subsequent compression. Initializes the MaxValList for tracking.
        """
        spectra = self.reader.read_all(limit=10 if debug else None)
        mzList = [sp.mz for sp in spectra]
        intensityList = [sp.intensity for sp in spectra]
        paramList = [sp.params for sp in spectra]
        self.mzList = mzList
        self.intensityList = intensityList
        self.paramList = paramList
        self.MaxValList = []

    def CompressAll(self, verbose=0, num_workers=4, debug=False):
        """
        Compress all loaded spectra using the neural network model.

        This method performs compression in memory after loading all spectra.
        It processes spectra in batches, extracts vector quantized codes, handles
        chimeric spectra if enabled, and saves compressed data to binary format.

        Args:
            verbose (int): Verbosity level for progress reporting. Defaults to 0.
            num_workers (int): Number of worker processes for parallel processing. Defaults to 4.
        """
        self.LoadAll(debug=debug)
        if verbose >= 1:
            print(f"Loaded {len(self.mzList)} spectra from {self.inputFilePath}")
            print(
                f"Starting Compression with batch size {self.batch_size} and quantizer {self.quantizer}"
            )
        tb = torchprocessor.SpectrumTorchBatchEncoder(
            self.mzList,
            self.intensityList,
            self.model,
            compounded=True,
            verbose=verbose,
        )
        result = tb.getReconstructIndices(
            outputOutOfRange=self.outputOutOfRange,
            quantizer=self.quantizer,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
        if verbose >= 1:
            print("Compression completed, processing results...")
        self.compounded_codes_list.extend(result.compounded_codes)
        self.compounded_idx_list.extend(result.compounded_idx)
        self.ExtendedMzList.extend(result.extended_mz)
        self.ExtendedIntensityList.extend(result.extended_it)
        self.MaxValList.extend(result.max_vals)
        output_codes = np.transpose(
            np.array(result.codes.cpu().numpy(), dtype=np.int32), (1, 0, 2)
        )  # N B D -> B N D
        # Push the first layer of quantized codes first, then push the 2nd layer of chimeric codes to the packer
        for j in range(output_codes.shape[0]):  # Data Size
            for k in range(self.quantizer):
                for l in range(
                    self.model.quantizedlen
                ):  # Number of integers in each quantizer
                    self.packer.push(output_codes[j, k, l])  # First in First out
        if verbose >= 1:
            print(f"Pushing Compounded Spectrum Codes")
        if len(self.compounded_codes_list) > 0:
            for j in range(len(self.compounded_codes_list)):
                for k in range(self.quantizer):  # Number of quantizers
                    for l in range(
                        self.model.quantizedlen
                    ):  # Number of integers in each quantizer
                        self.packer.push(self.compounded_codes_list[j][k, l])

        if verbose >= 1:
            print(f"Flushing Packer")
        self.packer.flush()
        self.packer.close()
        self.ProcessMetaData()

        if len(self.compounded_idx_list) > 0:
            if verbose >= 1:
                print(
                    f"Storing compounded data: {len(self.compounded_idx_list)} entries"
                )
            with open(self.OutputFileName + "_compounded.pkl", "wb") as f:
                pickle.dump(self.compounded_idx_list, f)


class MGFCompressor(SpectrumCompressor):
    """Backwards-compatible entry point (``mgfFilePath`` keyword)."""

    def __init__(self, model, mgfFilePath, *args, **kwargs):
        super().__init__(model, mgfFilePath, *args, **kwargs)
        self.mgfFilePath = mgfFilePath
