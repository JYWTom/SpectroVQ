# SpectroVQ
SpectroVQ: Noise-Aware Compression of Proteomics Data via Vector-Quantized Deep Learning improves MS/MS data storage and Peptide Identification ![link](https://www.biorxiv.org/content/10.64898/2026.09.05.749393v1.full)

![](https://github.com/kenll99minecart/SpectroVQ/blob/main/Figure1Final_v2.png)

### Reconstruction with Increasing Number of Quantizers

The animation below shows how the reconstructed spectrum improves as more residual quantizers are used (1 to 12). SpectroVQ allows variable compression rate by using different number of quantizers. Higher number of quantizers yields higher spectrum fidelity, but the storage size increase.
![](https://github.com/kenll99minecart/SpectroVQ/blob/main/quantizer_reconstruction_spectrum.gif)


### Huggingface Demo 🤗

A demo version of SpectroVQ is available in  https://huggingface.co/spaces/kenll/SpectroVQ.

## Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (recommended for training and inference)

### Install
Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. From the repository root:
```bash
uv sync
```
This creates `.venv` with the locked dependencies (including the `dev` group: pytest, ruff). Run commands with `uv run`, e.g. `uv run python main.py ...` or `uv run pytest`, or activate `.venv` and use `python` directly as in the examples below.

### PyTorch and CUDA
`uv sync` installs the default PyPI build of PyTorch. If you need a specific CUDA build, check your CUDA version with `nvidia-smi` and point uv at the matching PyTorch index by adding this to `pyproject.toml` (shown for CUDA 12.8; see the [PyTorch install guide](https://pytorch.org/get-started/locally/) for other versions), then run `uv sync` again:
```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

## Usage

Current model weights is available at Zenodo https://zenodo.org/records/20043624.

### 1. Training (train.py)
Train a new SpectroVQ model on consensus library data.

```bash
python train.py --data_path library.feather --output_dir ./models --batch_size 32 --learning_rate 1e-4 --max_epochs 50
```

**Arguments:**
- `--data_path`: Path to consensus library file (.feather format) [default: library.feather]
- `--output_dir`: Output directory for trained model checkpoints [default: .]
- `--batch_size`: Batch size for training [default: 32]
- `--num_workers`: Number of workers for data loading [default: 4]
- `--learning_rate`: Learning rate for training [default: 1e-4]
- `--max_epochs`: Maximum number of training epochs [default: 50]
- `--resume`: Continue from `<output_dir>/resume/resume.ckpt`
- `--checkpoint_minutes`: Minutes between resume checkpoints [default: 30]
- `--accelerator`: Lightning accelerator: `auto`, `cpu`, `cuda` or `mps` [default: auto]
- `--compile`: `torch.compile` the encoder and decoder: `none`, `default` or `max-autotune` [default: none]

**Pausing and resuming:**

Pause by sending SIGTERM to the training process, whose PID is printed at startup, or with **Pause** in the GUI's Train tab. Signal that process only, not its process group, because the data loader workers must keep running until it has saved. It finishes the current batch, saves to `<output_dir>/resume/resume.ckpt` and exits with code 75.

```bash
kill -TERM <pid>
python train.py --data_path library.feather --output_dir ./models --batch_size 32 --max_epochs 50 --resume
```

In the GUI, select the session in the sidebar and click **Resume training**.

- Use the same `--batch_size`, `--max_epochs` and library when resuming, because they change the learning rate curve or the data. The GUI reuses the session's settings.
- A pause during validation exits without saving, and the resume continues from the last periodic checkpoint.
- After a crash or `kill -9`, `--resume` continues from the last periodic checkpoint.
- A fresh start is refused while `resume.ckpt` exists in the output directory. The `resume` folder is deleted when a run finishes.
- A resumed run continues at the same batch with the same data order and augmentation. Model randomness (quantizer dropout, dropout) is not restored, so the result is not bit-identical to an uninterrupted run.
- Lightning's warning about a non-resumable dataloader on a mid-epoch resume is expected: the sampler handles this, and Lightning's check doesn't see it.

### 2. Denoising (denoise.py)
Denoise MGF spectra using a trained SpectroVQ model.

```bash
python denoise.py --input_mgf input.mgf --model_path model.ckpt --output_mgf denoised.mgf --num_quantizers 3 --num_workers 0
```

**Arguments:**
- `--input_mgf`: Path to input MGF file [required]
- `--model_path`: Path to trained model checkpoint (.ckpt or .pt file) [required]
- `--output_mgf`: Path to output denoised MGF file [required]
- `--yaml_config`: Path to YAML configuration file for model parameters [optional]
- `--accelerator`: Lightning accelerator: `auto`, `cpu`, `cuda` or `mps`; `--device` is an alias [default: auto]
- `--num_quantizers`: Number of quantizers to use for denoising [default: 3]
- `--retain_original_peaks`: Retain original peaks outside the m/z range [default: True]
- `--no_retain_original_peaks`: Do not retain original peaks outside the m/z range
- `--one_percent_threshold`: Apply 1% intensity threshold [default: True]
- `--no_one_percent_threshold`: Do not apply 1% intensity threshold
- `--compounded_spectra`: Enable chimeric spectrum reconstruction [default: True]
- `--no_compounded_spectra`: Disable chimeric spectrum reconstruction
- `--num_workers`: Number of workers for processing [default: 0]

### 3. Compression/Decompression (main.py)
Compress MS2 spectra (MGF or mzML) to `.vqms2`, or decompress back to MGF or mzML. Only MS2 spectra are supported: MS1 scans in an mzML input are skipped. Compression writes `output.vqms2` plus the side files `output.parquet` (metadata) and, when needed, `output_compounded.pkl`; keep them together.

#### Compression:
```bash
python main.py --compress --input input.mgf --output output.vqms2 --weights model.ckpt --compression_method gzip --compression_level 6 --batch_size 32 --quantizer 4
```

#### Decompression:
```bash
python main.py --decompress --input output.vqms2 --output restored.mgf --weights model.ckpt --compression_method gzip --compression_level 6 --batch_size 32 --quantizer 4
```

**Arguments:**
- `--compress, -m`: Enable compression mode
- `--decompress, -d`: Enable decompression mode
- `--input, -i`: Input file path [required]
- `--output, -o`: Output path [required]. Compression: `.vqms2` is added if missing
- `--weights, -w`: Path to model weights [required]
- `--compression_method, -cM`: Entropy coder [gzip, zlib, zstd] [optional, default: none]
- `--compression_level, -cL`: Level for the coder (gzip/zlib 1-9, zstd 1-22); give it together with `-cM`
- Decompression must use the same `-cM`, `-cL` and `-q` as compression
- `--batch_size, -b`: Batch size for processing [optional]
- `--quantizer, -q`: Number of quantizers; must match between compression and decompression [default: 4]
- `--accelerator`: Lightning accelerator: `auto`, `cpu`, `cuda` or `mps` [default: auto]
- `--compile`: `torch.compile` the encoder and decoder: `none`, `default` or `max-autotune` [default: none]
- `--stored_raw, -sR`: Store raw reconstructed spectra [flag]

### 4. Browser GUI (gui.py)
Runs training, denoising, compression and decompression from a browser.

```bash
python gui.py --ckpt model.ckpt --compile default
```

The Compress and Explore tabs accept MGF or mzML, the Decompress tab writes either, and the Train tab can pause, resume and compile a run. Denoising takes MGF only. `--compile` applies to the models the GUI loads, and the Train tab has its own setting. Run `python gui.py --help` for all options.

### torch.compile
`torch.compile` needs CUDA and your Python's headers (RHEL: `python3-devel`); without them it warns and runs eager. Compiling takes time up front, so it pays off on long runs rather than small files. Compiled output is very close to eager output but not byte-identical.

## File Formats
- **Input**: MGF files for denoising and compression; mzML files for compression. Only MS2 spectra are supported; MS1 is out of scope
- **Compressed Output**: .vqms2 files (SpectroVQ compressed format) with a .parquet metadata side file
- **Decompressed Output**: MGF, or mzML for archives compressed from mzML

### mzML metadata
Compressing an mzML keeps its metadata in the .parquet file: the run header (instrument, software, sample, processing history) in the Parquet footer, and each per-spectrum parameter in its own column. Decompressing to `.mzML` restores it, with these differences from the source:
- The file has MS2 spectra only, with no MS1 scans or chromatograms, and spectrum `index` values are renumbered from 0. Spectrum ids keep their original scan numbers.
- Parent scans are recorded with `sourceFileRef` and `externalSpectrumID`, pointing to the source raw file, because `spectrumRef` must resolve within the file.
- Base peak, total ion current and m/z range are recomputed from the reconstructed peaks.
- The processing history records the MS1 removal and the lossy reconstruction, including the model hash and quantizer count.
- Peaks are lossy reconstructions, not the original data.

`--stored_raw` is not supported for mzML output because the raw copies would duplicate spectrum ids. Archives compressed from MGF have no mzML metadata, so they can only be decompressed to MGF.
- **Model Checkpoints**: .ckpt or .pt files
- **Training Data**: .feather files (should contain the columns: 'PrecursorMZ', 'CompoundName', 'Comment', 'm/z', 'intensity', 'Charge')

### Training Data Format

The training data for `train.py` should be a `.feather` file containing the following columns:
- `PrecursorMZ`: Precursor m/z value
- `CompoundName`: Peptide sequence with charge (e.g., 'NTDSIELALSYAK/2')
- `Comment`: Metadata and annotations for the spectrum
- `m/z`: Array of m/z values for peaks
- `intensity`: Array of intensity values for peaks
- `Charge`: Charge state of the precursor ion

**Example row:**
```python
# Pandas DataFrame row structure:
{
    "PrecursorMZ": "712.8670",
    "CompoundName": "NTDSIELALSYAK/2",
    "Comment": "AvePrecursorMz=713.2927 BestRawSpectrum=20101210_Velos1_AnWe_SA_U2OS_4.23852.23852 BinaryFileOffset=4069648408 CollisionEnergy=40.0 ConsFracAssignedPeaks=0.660 DotConsensus=0.83,0.07;1/6 FracUnassigned=0.65,3/5;0.47,7/20;0.52,272/380 Inst=1/orbitrap,6,8 MassDiff=0.0050 MassDiffCounts=1/0",
    "m/z": array(
        [
            102.0552,
            110.0712,
            120.0808,
            126.0552,
            129.0708,
            129.0975,
            130.0864,
            136.0759,
            137.0793,
            143.0818,
            144.081,
            147.113,
            153.0659,
            155.1011,
            157.1261,
            169.0975,
            170.0927,
            171.0768,
            171.1128,
            173.1288,
            181.0608,
            183.1291,
            185.1234,
            185.1578,
            187.0716,
            187.1367,
            188.1032,
            189.0879,
            197.129,
            198.0876,
            199.0717,
            200.1394,
            201.1237,
            203.0669,
            211.1444,
            213.087,
            215.1395,
            216.0982,
            217.0822,
            218.1505,
            223.1073,
            228.1343,
            235.1068,
            243.1342,
            244.1654,
            251.1033,
            254.151,
            260.1975,
            270.0718,
            272.1602,
            286.1034,
            298.1448,
            304.1133,
            312.1555,
            313.1147,
            314.1263,
            314.1432,
            316.1508,
            330.1669,
            331.1248,
            355.1616,
            356.2192,
            367.2342,
            371.1931,
            373.1362,
            381.2138,
            383.1186,
            383.1939,
            400.1467,
            401.1298,
            417.2165,
            418.1581,
            427.183,
            427.2567,
            445.1924,
            468.2466,
            469.2494,
            496.2781,
            513.2329,
            514.2884,
            546.2512,
            558.2807,
            581.3308,
            582.3352,
            629.3163,
            652.3674,
            653.3726,
            765.4535,
            766.4645,
            894.4956,
            895.4965,
            1007.5797,
            1008.5803,
            1094.6135,
            1095.6171,
            1209.6326,
            1210.636,
        ]
    ),
    "intensity": array(
        [
            1251.9,
            3081.0,
            10000.0,
            460.7,
            1902.3,
            4007.2,
            1789.7,
            4250.5,
            413.5,
            2695.4,
            3648.6,
            3577.0,
            558.1,
            522.3,
            646.2,
            104.1,
            420.8,
            3134.1,
            174.1,
            1020.8,
            602.2,
            960.8,
            1101.9,
            468.9,
            714.2,
            112.2,
            5302.8,
            465.7,
            2731.5,
            1436.8,
            2044.1,
            1062.9,
            1553.4,
            492.7,
            579.8,
            1168.0,
            662.6,
            6195.6,
            675.1,
            2593.2,
            424.2,
            3342.1,
            349.8,
            822.8,
            383.4,
            243.2,
            614.5,
            577.7,
            126.7,
            375.6,
            678.7,
            1053.9,
            629.6,
            129.1,
            667.1,
            612.9,
            1127.5,
            369.9,
            328.6,
            578.0,
            437.2,
            162.7,
            120.6,
            584.7,
            558.3,
            2099.5,
            529.3,
            451.1,
            2640.1,
            628.3,
            138.7,
            783.3,
            170.2,
            448.8,
            345.4,
            3475.9,
            744.4,
            900.8,
            475.0,
            609.6,
            426.0,
            485.8,
            2657.4,
            501.3,
            499.5,
            3170.6,
            1354.8,
            2683.4,
            993.7,
            4282.7,
            2469.1,
            1064.8,
            807.5,
            1357.3,
            562.1,
            1604.0,
            1197.5,
        ]
    ),
    "Charge": 2,
}
```

## Notes
- GPU acceleration is highly recommended for training and large-scale processing
- The model automatically detects CUDA availability
- For training, ensure your consensus library contains proper peak annotations and metadata


## Pythonic Interface

To run the model throught python, use the following example code.
```python
from SpectrumProcessing import SpectraLoading
import torch
from model.getModel import getModel, ApplyWeights

# Model Loading
SpectroVQ = getModel("modelParams.yaml")
SpectroVQ = ApplyWeights(SpectroVQ, "model.ckpt")
SpectroVQ.eval()

# Spectrum Preprocessing
mz, it = [100.0, 200.0, 300.0, ...], [100.0, 200.0, 300.0, ...]
BinnedSpectrum, _ = SpectraLoading.massSpectrumToVector(
    mz,
    it,
    bin_size=0.1,
    SPECTRA_DIMENSION=13500,
    rawIT=False,
    Mode=None,
    mean0=False,
    CenterIntegerBins=False,
    AlterMZ=False,
    GenerateMZList=False,
    MZDiff=False,
    mzRange=[150, 1500],
)
BinnedSpectrum = torch.sqrt(BinnedSpectrum)

# Model Inference
NumberofQuantizers = 12
reconstructed_spectrum, codebookIndices, _, _ = SpectroVQ.encode(
    BinnedSpectrum, NumberofQuantizers, returnAll=True
)
```
