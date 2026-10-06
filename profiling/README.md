# CPU-only profiling of SpectroVQ compression / decompression

All numbers below were measured **on CPU only** (no CUDA) with the published weights
(`spectroVQ_weights.ckpt`, Zenodo record [20043624](https://zenodo.org/records/20043624)).

## Setup

| | |
|---|---|
| CPU | Intel Xeon (Sapphire Rapids class, AVX-512 / AMX), 4 vCPUs, 15 GB RAM |
| Software | Python 3.13, PyTorch 2.14.1+cpu, `torch.get_num_threads() == 4` |
| Model | 49.5 M params (encoder 24.7 M, decoder 24.8 M) + 12 codebooks of 1024×256; 27 codes of 10 bits per quantizer per spectrum |
| Default run | `--quantizer 4`, gzip level 6, `num_workers=0` |
| Real data | 2,000 spectra sampled from `predicted_best_ecoli.splib` (`splibLibraries.zip` on Zenodo), converted with `splib_to_mgf.py` (median 52 peaks per spectrum) |
| Synthetic data | 2,000 spectra from `make_synthetic_mgf.py` (b/y ions + noise, about 246 peaks per spectrum) |

Each configuration was run once on a shared VM. Repeated runs of the same decompression
setup varied by about 10%, so only differences well above that are meaningful.

## Headline results (2,000 E. coli library spectra, 4 threads)

| Configuration | Compress ms/spectrum | Decompress ms/spectrum | Output |
|---|---:|---:|---|
| Baseline: batch 32 (README example) | **100.3** | **53.4** | reference |
| Batch 32, DataLoader `num_workers=4` (what `main.py` uses) | 96.1 | 54.6 | identical codes |
| Batch 8 | 71.0 | 42.4 | identical codes |
| Batch 32 + flush codebook denormals | 86.8 | 54.8 | identical codes |
| Batch 8 + flush denormals | 55.8 | 42.3 | identical codes |
| **Batch 8 + flush denormals + dedupe encoder pass** | **43.8** (2.3× faster) | 48.0 | identical codes and compounded indices |
| Batch 8 + flush denormals, `--quantizer 8` | 64.9 | 45.5 | larger output (ratio 2.5×) |
| Batch 8 + flush denormals, zstd level 15 | 57.8 | 46.8 | ratio 3.4× (gzip 6 gives 4.2×) |
| Synthetic, batch 32 baseline | 107.5 | 67.9 | |
| Synthetic, batch 8 + flush denormals | 61.5 | 50.4 | |

"Identical codes" means the 415,476 10-bit codes in the `.vqms2` payload are bit-for-bit
the same as the baseline's. The decompressed MGFs differ only around the sixth significant
digit of intensity (for example 1894.4868 vs 1894.4873). That comes from float rounding in
the decoder, which depends on batch size.

Peak RSS is about 1.3 GB. Loading the model takes about 4 s.

## Where the time goes

### Compression (baseline, batch 32): 98.5% neural network

```
model.encoder             126.7 s  63.2%   189 calls  (3 encoder passes per batch)
model.decoder              45.0 s  22.5%    63 calls  (1 decoder pass per batch)
model.vq_encode            25.7 s  12.8%   189 calls
pre.spectrum_to_vector      0.8 s   0.4%
io.bitpacker_push           0.7 s   0.3%   415,476 calls
MGF read + parquet write    0.4 s   0.2%
```

For each batch, `SpectrumTorchBatchEncoder.getReconstructIndices` (`utils/Compression/torchprocessor.py`) runs:
1. `forwardWithnumQuantizers(x)`: encoder, VQ, then decoder (to find the residual / compounded peaks)
2. `encode(residual)`: encoder and VQ on the residual spectrum
3. `encode(x)`: encoder and VQ on **the same input as step 1**, which recomputes the codes step 1 already produced

### Decompression (baseline, batch 32): 98% decoder

```
model.decoder             104.5 s  97.9%   179 calls
everything else             2.2 s   2.1%   (bit unpacking, peak extraction, MGF write)
```

A spectrum flagged as "compounded" costs **3 decoder passes**, against 1 for a normal
spectrum. `DecompressAll` decodes its original codes and its residual codes in the compounded
loop, then decodes its original codes **again** in the raw loop. In this library sample,
1,847 of 2,000 spectra (92%) were flagged as compounded; all 2,000 synthetic spectra were. The
example in `main.out` (real Orbitrap data) had 3 of 10. Decompression cost therefore depends on
the data: about 1 + 2 × (fraction compounded) decoder passes per spectrum.

### Inside the network (per-layer, batch 32, ms per spectrum)

The two highest-resolution `SEANetResnetBlock`s in the encoder (4.6 + 4.3 ms) and their
mirrors in the decoder (4.4 + 4.6 ms) make up about half of all model time. Each 6-layer
Transformer costs about 2.9 ms. By operator: oneDNN convolution 37%, matmul 18%,
`reflection_pad1d` 11% (padding at the full 13,500-bin length), and ELU 8%.

### Scaling (from `bench_model_cpu.py`)

| threads \ batch | 1 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|
| encoder ms/spectrum, 1 thread | 50.9 | 38.3 | 52.2 | 91.2 |
| encoder ms/spectrum, 4 threads | 25.3 | **13.4** | 16.6 | 22.9 |
| decoder ms/spectrum, 4 threads | 23.4 | **12.1** | 19.0 | 25.2 |

- **Batch 8 is the sweet spot on CPU.** Larger batches push the 13,500-length activations
  out of cache and get slower per spectrum.
- Going from 1 to 4 threads gives about 3× on the encoder and decoder.
- **The number of quantizers barely affects compute.** VQ encoding at batch 32 costs 3.9, 4.4,
  4.3 and 4.7 ms/spectrum for 1, 4, 8 and 12 quantizers, and decoding is flat at about 18 to
  20 ms. Bit-packing time grows with the quantizer count, but it is negligible.
- `num_workers=4` in the DataLoader saves less than 5%, because preprocessing is only 0.4% of
  the time. On a 4-core machine, the workers compete with PyTorch's intra-op threads.

## Findings and recommendations

1. **Denormal floats in codebook 0 make VQ about 150× slower than it should be.** 49,664 of
   262,144 entries in `quantization.layers.0._codebook.embed` are denormal (|x| < 1.2e-38). x86
   handles them with microcode assists, so one (864×256)·(256×1024) distance matmul takes about
   123 ms instead of 0.8 ms. `torch.set_flush_denormal(True)` does **not** fix it (about 10%
   gain). Zeroing those entries once after loading makes `vq_encode` about 12× faster overall
   (25.7 s down to 2.1 s) and gives **bit-identical codes**. Suggested fix in `ApplyWeights`:
   ```python
   tiny = torch.finfo(torch.float32).tiny
   for layer in model.quantization.layers:
       e = layer._codebook.embed
       e.data[(e != 0) & (e.abs() < tiny)] = 0.0
   ```
   This affects GPU runs too, if their kernels are not set to flush-to-zero.
2. **Use batch size 8 on CPU** instead of 32: about 29% faster compression and 21% faster
   decompression, with identical codes.
3. **Remove the duplicated encoder+VQ pass in compression.** Return the codes from
   `forwardWithnumQuantizers` and reuse them instead of calling `model.encode(inputBatch, ...)`
   again. Measured with `--dedupe_encode`: 55.8 to 43.8 ms/spectrum (−22%), identical output.
4. **Cache the original-code decode in `DecompressAll`** for compounded spectra: decode once and
   reuse it in the raw loop. Estimated, not measured: removes 1 of the 3 decoder passes for each
   compounded spectrum (up to about 30% of decompression time on this library sample).
5. **gzip level 6 beats zstd level 15** on this payload (4.2× vs 3.4× overall ratio) at the same
   speed. The 10-bit codes are already close to incompressible, and the parquet metadata file is
   27% of the output for this sample.
6. **bf16 autocast is not worth it**: no speedup at batch 8, about 23% faster at batch 32 (still
   slower than fp32 at batch 8), and it changes 2.7% of the codes (`bench_optimizations_cpu.py`).
7. Pure-Python stages (per-value `BitPacker.push`/`BitUnpacker.pull`, peak extraction, MGF I/O)
   are under 2% of the time. They aren't worth optimizing until the network gets much faster.

Combining 1–3 (all output-preserving) gives **100 ms → 44 ms per spectrum for compression
(about 23 spectra/s on 4 cores)** and 53 → 42–48 ms per spectrum for decompression. Applying
recommendation 4 should lower decompression further on spectra that are mostly compounded.

### Issues found while profiling
- `main.py` hard-coded `SpectraStream.to('cuda')`, so it could not run on CPU at all. It now
  falls back to CPU when CUDA is unavailable (verified with a 200-spectrum compress/decompress
  round trip).
- `main.py` ignores `--quantizer` and always passes `quantizer=4` to the compressor and
  decompressor.
- `MGFDecompressor.__init__` leaves `compounded_idx_list` unset when no `_compounded.pkl` exists,
  so `DecompressAll` raises `AttributeError`.
- With gzip, `BitPacker.flush()` closes the `GzipFile` but not the underlying file it was given,
  so the `.vqms2` is only fully written once that file object is garbage-collected. This matters
  when compressing and decompressing in the same process.

## Reproduce

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy pandas pyarrow pyteomics pyyaml einops zstandard scipy matplotlib
curl -L -o spectroVQ_weights.ckpt https://zenodo.org/api/records/20043624/files/spectroVQ_weights.ckpt/content

# test data (either)
python profiling/make_synthetic_mgf.py --n 2000 --out synthetic.mgf
python profiling/splib_to_mgf.py --splib predicted_best_ecoli.splib --out ecoli.mgf --n 2000

# end-to-end stage breakdown (add --cprofile for function-level stats)
python profiling/profile_cpu.py --weights spectroVQ_weights.ckpt --mgf ecoli.mgf --batch_size 32
python profiling/profile_cpu.py --weights spectroVQ_weights.ckpt --mgf ecoli.mgf --batch_size 8 \
    --flush_denormals --dedupe_encode

# model-only: per-layer timing, batch/thread/quantizer sweeps; optimization variants
python profiling/bench_model_cpu.py --weights spectroVQ_weights.ckpt
python profiling/bench_optimizations_cpu.py --weights spectroVQ_weights.ckpt --batch_size 8
```

Raw outputs of the runs above are in `profiling/results/`.
