"""CPU-only profiling of SpectroVQ compression and decompression.

Runs the real MGFCompressor / MGFDecompressor pipeline on CPU and reports a
per-stage time breakdown (by wrapping the main functions with timers), end-to-end
throughput, peak memory and compression ratio. Optionally writes a cProfile dump.

Example:
    python profiling/profile_cpu.py --weights spectroVQ_weights.ckpt --mgf input.mgf \
        --quantizer 4 --batch_size 32 --threads 4 --method gzip --level 6
"""
import argparse
import cProfile
import functools
import gc
import json
import os
import pstats
import resource
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.Compression import batchprocessor, compress, torchprocessor  # noqa: E402
from utils.mgfHandler import compressor as mgf_compressor  # noqa: E402
from utils.mgfHandler import decompressor as mgf_decompressor  # noqa: E402
from utils.model import getModel  # noqa: E402
from utils.SpectrumProcessing import SpectraLoading  # noqa: E402

TIMES = defaultdict(float)
CALLS = defaultdict(int)


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMES[name] += time.perf_counter() - t
            CALLS[name] += 1
    return wrapper


def patch_attr(obj, attr, name):
    setattr(obj, attr, timed(name, getattr(obj, attr)))


def instrument_model(model):
    patch_attr(model.encoder, 'forward', 'model.encoder')
    patch_attr(model.decoder, 'forward', 'model.decoder')
    patch_attr(model.quantization, 'encodeWithLoss', 'model.vq_encode')
    patch_attr(model.quantization, 'decode', 'model.vq_decode')


def dedupe_encode(model):
    """Experimental: reuse the encoder+VQ result of forwardWithnumQuantizers for the following
    model.encode(...) call on the same input batch (SpectrumTorchBatchEncoder computes it twice)."""
    orig_encode = model.encode
    cache = {}

    def forward_with_n_q(x, n_q):
        z = model.encoder(x, None)
        zq = z.view(z.shape[0], model.quantizationdim, -1)
        embedding, codes, q_loss = model.quantization.encodeWithLoss(zq, n_q)
        cache['key'], cache['val'] = (x, n_q), (embedding, codes, q_loss, zq)
        decoded = model.decoder(embedding)
        return decoded[:, :, :x.shape[2]], q_loss, zq

    def encode(x, numquantizer=16, returnCodebookIndices=False, returnAll=False):
        key = cache.get('key')
        if returnAll and key is not None and key[0] is x and key[1] == numquantizer:
            cache.pop('key')
            return cache.pop('val')
        return orig_encode(x, numquantizer=numquantizer, returnCodebookIndices=returnCodebookIndices,
                           returnAll=returnAll)

    model.forwardWithnumQuantizers = forward_with_n_q
    model.encode = encode


def instrument_pipeline():
    # Preprocessing: binning of peaks into the 13500-d vector (called by the Dataset)
    patch_attr(SpectraLoading, 'massSpectrumToVector', 'pre.spectrum_to_vector')
    # Bit packing (entropy-coding stage) is pure python, one call per code
    patch_attr(compress.BitPacker, 'push', 'io.bitpacker_push')
    patch_attr(compress.BitPacker, 'flush', 'io.bitpacker_flush')
    patch_attr(compress.BitUnpacker, 'pull', 'io.bitunpacker_pull')
    patch_attr(mgf_compressor.MGFCompressor, 'LoadAll', 'io.mgf_read')
    patch_attr(mgf_compressor.MGFCompressor, 'ProcessMetaData', 'io.metadata_parquet_write')
    patch_attr(mgf_decompressor.pd, 'read_parquet', 'io.metadata_parquet_read')
    patch_attr(batchprocessor.SpectrumBatchDecoder, 'postprocessingSpectrum', 'post.vector_to_peaks')
    patch_attr(mgf_decompressor.MGFDecompressor, 'formatMGFformat', 'post.format_mgf_records')
    patch_attr(mgf_decompressor.mgf, 'write', 'io.mgf_write')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def level_kwargs(method, level):
    kw = {'gzip_compression_level': None, 'zlib_compression_level': None, 'zstd_compression_level': None}
    if method != 'none':
        kw[f'{method}_compression_level'] = level
    return kw


def report(title, wall, n_spectra):
    print(f'\n=== {title}: {wall:.2f}s wall, {n_spectra / wall:.1f} spectra/s, '
          f'{1000 * wall / n_spectra:.2f} ms/spectrum ===')
    accounted = 0.0
    rows = sorted(TIMES.items(), key=lambda kv: -kv[1])
    for name, t in rows:
        # encoder/decoder/vq timings are nested inside nothing else we time, so they sum
        accounted += t
        print(f'  {name:28s} {t:8.2f}s {100 * t / wall:5.1f}%  calls={CALLS[name]}')
    print(f'  {"other (unattributed)":28s} {wall - accounted:8.2f}s {100 * (wall - accounted) / wall:5.1f}%')
    return {k: {'seconds': v, 'calls': CALLS[k]} for k, v in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--mgf', required=True)
    p.add_argument('--out_dir', default='profile_out')
    p.add_argument('--quantizer', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--threads', type=int, default=torch.get_num_threads())
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers for compression (main.py uses the default of 4). '
                        'Stage timings for preprocessing are only captured with 0.')
    p.add_argument('--method', choices=['gzip', 'zlib', 'zstd', 'none'], default='gzip')
    p.add_argument('--level', type=int, default=6)
    p.add_argument('--cprofile', action='store_true', help='Also dump cProfile stats')
    p.add_argument('--json', help='Write results to this JSON file')
    p.add_argument('--dedupe_encode', action='store_true',
                   help='Experimental: skip the duplicated encoder+VQ pass during compression')
    p.add_argument('--flush_denormals', action='store_true',
                   help='Zero denormal codebook entries after loading (output is bit-identical, VQ is much faster)')
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, os.path.splitext(os.path.basename(a.mgf))[0])
    results = {'config': vars(a), 'torch': torch.__version__, 'cpu_count': os.cpu_count()}

    t = time.perf_counter()
    model = getModel.getModel()
    model = getModel.ApplyWeights(model, a.weights)
    model.to('cpu').eval()
    if a.flush_denormals:
        from bench_optimizations_cpu import flush_codebook_denormals
        print(f'Flushed {flush_codebook_denormals(model)} denormal codebook values')
    results['model_load_s'] = time.perf_counter() - t
    print(f'Model load: {results["model_load_s"]:.2f}s, threads={torch.get_num_threads()}')

    instrument_model(model)
    if a.dedupe_encode:
        dedupe_encode(model)
    instrument_pipeline()
    kw = level_kwargs(a.method, a.level)

    # ---------------- Compression ----------------
    prof = cProfile.Profile() if a.cprofile else None
    TIMES.clear(); CALLS.clear()
    t = time.perf_counter()
    if prof: prof.enable()
    comp = mgf_compressor.MGFCompressor(mgfFilePath=a.mgf, OutputFileName=base, model=model,
                                        batch_size=a.batch_size, quantizer=a.quantizer, **kw)
    comp.CompressAll(verbose=0, num_workers=a.num_workers)
    # Make sure the .vqms2 is fully on disk before decompressing in the same process
    # (gzip's GzipFile.close() does not close the underlying file object it was given).
    if a.method == 'none':
        comp.packer.fo.close()
    elif a.method == 'zstd':
        comp.packer.fo.fo.close()
    if prof: prof.disable()
    wall_c = time.perf_counter() - t
    n = len(comp.mzList)
    n_compounded = len(comp.compounded_idx_list)
    del comp
    gc.collect()
    results['compress'] = {'wall_s': wall_c, 'n_spectra': n, 'n_compounded': n_compounded,
                           'stages': report('Compression', wall_c, n), 'peak_rss_mb': peak_rss_mb()}
    if prof:
        prof.dump_stats(base + '_compress.prof')
        print('\nTop compression functions (cumulative):')
        pstats.Stats(prof).sort_stats('tottime').print_stats(15)

    in_size = file_size(a.mgf)
    out_files = [base + ext for ext in ('.vqms2', '.parquet', '_compounded.pkl')]
    sizes = {os.path.basename(f): file_size(f) for f in out_files}
    total = sum(sizes.values())
    results['sizes'] = {'input_mgf': in_size, **sizes, 'total_compressed': total,
                        'ratio': in_size / total if total else None}
    print(f'\nSizes: input {in_size / 1e6:.2f} MB -> {total / 1e6:.3f} MB '
          f'(ratio {in_size / total:.1f}x) {sizes}')

    # ---------------- Decompression ----------------
    prof = cProfile.Profile() if a.cprofile else None
    TIMES.clear(); CALLS.clear()
    t = time.perf_counter()
    if prof: prof.enable()
    dec = mgf_decompressor.MGFDecompressor(model=model, inputFileName=base + '.vqms2',
                                           outputfilemgf=base + '_decompressed.mgf',
                                           batch_size=a.batch_size, quantizer=a.quantizer, **kw)
    if not hasattr(dec, 'compounded_idx_list'):  # decompressor leaves this unset when no .pkl exists
        dec.compounded_idx_list = []
    dec.DecompressAll(verbose=0)
    if prof: prof.disable()
    wall_d = time.perf_counter() - t
    results['decompress'] = {'wall_s': wall_d, 'n_spectra': n, 'stages': report('Decompression', wall_d, n),
                             'peak_rss_mb': peak_rss_mb()}
    if prof:
        prof.dump_stats(base + '_decompress.prof')
        print('\nTop decompression functions (cumulative):')
        pstats.Stats(prof).sort_stats('tottime').print_stats(15)

    print(f'\nPeak RSS: {peak_rss_mb():.0f} MB')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
