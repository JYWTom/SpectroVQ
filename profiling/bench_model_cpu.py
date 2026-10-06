"""Model-only CPU micro-benchmarks for SpectroVQ (no file I/O).

1. Per-module breakdown of encoder / VQ / decoder via torch.profiler.
2. Latency vs batch size, thread count and number of quantizers.

The model's compute does not depend on spectrum content, so a random sparse
input of the right shape (1 x 13500 bins) gives representative timings.
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import getModel  # noqa: E402


def fake_batch(bs, size=13500, peaks=250, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(bs, 1, size)
    idx = torch.randint(0, size, (bs, peaks), generator=g)
    x.scatter_(2, idx.unsqueeze(1), torch.rand(bs, 1, peaks, generator=g))
    return x.sqrt()


def bench(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def stages(model, x, nq):
    with torch.inference_mode():
        z = model.encoder(x, None)
        zq = z.view(z.shape[0], model.quantizationdim, -1)
        emb, codes, _ = model.quantization.encodeWithLoss(zq, nq)
        return z, zq, emb, codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--json', default=None)
    p.add_argument('--batch_sizes', default='1,8,32,128')
    p.add_argument('--threads', default='1,2,4')
    p.add_argument('--quantizers', default='1,4,8,12')
    a = p.parse_args()

    model = getModel.ApplyWeights(getModel.getModel(), a.weights).eval()
    enc_p = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    dec_p = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    vq_p = sum(b.numel() for b in model.quantization.buffers()) / 1e6
    print(f'params: encoder {enc_p:.1f}M, decoder {dec_p:.1f}M, codebooks {vq_p:.1f}M')
    res = {'params_M': {'encoder': enc_p, 'decoder': dec_p, 'codebooks': vq_p}}

    # ---- 1. Per-module breakdown with torch.profiler (bs=32, 4 quantizers, all threads)
    torch.set_num_threads(os.cpu_count())
    x = fake_batch(32)
    stages(model, x, 4)
    # Time each top-level layer of the encoder/decoder trunks (bs=32) with forward hooks
    layer_times = {}

    def add_hooks(prefix, seq):
        for i, layer in enumerate(seq):
            name = f'{prefix}[{i:02d}] {type(layer).__name__}'
            layer.register_forward_pre_hook(lambda m, inp, n=name: layer_times.setdefault(n, []).append(-time.perf_counter()))
            layer.register_forward_hook(lambda m, inp, out, n=name: layer_times[n].__setitem__(-1, layer_times[n][-1] + time.perf_counter()))

    add_hooks('enc', model.encoder.model)
    add_hooks('dec', model.decoder.model)
    with torch.inference_mode():
        for _ in range(3):
            z = model.encoder(x, None)
            model.decoder(z, None)
    layer_ms = {k: 1000 * min(v) / 32 for k, v in layer_times.items()}
    for k, v in layer_ms.items():
        print(f'  {k:40s} {v:7.2f} ms/spectrum')
    res['layer_ms_per_spectrum_bs32'] = layer_ms

    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function('encoder'):
            z = model.encoder(x, None)
        zq = z.view(z.shape[0], model.quantizationdim, -1)
        with record_function('vq.encode_q4'):
            _, codes, _ = model.quantization.encodeWithLoss(zq, 4)
        with record_function('decoder'):
            model.reconstructIndices(codes)
    table = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=15)
    print(table)
    res['op_table'] = table

    # ---- 2. Scaling sweeps
    sweep = []
    for th in [int(t) for t in a.threads.split(',')]:
        torch.set_num_threads(th)
        for bs in [int(b) for b in a.batch_sizes.split(',')]:
            x = fake_batch(bs)
            with torch.inference_mode():
                z = model.encoder(x, None)
                zq = z.view(z.shape[0], model.quantizationdim, -1)
                _, codes4, _ = model.quantization.encodeWithLoss(zq, 4)
                t_enc = bench(lambda: model.encoder(x, None))
                t_vq = bench(lambda: model.quantization.encodeWithLoss(zq, 4))
                t_dec = bench(lambda: model.reconstructIndices(codes4))
            row = dict(threads=th, batch=bs, encoder_ms_per_spec=1000 * t_enc / bs,
                       vq4_ms_per_spec=1000 * t_vq / bs, decoder_ms_per_spec=1000 * t_dec / bs)
            sweep.append(row)
            print(row)
    res['sweep'] = sweep

    torch.set_num_threads(os.cpu_count())
    x = fake_batch(32)
    qsweep = []
    with torch.inference_mode():
        z = model.encoder(x, None)
        zq = z.view(z.shape[0], model.quantizationdim, -1)
        for nq in [int(q) for q in a.quantizers.split(',')]:
            _, codes, _ = model.quantization.encodeWithLoss(zq, nq)
            t_vq = bench(lambda: model.quantization.encodeWithLoss(zq, nq))
            t_dec = bench(lambda: model.reconstructIndices(codes))
            row = dict(quantizers=nq, vq_encode_ms_per_spec=1000 * t_vq / 32,
                       decode_ms_per_spec=1000 * t_dec / 32,
                       payload_bytes_per_spec=nq * model.quantizedlen * 10 / 8)
            qsweep.append(row)
            print(row)
    res['quantizer_sweep'] = qsweep

    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
