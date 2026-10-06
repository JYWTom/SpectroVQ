"""Measure candidate CPU optimizations for SpectroVQ and their effect on the output.

Variants (per-spectrum latency of encoder + VQ(4) + decoder at a given batch size):
  fp32            : model as shipped
  fp32+ftz        : codebook denormals flushed to zero (codebook 0 contains ~19% denormals,
                    which make its distance matmul ~150x slower on x86)
  bf16            : fp32+ftz plus torch.autocast(cpu, bfloat16) around encoder/decoder (AMX/AVX512-BF16)
Accuracy is reported as the fraction of VQ codes identical to the fp32 baseline and the
max/mean abs difference of the decoded spectrum.
"""
import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import getModel  # noqa: E402
from bench_model_cpu import fake_batch, bench  # noqa: E402


def flush_codebook_denormals(model):
    tiny = torch.finfo(torch.float32).tiny
    n = 0
    for layer in model.quantization.layers:
        e = layer._codebook.embed
        mask = (e != 0) & (e.abs() < tiny)
        n += int(mask.sum())
        e.data[mask] = 0.0
    return n


def run(model, x, nq, bf16):
    ctx = torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else torch.autocast('cpu', enabled=False)
    with torch.inference_mode():
        with ctx:
            z = model.encoder(x, None)
        z = z.float()
        zq = z.view(z.shape[0], model.quantizationdim, -1)
        _, codes, _ = model.quantization.encodeWithLoss(zq, nq)
        emb = model.quantization.decode(codes)
        with ctx:
            y = model.decoder(emb, None)
        return codes, y.float()[:, :, :model.inputSize]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_batches', type=int, default=4)
    p.add_argument('--quantizer', type=int, default=4)
    p.add_argument('--json')
    a = p.parse_args()
    torch.set_num_threads(os.cpu_count())

    base = getModel.ApplyWeights(getModel.getModel(), a.weights).eval()
    ftz = copy.deepcopy(base)
    n_flushed = flush_codebook_denormals(ftz)
    print(f'flushed {n_flushed} denormal codebook entries')

    xs = [fake_batch(a.batch_size, seed=i) for i in range(a.n_batches)]
    ref = [run(base, x, a.quantizer, False) for x in xs]
    variants = {'fp32': (base, False), 'fp32+ftz': (ftz, False), 'bf16+ftz': (ftz, True)}
    out = {'batch_size': a.batch_size, 'quantizer': a.quantizer, 'flushed_denormals': n_flushed, 'variants': {}}
    for name, (model, bf16) in variants.items():
        t = bench(lambda: [run(model, x, a.quantizer, bf16) for x in xs], reps=2)
        ms = 1000 * t / (a.batch_size * a.n_batches)
        res = [run(model, x, a.quantizer, bf16) for x in xs]
        code_eq = sum((r[0] == c[0]).sum().item() for r, c in zip(res, ref)) / sum(c[0].numel() for c in ref)
        first_q_eq = sum((r[0][0] == c[0][0]).sum().item() for r, c in zip(res, ref)) / sum(c[0][0].numel() for c in ref)
        diff = torch.cat([(r[1] - c[1]).abs().flatten() for r, c in zip(res, ref)])
        row = dict(ms_per_spectrum=ms, code_agreement=code_eq, first_quantizer_agreement=first_q_eq,
                   decoded_max_abs_diff=diff.max().item(), decoded_mean_abs_diff=diff.mean().item())
        out['variants'][name] = row
        print(name, {k: round(v, 5) for k, v in row.items()})
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
