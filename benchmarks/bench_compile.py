"""Eager vs torch.compile for SpectroVQ: steady-state time and compile time.

Configs:
  eager    no compile
  encdec   encoder + decoder compiled in place (what --compile does)
  whole    torch.compile(model), including the quantizer (expect graph breaks/recompiles)

Measured per config:
  train    forward + backward at --train_batch (default 32), model.train(), use_vq=True
  infer    encode + decodeLatent at --infer_batch (default 128), eval, no_grad

Run from the repository root, e.g.  uv run python -m benchmarks.bench_compile
Add TORCH_LOGS=graph_breaks,recompiles to see why a config is slow.
"""

import argparse
import copy
import time

import torch

from utils.compile import COMPILE_MODES, compile_model
from utils.device import ACCELERATORS, resolve_device
from utils.model.getModel import getModel

LENGTH = 13500


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device, steps, warmup):
    """Return (first call seconds, steady-state ms per call)."""
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    first = time.perf_counter() - t0
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    sync(device)
    return first, (time.perf_counter() - t0) / steps * 1e3


def build(base, config, mode, device):
    model = copy.deepcopy(base).to(device)
    if config == "encdec":
        compile_model(model, mode, device, force=True)
    return model


def bench(base, config, mode, device, args):
    rows = []
    # train
    model = build(base, config, mode, device).train()
    run = torch.compile(model, mode=None if mode == "default" else mode) if config == "whole" else model
    x = torch.rand(args.train_batch, 1, LENGTH, device=device) ** 4

    def train_step():
        out, qloss, _ = run(x, use_vq=True)
        (out.mean() + qloss.mean()).backward()
        model.zero_grad(set_to_none=True)

    rows.append(("train", *timed(train_step, device, args.steps, args.warmup)))

    # infer
    model = build(base, config, mode, device).eval()
    x = torch.rand(args.infer_batch, 1, LENGTH, device=device) ** 4
    if config == "whole":
        model.encode = torch.compile(model.encode)
        model.decodeLatent = torch.compile(model.decodeLatent)

    @torch.no_grad()
    def infer_step():
        emb, _, _, _ = model.encode(
            x, returnCodebookIndices=True, returnAll=True, numquantizer=4
        )
        model.decodeLatent(emb)

    rows.append(("infer", *timed(infer_step, device, args.steps, args.warmup)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--accelerator", default="auto", choices=ACCELERATORS)
    p.add_argument("--mode", default="default", choices=COMPILE_MODES[1:])
    p.add_argument("--configs", nargs="+", default=["eager", "encdec", "whole"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--train_batch", type=int, default=32)
    p.add_argument("--infer_batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device(args.accelerator)
    torch.manual_seed(args.seed)
    base = getModel()
    print(f"device={device} torch={torch.__version__} mode={args.mode}")
    print(f"{'config':8} {'phase':6} {'first call s':>13} {'steady ms':>10} {'speedup':>8}")
    eager_ms = {}
    for config in args.configs:
        for phase, first, ms in bench(base, config, args.mode, device, args):
            if config == "eager":
                eager_ms[phase] = ms
            speed = f"{eager_ms[phase] / ms:.2f}x" if phase in eager_ms else "-"
            print(f"{config:8} {phase:6} {first:13.1f} {ms:10.1f} {speed:>8}")


if __name__ == "__main__":
    main()
