"""Device selection for non-Trainer code (inference, GUI), delegated to Lightning.

Training gets its device from ``L.Trainer(accelerator=...)``. Scripts that run the
model directly use the same Lightning accelerator logic through Fabric, so
"auto" resolves identically everywhere (CUDA, then MPS, then CPU).
"""

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators import CUDAAccelerator, MPSAccelerator

ACCELERATORS = ("auto", "cpu", "cuda", "mps")


def resolve_device(accelerator: str = "auto") -> torch.device:
    """Return the torch.device Lightning would use for ``accelerator``."""
    return Fabric(accelerator=accelerator, devices=1).device


def available_accelerators() -> list[str]:
    """Accelerators usable on this machine, according to Lightning."""
    usable = ["auto"]
    if CUDAAccelerator.is_available():
        usable.append("cuda")
    if MPSAccelerator.is_available():
        usable.append("mps")
    usable.append("cpu")
    return usable
