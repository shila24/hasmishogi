"""Device selection for CPU, Apple Silicon (MPS) and NVIDIA (CUDA)."""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    if mps_available():
        devices.append("mps")
    return devices


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto|cpu|mps|cuda|cuda:N`` to a torch.device, failing loudly."""
    if isinstance(spec, torch.device):
        return spec
    text = (spec or "auto").strip().lower()
    if text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device '{text}' requested but CUDA is not available; available: {available_devices()}")
        if ":" in text:
            index = int(text.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"cuda:{index} does not exist; available: {available_devices()}")
        return torch.device(text)
    if text == "mps":
        if not mps_available():
            raise RuntimeError(f"device 'mps' requested but Metal is not available; available: {available_devices()}")
        return torch.device("mps")
    if text == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device spec '{spec}'; use auto, cpu, mps, cuda or cuda:N")


# Rows per batched self-play forward when device.inference_batch is null, from end-to-end collection sweeps
# (scripts/benchmark_collection.py, hasami_learn/devices.py). CPU: PyTorch's convolution
# kernel on the M2 falls off a cliff at batch >= 16. MPS: rounds carry ~17-60 pending rows, so larger fixed shapes
# only add padding. CUDA: provisional until the WSL2 sweep.
DEFAULT_INFERENCE_BATCH = {"cpu": 8, "mps": 32, "cuda": 64}


def resolve_inference_batch(configured: int | None, device: torch.device) -> int:
    """The configured batch size, or this device type's default; never another device's limit."""
    if configured is not None:
        if type(configured) is not int or configured < 1:
            raise ValueError("inference_batch must be a positive integer or None")
        return configured
    return DEFAULT_INFERENCE_BATCH.get(device.type, DEFAULT_INFERENCE_BATCH["cpu"])


def amp_dtype(device: torch.device, amp: str = "auto") -> torch.dtype | None:
    """Mixed precision policy: bf16 on CUDA when enabled, float32 elsewhere."""
    if amp == "off":
        return None
    if device.type == "cuda":
        return torch.bfloat16
    return None


def autocast_context(device: torch.device, amp: str = "auto"):
    dtype = amp_dtype(device, amp)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def seed_everything(seed: int, deterministic: bool = False) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch signatures
            torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def configure_threads(threads: int | None) -> None:
    if threads:
        torch.set_num_threads(int(threads))


def to_device(batch: dict[str, Any], device: torch.device, non_blocking: bool = False) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    pin = device.type == "cuda"
    for key, value in batch.items():
        tensor = torch.as_tensor(value) if not isinstance(value, torch.Tensor) else value
        if pin and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        out[key] = tensor.to(device, non_blocking=non_blocking and pin)
    return out


def is_wsl() -> bool:
    """True when running inside Windows Subsystem for Linux (CUDA via the Windows driver)."""
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def device_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "available": available_devices(),
        "auto_selects": str(resolve_device("auto")),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_available(),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        "num_threads": torch.get_num_threads(),
        "wsl": is_wsl(),
    }
    if is_wsl() and not torch.cuda.is_available():
        report["hint"] = ("WSL2 without CUDA: install the NVIDIA Windows driver with WSL support and a CUDA build of torch "
                          "(see docs/CUDA_WSL2_ja.md)")
    if torch.cuda.is_available():
        report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return report


def module_device(parameters: Iterable[torch.Tensor]) -> torch.device:
    for parameter in parameters:
        return parameter.device
    return torch.device("cpu")
