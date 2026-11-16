"""Print a sanitized description of the validation machine as JSON (no host, user, IP or absolute paths).

Run with the project virtual environment's interpreter so torch/CUDA details are included:
    .venv/bin/python scripts/validation_env.py
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*cmd: str) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0].strip() if text else None


def _os_release() -> dict[str, str]:
    data = {}
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value.strip().strip('"')
    return {"name": data.get("NAME", platform.system()), "version": data.get("VERSION_ID", platform.release())}


def _cpu() -> dict[str, object]:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                model = line.split(":", 1)[1].strip()
                break
    if model is None and sys.platform == "darwin":
        model = _run("sysctl", "-n", "machdep.cpu.brand_string")
    return {"model": model or platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
            "machine": platform.machine()}


def _memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"MemTotal:\s+(\d+) kB", meminfo.read_text())
        if match:
            return round(int(match.group(1)) / 1024 ** 2, 1)
    if sys.platform == "darwin":
        value = _run("sysctl", "-n", "hw.memsize")
        return round(int(value) / 1024 ** 3, 1) if value and value.isdigit() else None
    return None


def _filesystem() -> dict[str, object]:
    """Filesystem type of the checkout; WSL2 checkouts under /mnt/<drive> are slow and fsync-weak."""
    root = str(ROOT)
    fstype, best = None, ""
    mounts = Path("/proc/mounts")
    if mounts.is_file():
        for line in mounts.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and root.startswith(parts[1]) and len(parts[1]) > len(best):
                best, fstype = parts[1], parts[2]
    return {"type": fstype, "under_windows_drive": bool(re.match(r"^/mnt/[a-z]/", root))}


def _torch() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, object] = {"installed": True, "version": torch.__version__,
                               "threads": torch.get_num_threads(),
                               "cuda_available": torch.cuda.is_available(),
                               "cuda_version": torch.version.cuda,
                               "mps_available": bool(getattr(torch.backends, "mps", None)
                                                     and torch.backends.mps.is_available())}
    if torch.cuda.is_available():
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 1),
                         "capability": ".".join(map(str, torch.cuda.get_device_capability(i)))}
                        for i in range(torch.cuda.device_count())]
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info


def main() -> int:
    release = platform.release()
    git_sha = _run("git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD")
    dirty = _run("git", "-C", str(ROOT), "status", "--porcelain")
    report = {
        "os": _os_release(),
        "kernel": release,
        "wsl": "microsoft" in release.lower() or bool(os.environ.get("WSL_DISTRO_NAME")),
        "cpu": _cpu(),
        "memory_gb": _memory_gb(),
        "python": platform.python_version(),
        "compiler": _run("c++", "--version"),
        "cmake": _run("cmake", "--version"),
        "nvidia_driver": _run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "torch": _torch(),
        "checkout": {"commit": git_sha, "dirty": bool(dirty), "filesystem": _filesystem()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
