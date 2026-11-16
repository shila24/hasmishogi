"""Injectable clocks, suspend detection, disk guard, and isolated device probes."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


class Clock:
    wall = staticmethod(time.time)
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    @staticmethod
    def boot():
        return time.clock_gettime(time.CLOCK_BOOTTIME) if hasattr(time, "CLOCK_BOOTTIME") else time.monotonic()


class SuspendDetector:
    def __init__(self, clock, threshold):
        self.clock, self.threshold = clock, threshold
        self.previous = self.sample()

    def sample(self):
        return self.clock.wall(), self.clock.monotonic(), self.clock.boot()

    def check(self, marker):
        current = self.sample()
        wall, mono, boot = [a - b for a, b in zip(current, self.previous)]
        self.previous = current
        external = Path(marker).exists()
        if external:
            Path(marker).unlink(missing_ok=True)
        detected = external or wall > self.threshold or abs(wall - mono) > self.threshold or boot - mono > self.threshold
        return {"wall_gap": wall, "monotonic_gap": mono, "boot_gap": boot, "external": external} if detected else None


def free_gb(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 2**30


def probe(device, path, timeout):
    # Import torch exclusively in the disposable probe process.
    script = '''import json, shutil, sys
import torch
from hasami_learn.devices import resolve_device
device = resolve_device(sys.argv[1])
x = torch.ones((8, 8), device=device)
y = x @ x
if device.type == "cuda":
    torch.cuda.synchronize(device)
assert y.cpu()[0, 0].item() == 8
print(json.dumps({"ok": True, "device": str(device), "free_gb": shutil.disk_usage(sys.argv[2]).free / 2**30}))
'''
    try:
        result = subprocess.run([sys.executable, "-c", script, device, str(path)],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            return {"ok": False, "error": result.stderr[-2000:]}
        value = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(value, dict) or value.get("ok") is not True:
            raise ValueError("invalid probe response")
        return value
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
