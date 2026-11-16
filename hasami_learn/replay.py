"""Fixed-capacity numpy ring buffer with npz persistence."""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from .games import get_spec
from .runlock import writer_path

FieldSpec = dict[str, tuple[tuple[int, ...], Any]]


def _total_ram_bytes() -> int | None:
    """Read physical RAM from Linux meminfo, or return None if unavailable."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields and fields[0] == "MemTotal:":
                if len(fields) != 3 or fields[2] != "kB":
                    return None
                total = int(fields[1]) * 1024
                return total if total > 0 else None
    except (OSError, ValueError, UnicodeError):
        return None
    return None


def memory_warning(capacity_nbytes: int, total_ram_bytes: int | None = None,
                   fraction: float = 0.5) -> str | None:
    """Warn when full replay storage exceeds a fraction of known physical RAM."""
    if total_ram_bytes is None:
        total_ram_bytes = _total_ram_bytes()
    if total_ram_bytes is None or total_ram_bytes <= 0:
        return None
    if capacity_nbytes <= fraction * total_ram_bytes:
        return None
    return (
        f"Warning: replay buffer at full capacity requires {capacity_nbytes} bytes "
        f"({capacity_nbytes / 2**30:.2f} GiB), exceeding {fraction:.0%} of physical RAM "
        f"({total_ram_bytes} bytes, {total_ram_bytes / 2**30:.2f} GiB)."
    )


def storage_dtype(name: str, dtype: Any) -> Any:
    if name in ("mask", "next_mask"):
        return np.uint8
    if name == "policy":
        return np.float16
    return dtype


def fields_for(spec, num_planes=None, kind="pv") -> FieldSpec:
    """Derive replay shapes; accept the legacy (num_planes, kind) call."""
    if isinstance(spec, int):
        kind = num_planes if isinstance(num_planes, str) else kind
        num_planes, spec = spec, get_spec()
    if num_planes is None:
        num_planes = spec.make_encoder("").num_planes
    planes = ((num_planes, *spec.board_shape), np.float32)
    mask = ((spec.action_size,), np.float32)
    fields = {"planes": planes, "mask": mask}
    if kind == "pv":
        fields.update(policy=mask, value=((), np.float32))
    elif kind == "q":
        fields.update(action=((), np.int64), reward=((), np.float32), next_planes=planes,
                      next_mask=mask, done=((), np.float32))
    else:
        raise ValueError(f"unknown replay kind {kind}")
    return fields


PV_FIELDS = fields_for(get_spec(), kind="pv")
Q_FIELDS = fields_for(get_spec(), kind="q")


class ReplayBuffer:
    def __init__(self, capacity: int, fields: FieldSpec, rng: np.random.Generator | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.fields = dict(fields)
        self.rng = rng or np.random.default_rng(0)
        # Masks are stored as uint8 and policies as float16 to keep large buffers in memory;
        # sample()/all() upcast to the declared dtype so learners never see the storage types.
        self._arrays = {name: np.zeros((self.capacity, *shape), dtype=storage_dtype(name, dtype)) for name, (shape, dtype) in self.fields.items()}
        self.size = 0
        self.position = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    @property
    def sample_nbytes(self) -> int:
        """Bytes per stored sample, excluding array and Python object overhead."""
        return sum(prod(array.shape[1:]) * array.dtype.itemsize for array in self._arrays.values())

    @property
    def capacity_nbytes(self) -> int:
        """Storage bytes at full capacity, independent of current occupancy."""
        return self.capacity * self.sample_nbytes

    def clear(self) -> None:
        self.size = 0
        self.position = 0

    def add(self, **sample: Any) -> None:
        self._check_keys(sample)
        for name, value in sample.items():
            self._arrays[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def add_batch(self, **batch: Any) -> None:
        self._check_keys(batch)
        count = len(next(iter(batch.values())))
        for index in range(count):
            self.add(**{name: value[index] for name, value in batch.items()})

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("replay buffer is empty")
        index = self.rng.integers(0, self.size, size=min(batch_size, self.size))
        return {name: array[index].astype(self.fields[name][1], copy=False) for name, array in self._arrays.items()}

    def all(self) -> dict[str, np.ndarray]:
        return {name: array[: self.size].astype(self.fields[name][1], copy=False) for name, array in self._arrays.items()}

    def nbytes(self) -> int:
        return int(sum(array.nbytes for array in self._arrays.values()))

    def save(self, path: str | Path) -> Path:
        target = writer_path(path)
        # numpy appends .npz when the caller omits the extension.
        archive = writer_path(target if str(target).endswith(".npz") else str(target) + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, __size=np.int64(self.size), __position=np.int64(self.position), __total=np.int64(self.total_added), **self._arrays)
        return target

    def load(self, path: str | Path) -> None:
        with np.load(Path(path)) as data:
            for name in self.fields:
                if name not in data:
                    raise ValueError(f"replay file lacks field {name}")
                array = data[name]
                if array.shape[1:] != self._arrays[name].shape[1:]:
                    raise ValueError(f"replay field {name} has shape {array.shape[1:]}, expected {self._arrays[name].shape[1:]}")
                count = min(array.shape[0], self.capacity)
                self._arrays[name][:count] = array[:count]
            self.size = int(min(int(data["__size"]), self.capacity))
            self.position = int(data["__position"]) % self.capacity
            self.total_added = int(data["__total"])

    def _check_keys(self, sample: dict[str, Any]) -> None:
        if set(sample) != set(self.fields):
            raise ValueError(f"replay sample keys {sorted(sample)} do not match fields {sorted(self.fields)}")
