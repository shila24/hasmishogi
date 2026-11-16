"""Standard-library configuration and durable NNUE control records."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path
import signal
import subprocess

from hasami_learn.autoloop.state import atomic_json, read_json, sha256, validate_schema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "hasami_learn/schemas/nnue-config.schema.json"
BOOTSTRAP_SHA256 = hashlib.sha256(b"bootstrap-v1").hexdigest()
GENERATION = dict(games=1000, seed=0, nodes=2000, depth=64, threads=8, max_threads=24,
                  hash_mb=16, max_hash_mb=1024, max_moves=320, opening_plies=8,
                  random_until=32, random_probability=.05, temperature=0., near_best=100.,
                  quiet_check=True, quiet_capture=True, byte_budget=30 * 2**30, first_id=0)
MATCH = dict(games=400, seed=5000, nodes=2000, time_ms=0, threads=8, hash_mb=16,
             max_moves=320, opening_plies=8, save_records=False)
TRAINING = dict(epochs=3, width=256, hidden1=32, hidden2=32, batch_size=1024,
                lr=.001, weight_decay=.01, scale=600., mix=.8, mu=0., seed=0,
                validation_fraction=.1, device="cuda", amp="off", threads=1,
                factorised=True, workers=0, shuffle_buffer=65536, feature_backend="auto",
                correlation_clip=3000., patience=2, min_delta=0., lr_schedule="none", schedule_epochs=10)


def validate_section(data, name):
    validate_schema(data, read_json(SCHEMA)["properties"][name])


def validate_record(data, name):
    validate_schema(data, read_json(SCHEMA)["$defs"][name])


def load_config(path):
    value = read_json(path)
    validate_schema(value, read_json(SCHEMA))
    config = {**value, "generation": {**GENERATION, **value["generation"]},
              "training": {**TRAINING, **value["training"]}, "match": {**MATCH, **value["match"]}}
    g, m = config["generation"], config["match"]
    if g["threads"] > g["max_threads"] or g["threads"] * g["hash_mb"] > g["max_hash_mb"]:
        raise ValueError("Generation resource cap exceeded")
    if g["opening_plies"] >= g["max_moves"] or g["random_until"] > g["max_moves"]:
        raise ValueError("Generation opening exceeds rule limit")
    if m["games"] % 2 or m["opening_plies"] >= m["max_moves"]:
        raise ValueError("Match requires full opening pairs within rule limit")
    if bool(m["nodes"]) == bool(m["time_ms"]):
        raise ValueError("Match requires exactly one of nodes and time_ms")
    training = config["training"]
    if training["shuffle_buffer"] < training["batch_size"]:
        raise ValueError("Shuffle buffer must hold at least one batch")
    if training["lr_schedule"] == "cosine" and training["epochs"] > training["schedule_epochs"]:
        raise ValueError("Cosine schedule horizon is shorter than training")
    if "replay" in config:
        replay = config["replay"]
        if replay["max_cycles"] * g["byte_budget"] > replay["data_byte_budget"]:
            raise ValueError("Replay plan exceeds the total data byte budget")
        if "distillation" in config:
            raise ValueError("Cumulative replay with distillation is not supported")
        if g["first_id"] + replay["max_cycles"]*g["games"] > 2**64:
            raise ValueError("Replay game IDs exceed uint64")
    return config


def identity(weights):
    return sha256(weights) if weights else BOOTSTRAP_SHA256


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "control.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Stop:
    """Forward direct signals to a child; retain autoloop's process group."""
    def __init__(self):
        self.requested = False
        self.child = None
        self.previous = {}

    def __enter__(self):
        def receive(signum, frame):
            self.requested = True
            if self.child is not None and self.child.poll() is None:
                try:
                    self.child.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.previous = {s: signal.signal(s, receive) for s in (signal.SIGTERM, signal.SIGINT)}
        return self

    def __exit__(self, *args):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)

    def run(self, command):
        if self.requested:
            raise InterruptedError("NNUE stage stopped")
        self.child = subprocess.Popen(command)
        try:
            code = self.child.wait()
        finally:
            self.child = None
        if self.requested:
            raise InterruptedError("NNUE stage stopped")
        if code:
            raise subprocess.CalledProcessError(code, command)
