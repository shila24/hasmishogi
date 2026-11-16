"""Deterministic streaming NNUE training on hlnn1 files, with atomic CPU-safe resume."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import glob
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import torch
import torch.nn.functional as F

from hasami_learn.devices import resolve_device, seed_everything, autocast_context
from .data import TRUNCATED, atomic_write, digest_file, read_shard, sync_directory
from .export import export_weights, load_weights
from .loader import feature_batches, resolve_backend, validation_game
from .model import NNUE


def targets(records, *, scale, mix, mu, device):
    scores = torch.tensor([r.score for r in records], dtype=torch.float32, device=device)
    results = torch.tensor([(r.result+1)/2 for r in records], dtype=torch.float32, device=device)
    truncated = torch.tensor([bool(r.flags & TRUNCATED) for r in records], device=device)
    score_value = torch.sigmoid(scores / scale)
    target = torch.where(truncated, score_value, mix * score_value + (1-mix) * results)
    teacher = torch.tensor([r.teacher for r in records], dtype=torch.float32, device=device)
    return torch.where(torch.isfinite(teacher), (1-mu)*target + mu*(teacher.nan_to_num()+1)/2, target)


def batches(paths, *, validation, seed, fraction, batch_size, epoch):
    rng = random.Random(seed + epoch)
    buffer = []
    for path in paths:
        for record in read_shard(path):
            if validation_game(record.game_id, seed, fraction) != validation:
                continue
            buffer.append(record)
            if len(buffer) >= batch_size * 4:
                if not validation:
                    rng.shuffle(buffer)
                for start in range(0, len(buffer), batch_size):
                    yield buffer[start:start+batch_size]
                buffer = []
    if not validation:
        rng.shuffle(buffer)
    for start in range(0, len(buffer), batch_size):
        yield buffer[start:start+batch_size]


def save_checkpoint(path, state):
    """Flush all tensors and optimizer state before publishing one resumable file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Correlation:
    """Streaming CPU float64 moments; MPS never receives float64 tensors."""
    def __init__(self):
        self.count = 0
        self.mean_x = self.mean_y = self.xx = self.yy = self.xy = 0.0

    def add(self, x, y):
        size = x.numel()
        if not size:
            return
        mean_x, mean_y = x.mean().item(), y.mean().item()
        dx, dy = mean_x-self.mean_x, mean_y-self.mean_y
        total = self.count+size
        weight = self.count*size/total
        centered_x, centered_y = x-mean_x, y-mean_y
        self.xx += centered_x.square().sum().item() + dx*dx*weight
        self.yy += centered_y.square().sum().item() + dy*dy*weight
        self.xy += (centered_x*centered_y).sum().item() + dx*dy*weight
        self.mean_x += dx*size/total
        self.mean_y += dy*size/total
        self.count = total

    def value(self):
        denominator = math.sqrt(self.xx*self.yy)
        return max(-1., min(1., self.xy/denominator)) if denominator else None


def batch_targets(batch, *, scale, mix, mu):
    """Mate scores saturate sigmoid; clipping is for metrics only."""
    score_value = torch.sigmoid(batch["score"] / scale)
    truncated = batch["flags"].to(torch.int64).bitwise_and(TRUNCATED).bool()
    target = torch.where(truncated, score_value,
                         mix*score_value + (1-mix)*(batch["result"]+1)/2)
    teacher = batch["teacher"]
    return torch.where(torch.isfinite(teacher), (1-mu)*target+mu*(teacher.nan_to_num()+1)/2, target)


def run_partition(model, optimizer, entries, *, validation, epoch, config, device):
    model.train(not validation)
    total_loss, count = 0., 0
    clipped, sigmoid, result = Correlation(), Correlation(), Correlation()
    started = time.monotonic()
    loader = feature_batches(entries, validation=validation, epoch=epoch,
                             seed=config["seed"], fraction=config["validation_fraction"],
                             batch_size=config["batch_size"], workers=config["workers"],
                             shuffle_buffer=config["shuffle_buffer"], backend=config["feature_backend"],
                             pin_memory=device.type == "cuda")
    with torch.no_grad() if validation else nullcontext():
        for host in loader:
            batch = {key: value.to(device, non_blocking=device.type == "cuda")
                     for key, value in host.items() if key != "game_id"}
            target = batch_targets(batch, scale=config["scale"], mix=config["mix"], mu=config["mu"])
            with autocast_context(device, config["amp"]):
                prediction = model.forward_training(batch["indices"].long())
                loss = F.binary_cross_entropy_with_logits(prediction.float()/config["scale"], target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite NNUE loss")
            if not validation:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = len(target)
            total_loss += loss.item()*size
            count += size
            if validation:
                x = prediction.detach().cpu().double()
                y = host["score"].double()
                limit = config["correlation_clip"]
                clipped.add(x.clamp(-limit, limit), y.clamp(-limit, limit))
                sigmoid.add(torch.sigmoid(x/config["scale"]), torch.sigmoid(y/config["scale"]))
                completed = host["flags"].long().bitwise_and(TRUNCATED) == 0
                result.add(x.clamp(-limit, limit)[completed], host["result"].double()[completed])
    if not count:
        raise ValueError("Game-id split has an empty partition; supply more games or change seed")
    prefix = "validation" if validation else "train"
    row = {prefix+"_loss": total_loss/count, prefix+"_records": count}
    if validation:
        row.update(teacher_correlation=clipped.value(),
                   teacher_correlation_basis=f"scores and predictions clipped to +/-{config['correlation_clip']:g} cp",
                   sigmoid_correlation=sigmoid.value(), result_correlation=result.value(),
                   result_correlation_records=result.count)
    seconds = time.monotonic()-started
    return row, {prefix+"_seconds": seconds, prefix+"_positions_per_second": count/seconds}


def train(data, out, *, data_list=None, init=None, resume=None, epochs=3, width=256, hidden1=32, hidden2=32,
          batch_size=1024, lr=0.001, weight_decay=0.01, scale=600.0, mix=0.8, mu=0.0,
          seed=0, validation_fraction=0.1, device="cpu", amp="off", threads=1,
          factorised=True, workers=0, shuffle_buffer=65536, feature_backend="auto",
          correlation_clip=3000., patience=2, min_delta=0., lr_schedule="none", schedule_epochs=10):
    """Stream immutable shards; select/export best loss and resume completed epochs.

    The schedule horizon is explicit and part of identity, so extending --epochs
    on resume never silently changes learning rates. Wall times live separately
    from deterministic metrics. Incomplete epochs are replayed after a stop.
    """
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or not 1 <= batch_size <= 65536
            or type(threads) is not int or threads < 1 or not 0 < validation_fraction < 1
            or not all(math.isfinite(v) for v in (lr, weight_decay, scale, mix, mu, validation_fraction,
                                                  correlation_clip, min_delta))
            or lr <= 0 or weight_decay < 0 or scale <= 0 or not 0 <= mix <= 1 or not 0 <= mu <= 1
            or amp not in ("off", "auto") or (init and resume) or type(factorised) is not bool
            or type(workers) is not int or not 0 <= workers <= 16
            or type(shuffle_buffer) is not int or not batch_size <= shuffle_buffer <= 1048576
            or correlation_clip <= 0 or type(patience) is not int or patience < 0 or min_delta < 0
            or lr_schedule not in ("none", "step", "cosine")
            or type(schedule_epochs) is not int or schedule_epochs < 1
            or (lr_schedule == "cosine" and epochs > schedule_epochs)):
        raise ValueError("Invalid NNUE training configuration")
    if data_list:
        if data:
            raise ValueError("Use data patterns or a frozen data list, not both")
        from .window import window_paths
        # Verified replay filenames must never be reinterpreted as patterns.
        paths = window_paths(data_list)
    else:
        matches = [glob.glob(str(pattern)) for pattern in (data or [])]
        if any(not group for group in matches):
            raise ValueError("A requested hlnn1 shard pattern matched no files")
        paths = [p for group in matches for p in group]
    paths = sorted({str(Path(p).resolve()) for p in paths})
    if not paths:
        raise ValueError("No hlnn1 data shards")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    backend = resolve_backend(feature_backend)
    torch.set_num_threads(threads)
    seed_everything(seed, deterministic=True)
    if init:
        initial = load_weights(init)
        width, hidden1, hidden2 = initial.width, initial.hidden1, initial.hidden2
    model = NNUE(width, hidden1, hidden2, factorised=factorised)
    if init:
        state = model.state_dict()
        state.update(initial.state_dict())
        model.load_state_dict(state)
        del initial, state
    model.to(device)
    config = {"seed": seed, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
              "scale": scale, "mix": mix, "mu": mu, "validation_fraction": validation_fraction,
              "amp": amp, "architecture": model.architecture, "device": str(device), "threads": threads,
              "factorised": factorised, "workers": workers, "shuffle_buffer": shuffle_buffer,
              "feature_backend": backend, "forward": "vectorised-v1", "correlation_clip": correlation_clip,
              "patience": patience, "min_delta": min_delta, "lr_schedule": lr_schedule,
              "schedule_epochs": schedule_epochs, "mate_target": "sigmoid saturation, unclipped scores",
              "data": [[p, digest_file(p), digest_file(p+".json")] for p in paths]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                   gamma=0.5 if lr_schedule == "step" else 1.0)
    start, history, timings = 0, [], []
    best_loss, patience_loss, best_epoch, bad_epochs = math.inf, math.inf, 0, 0
    best_model, stopped = None, False
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "hlnn-checkpoint-v2" or checkpoint.get("config") != config:
            raise ValueError("NNUE resume configuration/data mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, history, timings = checkpoint["epoch"], checkpoint["metrics"], checkpoint["timings"]
        best_model, best_loss, best_epoch = checkpoint["best_model"], checkpoint["best_loss"], checkpoint["best_epoch"]
        patience_loss, bad_epochs, stopped = checkpoint["patience_loss"], checkpoint["bad_epochs"], checkpoint["stopped"]
        if start > epochs:
            raise ValueError("Resume epoch exceeds requested total")
        del checkpoint
    elif (output / "checkpoint.pt").exists():
        raise ValueError("Output already contains a checkpoint; use --resume")
    for epoch in range(start, epochs):
        if stopped:
            break
        row = {"epoch": epoch+1, "lr": optimizer.param_groups[0]["lr"]}
        timing = {"epoch": epoch+1, "device": str(device), "feature_backend": backend}
        for validation in (False, True):
            metrics, performance = run_partition(model, optimizer, config["data"], validation=validation,
                                                  epoch=epoch, config=config, device=device)
            row.update(metrics)
            timing.update(performance)
        if row["validation_loss"] < best_loss:
            best_loss, best_epoch = row["validation_loss"], epoch+1
            best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if row["validation_loss"] < patience_loss-min_delta:
            patience_loss, bad_epochs = row["validation_loss"], 0
        else:
            bad_epochs += 1
        stopped = patience > 0 and bad_epochs >= patience
        row.update(best_epoch=best_epoch, best_validation_loss=best_loss, early_stopped=stopped)
        scheduler.step()
        history.append(row)
        timings.append(timing)
        # The checkpoint is authoritative; publication can be retried without training.
        save_checkpoint(output / "checkpoint.pt", {"format": "hlnn-checkpoint-v2", "config": config,
            "epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "metrics": history, "timings": timings,
            "best_model": best_model, "best_loss": best_loss, "best_epoch": best_epoch,
            "patience_loss": patience_loss, "bad_epochs": bad_epochs, "stopped": stopped})
        atomic_write(output / "metrics.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in history).encode())
        atomic_write(output / "throughput.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in timings).encode())
        print(json.dumps({**row, **timing}, sort_keys=True), flush=True)
    atomic_write(output / "metrics.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in history).encode())
    atomic_write(output / "throughput.jsonl", "".join(json.dumps(r, sort_keys=True)+"\n" for r in timings).encode())
    model.load_state_dict(best_model)
    export_weights(model, output / "weights.hlnnw1")
    metadata = {**config, "selected_epoch": best_epoch, "best_validation_loss": best_loss,
                "completed_epochs": len(history), "early_stopped": stopped}
    atomic_write(output / "training.json", (json.dumps(metadata, sort_keys=True)+"\n").encode())
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", nargs="+")
    source.add_argument("--data-list")
    parser.add_argument("--out", required=True)
    parser.add_argument("--init")
    parser.add_argument("--resume")
    for name, default in (("epochs", 3), ("width", 256), ("hidden1", 32), ("hidden2", 32),
                          ("batch-size", 1024), ("seed", 0), ("threads", 1), ("workers", 0),
                          ("shuffle-buffer", 65536), ("patience", 2), ("schedule-epochs", 10)):
        parser.add_argument("--"+name, type=int, default=default)
    for name, default in (("lr", 0.001), ("weight-decay", 0.01), ("scale", 600.0),
                          ("mix", 0.8), ("mu", 0.0), ("validation-fraction", 0.1),
                          ("correlation-clip", 3000.), ("min-delta", 0.)):
        parser.add_argument("--"+name, type=float, default=default)
    parser.add_argument("--factorised", choices=("true", "false"), default="true")
    parser.add_argument("--feature-backend", choices=("auto", "native", "python"), default="auto")
    parser.add_argument("--lr-schedule", choices=("none", "step", "cosine"), default="none")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", choices=("off", "auto"), default="off")
    args = vars(parser.parse_args(argv))
    args["factorised"] = args["factorised"] == "true"
    train(**args)


if __name__ == "__main__":
    main()
