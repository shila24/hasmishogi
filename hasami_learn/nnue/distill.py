"""Add family-Z values in batches, replaying the generator's complete histories."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from .control import atomic_json, lock, read_json, sha256, validate_record
from .data import read_shard, write_shard
from .generate import validate_manifest


def distill(data, checkpoint, out, *, device="auto", batch_size=256):
    import numpy as np
    import torch
    from hasami_learn.devices import resolve_device
    from hasami_learn.games import get_spec
    from hasami_learn.manifest import load_checkpoint
    from shogi import create_game
    from shogi.model import Rules

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    data, out = Path(data), Path(out)
    manifest = validate_manifest(data)
    rules = Rules(max_moves=manifest["settings"]["max_moves"])
    resolved = resolve_device(device)
    loaded = load_checkpoint(checkpoint, resolved, expect_game_id="shogi", expect_rules=asdict(rules))
    model = loaded.network.eval()
    if loaded.manifest.algorithm != "alphazero" or loaded.manifest.network["head"] != "policy_value":
        raise ValueError("Distillation requires a family-Z AlphaZero policy/value checkpoint")
    spec = get_spec(loaded.manifest.game_id, loaded.manifest.game_params)
    encoder = spec.make_encoder(loaded.manifest.encoding_id, **(loaded.manifest.encoder_params or {}))
    provenance = dict(format="hlnn-distill-v1", source_manifest_sha256=sha256(data / "manifest.json"),
                      teacher_sha256=loaded.manifest.weights_sha256, device=str(resolved), batch_size=batch_size,
                      history="complete generator move history", rules=asdict(rules))
    with lock(out):
        if (out / "settings.json").exists() and read_json(out / "settings.json") != provenance:
            raise ValueError("Distillation resume provenance mismatch")
        atomic_json(out / "settings.json", provenance)
        shards = []
        for row in manifest["games"]:
            if row["evicted"]:
                continue
            source = data / row["shard"]
            destination = out / row["shard"]
            sidecar = Path(str(destination)+".json")
            if sidecar.exists():
                for _ in read_shard(destination):
                    pass
            else:
                destination.unlink(missing_ok=True)
                game = create_game("cpp", rules)
                try:
                    def annotated():
                        pending, planes = [], []
                        def evaluate():
                            with torch.inference_mode():
                                _, values = model(torch.as_tensor(np.stack(planes), device=resolved))
                                if values is None:
                                    raise ValueError("Teacher lacks value head")
                                values = values.float().cpu().tolist()
                            if len(values) != len(pending):
                                raise ValueError("Teacher batch shape mismatch")
                            return [replace(r, teacher=float(v)) for r, v in zip(pending, values)]
                        for record in read_shard(source):
                            while game.ply < record.ply:
                                game.play(row["moves"][game.ply])
                            if game.position != record.position:
                                raise ValueError("Distillation replay disagrees with data")
                            pending.append(record); planes.append(encoder.encode(game))
                            if len(pending) == batch_size:
                                yield from evaluate()
                                pending.clear(); planes.clear()
                        if pending:
                            yield from evaluate()
                    write_shard(destination, annotated())
                finally:
                    game.close()
            shards.append(dict(path=destination.name, sha256=sha256(destination), sidecar_sha256=sha256(sidecar)))
        result = {**provenance, "status": "complete", "shards": shards}
        validate_record(result, "distillation_manifest")
        atomic_json(out / "manifest.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    distill(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
