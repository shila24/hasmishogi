"""Validated game records as dense policy/value imitation datasets.

Unfinished records have zero value targets; their outcomes remain ongoing.
Metadata is JSON stored as a Unicode scalar, never a pickled Python object.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
from typing import Any

import numpy as np

from .games import get_spec
from .games.base import as_v2, encode_player
from .selfplay import run_table_game, choose_agent
from hasami.agents import SearchLimits
from hasami.model import GameLike
from .games.records import _SEED_HINT, _TRANSLATION_HINT, adapter_history, has_chance, history_matches, load_record, save_record
from .games import Encoder
from .selfplay import augment as augment_samples

ARRAY_FIELDS = ("planes", "mask", "policy", "value")
Dataset = dict[str, Any]


def iter_record_samples(
    game: GameLike, encoder: Encoder, augment: bool = False, *, spec=None,
) -> Iterator[tuple[np.ndarray, np.ndarray, int, float]]:
    """Replay the full recorded history, yielding targets before each move."""
    spec = as_v2(spec or get_spec())
    values = spec.terminal_values(game) if game.outcome.terminal else [0.0] * spec.num_players
    record = getattr(game, "record", {})
    seed = record.get("seed") if record.get("seed") is not None else getattr(game, "seed", None)
    if seed is None and has_chance(spec):
        raise ValueError("cannot replay the game for extraction: " + _SEED_HINT)
    work = spec.create_game("python", spec.rules_dict(game),
                            game.initial.to_dict() if hasattr(game.initial, "to_dict") else game.initial, seed=seed)
    try:
        steps = _history_steps(game, spec)
        for step in steps:
            actions = ({int(p): a for p, a in step.items()} if isinstance(step, dict)
                       else {spec.actors(work)[0]: step})
            for player, action in actions.items():
                # Yield independent arrays: consumers may collect samples while the encoder reuses a buffer.
                planes = np.array(encode_player(encoder, work, player), copy=True)
                mask = np.array(spec.legal_mask(work, player), copy=True)
                value = values[player]
                yield planes, mask, action, value
                if augment:
                    for flipped, flipped_mask, _, flipped_action in augment_samples(spec, planes, mask, None, action):
                        yield flipped, flipped_mask, int(flipped_action), value
            spec.step(work, actions)
        if not _same_engine_history(work, game):
            raise ValueError("replayed history differs from the game's engine history; " + _TRANSLATION_HINT)
    finally:
        work.close()


def _same_engine_history(replayed, game) -> bool:
    """False when both engines expose comparable ``moves`` and they differ (an adapter that translates
    action ids without a history_actions hook); True when there is nothing to compare."""
    try:
        return not (hasattr(replayed, "moves") and hasattr(game, "moves")) or list(replayed.moves) == list(game.moves)
    except (TypeError, ValueError):
        return True


def _history_steps(game, spec=None):
    """The complete decision history of a game: the adapter's history_actions hook when it has one,
    else the live engine's joint moves when it records them, else the loaded record's joint actions
    while the history is unchanged, else the scalar moves (adapter action ids by contract)."""
    if spec is not None and getattr(spec, "history_actions", None) is not None:
        return adapter_history(spec, game)
    moves = list(getattr(game, "moves", []) or [])
    if moves and all(isinstance(move, dict) for move in moves):
        return moves
    record = getattr(game, "record", {}) or {}
    recorded = record.get("joint_actions")
    if recorded is not None and history_matches(recorded, moves):
        return recorded
    # The scalar moves are the current truth (a continued or edited record); single-actor steps
    # replay through spec.actors during extraction.
    return moves


def _decision_count(game, spec=None):
    return sum(len(step) if isinstance(step, dict) else 1 for step in _history_steps(game, spec))


def build_dataset(
    record_paths: Iterable[str | Path], encoder: Encoder,
    backend: str = "python", augment: bool = True, *, spec=None, game_params: dict | None = None, encoder_params: dict | None = None,
) -> Dataset:
    """Validate records and allocate dense arrays once after counting their plies."""
    spec = as_v2(spec or get_spec())
    paths = [Path(path) for path in record_paths]
    lengths = []
    wins = [0] * (spec.num_players + 1)
    ongoing = 0
    for path in paths:
        game = load_record(spec, path, backend)
        try:
            lengths.append(_decision_count(game, spec))
            if game.outcome.terminal:
                wins[game.outcome.winner if 0 <= game.outcome.winner < spec.num_players else spec.num_players] += 1
            else:
                ongoing += 1
        finally:
            game.close()
    count = sum(lengths) * (1 + len(spec.symmetries()) if augment else 1)
    data = {
        "planes": np.empty((count, encoder.num_planes, *spec.board_shape), np.float32),
        "mask": np.empty((count, spec.action_size), np.float32),
        "policy": np.zeros((count, spec.action_size), np.float32),
        "value": np.empty(count, np.float32),
        "meta": {"game_id": spec.game_id, "num_players": spec.num_players, "game_params": dict(game_params or {}), "board_shape": list(spec.board_shape),
                 "action_size": spec.action_size, "samples": count, "games": len(paths), "plies": sum(lengths),
                 "sources": [str(path) for path in paths], "wins": wins,
                 "ongoing": ongoing, "augment": augment,
                 "group_size": (1 + len(spec.symmetries())) if augment else 1,
                 "encoding_id": encoder.encoding_id, "num_planes": encoder.num_planes,
                 "encoder_params": dict(encoder_params or {})},
    }
    index = 0
    for path, length in zip(paths, lengths):
        game = load_record(spec, path, backend)
        try:
            if _decision_count(game, spec) != length:
                raise ValueError(f"record changed while building dataset: {path}")
            for planes, mask, action, value in iter_record_samples(game, encoder, augment, spec=spec):
                data["planes"][index] = planes
                data["mask"][index] = mask
                data["policy"][index, action] = 1.0
                data["value"][index] = value
                index += 1
        finally:
            game.close()
    return data


def _identity_params(meta: dict, key: str) -> dict:
    """Legacy datasets without ``key`` were built with default parameters; malformed values are rejected."""
    value = meta.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"dataset metadata {key} must be an object")
    return dict(value)


def validate_dataset(data: Dataset, spec=None, game_params: dict | None = None, encoder_params: dict | None = None) -> None:
    """Reject malformed arrays and labels outside the supplied legal mask."""
    if set(data) != {*ARRAY_FIELDS, "meta"} or not isinstance(data["meta"], dict):
        raise ValueError("dataset requires planes, mask, policy, value and JSON metadata")
    game_id = data["meta"].get("game_id", "hasami")
    spec = spec or get_spec(game_id, data["meta"].get("game_params"))
    if game_id != spec.game_id:
        raise ValueError("dataset game does not match selected spec")
    if data["meta"].get("num_players", 2) != spec.num_players:
        raise ValueError("dataset player count does not match selected spec")
    saved_params = _identity_params(data["meta"], "game_params")
    if game_params is not None and saved_params != dict(game_params):
        # Same game id but a different parameterisation (rule variant): masks and outcome labels would not match.
        raise ValueError(f"dataset game_params {saved_params} do not match the configured game_params {dict(game_params)}")
    saved_encoder = _identity_params(data["meta"], "encoder_params")
    if encoder_params is not None and saved_encoder != dict(encoder_params):
        raise ValueError(f"dataset encoder_params {saved_encoder} do not match the configured encoder_params {dict(encoder_params)}")
    planes = data["planes"]
    if not isinstance(planes, np.ndarray) or planes.ndim != 4 or planes.shape[1] < 1 or planes.shape[2:] != tuple(spec.board_shape):
        raise ValueError("dataset planes must have shape (N,C,H,W)")
    count = len(planes)
    shapes = {"planes": planes.shape, "mask": (count, spec.action_size),
              "policy": (count, spec.action_size), "value": (count,)}
    for key, shape in shapes.items():
        array = data[key]
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"dataset {key} must be float32 with shape {shape}")
    # Bound temporary validation arrays even for large dense datasets.
    for start in range(0, count, 1024):
        batch = {key: data[key][start:start + 1024] for key in ARRAY_FIELDS}
        if any(not np.isfinite(array).all() for array in batch.values()):
            raise ValueError("dataset contains nonfinite values")
        mask, policy = batch["mask"], batch["policy"]
        if not np.isin(mask, (0, 1)).all() or not (mask.sum(1) > 0).all():
            raise ValueError("dataset mask must be binary with legal actions in every row")
        if (not np.isin(policy, (0, 1)).all() or not (policy.sum(1) == 1).all()
                or (policy[mask == 0] != 0).any()):
            raise ValueError("dataset policy must be one-hot over legal actions")
        if (np.abs(batch["value"]) > 1).any():
            raise ValueError("dataset value must be in [-1,1]")


def save_dataset(path: str | Path, data: Dataset, *, spec=None) -> Path:
    """Save arrays and JSON metadata without enabling pickle."""
    validate_dataset(data, spec=spec)
    metadata = np.asarray(json.dumps(data["meta"], allow_nan=False))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        np.savez_compressed(stream, **{key: data[key] for key in ARRAY_FIELDS}, meta=metadata)
    return target


def load_dataset(path: str | Path, *, spec=None, game_params: dict | None = None, encoder_params: dict | None = None) -> Dataset:
    """Load only numeric arrays and JSON metadata, then validate training labels."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {*ARRAY_FIELDS, "meta"}:
            raise ValueError("unexpected dataset fields")
        data = {key: archive[key] for key in ARRAY_FIELDS}
        data["meta"] = json.loads(str(archive["meta"].item()))
    validate_dataset(data, spec=spec, game_params=game_params, encoder_params=encoder_params)
    return data


def generate_games(
    out_dir: str | Path, games: int, generator: str = "",
    backend: str = "python", max_plies: int | None = None, seed: int = 0,
    random_opening_plies: int = 4, *, spec=None,
) -> list[Path]:
    """Play seeded baselines; a ply cap leaves the saved game ongoing."""
    spec = as_v2(spec or get_spec())
    generator = generator or getattr(spec, "default_teacher", "random")
    max_plies = spec.default_max_plies if max_plies is None else max_plies
    if type(games) is not int or games < 0:
        raise ValueError("games must be a nonnegative integer")
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    if type(random_opening_plies) is not int or random_opening_plies < 0:
        raise ValueError("random_opening_plies must be a nonnegative integer")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for index in range(games):
        players = [spec.baseline_agent(generator, seed=seed + spec.num_players * index + side)
                   for side in range(spec.num_players)]
        def actor(game, planes, mask, ply, player):
            if ply < random_opening_plies:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                agent, limits = players[player]
                action = choose_agent(agent, game, limits or SearchLimits(), None, player).action
            return action, None, {}
        path = target / f"game_{index:06d}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite game record: {path}")
        run_table_game(dict.fromkeys(range(spec.num_players), actor), spec.make_encoder(""),
                       spec=spec, backend=backend, max_plies=max_plies, seed=seed + index,
                       on_finish=lambda game, trajectory: save_record(spec, game, path, trajectory=trajectory))
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="hasami")
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--generator", default="")
    parser.add_argument("--out-games", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # Accept the separator forwarded by argparse.REMAINDER in the parent CLI.
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = parser.parse_args(argv)
    if args.generate < 0:
        parser.error("--generate must be nonnegative")
    if args.generate and args.out_games is None:
        parser.error("--generate requires --out-games")
    spec = get_spec(args.game)
    paths = sorted(args.records_dir.glob("*.json")) if args.records_dir else []
    if args.generate:
        paths.extend(generate_games(args.out_games, args.generate, args.generator, args.backend, seed=args.seed, spec=spec))
    if not paths:
        parser.error("provide records with --records-dir or --generate")
    data = build_dataset(paths, spec.make_encoder(""), args.backend, not args.no_augment, spec=spec,
                         game_params=getattr(args, "game_params", None) or {},
                         encoder_params=getattr(args, "encoder_params", None) or {})
    save_dataset(args.out, data)
    print(f"saved {len(data['value'])} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
