"""Full-history records for game adapters, with optional engine format hooks."""

import json
from pathlib import Path


def _plain(value):
    return value.to_dict() if hasattr(value, "to_dict") else value


def _normalised_joint(step):
    return {int(player): int(action) for player, action in step.items()}


def history_matches(recorded, moves) -> bool:
    """True when a recorded joint history is exactly the engine's current history.

    Engines that store scalar moves are compared step by step: a recorded step must contain a
    single actor whose action equals the scalar move (an undo followed by a different move changes
    the scalar history while lengths stay equal).
    """
    if recorded is None or len(recorded) != len(moves):
        return False
    for step, move in zip(recorded, moves):
        joint = _normalised_joint(step)
        if isinstance(move, dict):
            if joint != _normalised_joint(move):
                return False
        elif len(joint) != 1 or next(iter(joint.values())) != int(move):
            return False
    return True


def _engine_joint_actions(game):
    """Joint actions recorded by an engine whose ``moves`` hold per-step player->action dicts."""
    moves = list(getattr(game, "moves", []) or [])
    if moves and all(isinstance(move, dict) for move in moves):
        return [{int(player): int(action) for player, action in move.items()} for move in moves]
    return None


def adapter_history(spec, game):
    """Joint actions in adapter action space: the adapter's ``history_actions`` hook when it has one
    (engines whose scalar moves differ from action ids), else the engine's own joint moves, else None."""
    hook = getattr(spec, "history_actions", None)
    if hook is not None:
        return [_normalised_joint(step) for step in hook(game)]
    return _engine_joint_actions(game)


def engine_history_consistent(spec, game, joint):
    """False when a scalar-move engine's history cannot be the given joint actions (an adapter that
    translates action ids without a ``history_actions`` hook); True when there is nothing to compare."""
    if getattr(spec, "history_actions", None) is not None or not hasattr(game, "moves"):
        return True
    try:
        return history_matches(joint, list(game.moves))
    except (TypeError, ValueError):
        return True


_TRANSLATION_HINT = ("adapters whose step() translates action ids must implement history_actions(game)")
_SEED_HINT = ("a game with chance events replays only from its creation seed: expose game.seed, save with "
              "the trajectory, or continue a loaded record")


def has_chance(spec) -> bool:
    return bool(getattr(spec, "has_chance", True))


def _replays_to(game, moves) -> bool:
    """False when the engine exposes comparable scalar ``moves`` that differ from ``moves`` after replay."""
    if not hasattr(game, "moves"):
        return True
    try:
        return [int(m) for m in game.moves] == [int(m) for m in moves]
    except (TypeError, ValueError):
        return True


def save_record(spec, game, path, trajectory=None):
    """Save a game; v2 metadata (players, seed, joint actions, rewards) is derived from the
    trajectory, else from the live engine, else from a loaded record that has not been continued."""
    if hasattr(spec, "save_record"):
        return spec.save_record(game, path)
    live = game.game if isinstance(game, RecordedGame) else game
    data = {"version": 1, "game_id": spec.game_id, "ruleset_id": spec.ruleset_id,
            "rules": spec.rules_dict(live), "initial": _plain(live.initial),
            "moves": list(live.moves), "winner": live.outcome.winner,
            "reason": live.outcome.reason, "num_players": spec.num_players,
            "seed": getattr(live, "seed", None)}
    joint = rewards = None
    truncated = bool(getattr(live, "truncated", False))
    record = game.record if isinstance(game, RecordedGame) else None
    if record is not None and data["seed"] is None:
        data["seed"] = record.get("seed")  # engines may accept a seed without exposing it publicly
    if trajectory is not None:
        joint, rewards, truncated = trajectory.joint_actions, trajectory.rewards, trajectory.truncated
        data["seed"] = trajectory.seed
    if data["seed"] is None and has_chance(spec):
        raise ValueError("cannot save a replayable record: " + _SEED_HINT)
    if trajectory is None:
        joint = adapter_history(spec, live)
        recorded = record.get("joint_actions", []) if record is not None and record.get("version") == 2 else None
        if joint is None and recorded is not None:
            # Scalar-move engine: reuse the recorded joint actions only when they match the live
            # history exactly; otherwise the scalar moves are the truth and a v1 record is written
            # (single-actor steps replay through spec.actors on load).
            joint = recorded if history_matches(recorded, list(live.moves)) else None
        if joint is not None:
            unchanged = recorded is not None and history_matches(recorded, joint)
            if unchanged:
                # Runner-imposed caps live outside the engine: keep the recorded flag for an unchanged history.
                truncated = bool(record.get("truncated", truncated))
                rewards = record.get("rewards") if record.get("rewards") is not None else (
                    list(spec.terminal_values(live)) if live.outcome.terminal else None)
            else:
                rewards = list(spec.terminal_values(live)) if live.outcome.terminal else None
    if joint is not None:
        data.update(version=2, joint_actions=joint, rewards=rewards, truncated=truncated)
    Path(path).write_text(json.dumps(data, allow_nan=False) + "\n", encoding="utf-8")
    return Path(path)


def load_record(spec, path, backend="python"):
    if hasattr(spec, "load_record"):
        return spec.load_record(path, backend)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (data.get("version") not in (1, 2) or data.get("game_id") != spec.game_id
            or data.get("ruleset_id") != spec.ruleset_id):
        raise ValueError("record game or version mismatch")
    if data.get("num_players", 2) != spec.num_players:
        raise ValueError("record player count mismatch")
    if data.get("seed") is None and has_chance(spec):
        raise ValueError("record has no seed; " + _SEED_HINT)
    if data["version"] == 1 and getattr(spec, "history_actions", None) is not None:
        raise ValueError("version-1 records hold engine moves, which this adapter does not use as action ids; "
                         "save the game again to produce a version-2 record")
    game = spec.create_game(backend, data["rules"], data["initial"], seed=data.get("seed"))
    try:
        if data["version"] == 2:
            for joint in data["joint_actions"]:
                actions = {int(player): action for player, action in joint.items()}
                if set(actions) != set(spec.actors(game)) or any(
                        type(action) is not int or action not in spec.legal_actions(game, player)
                        for player, action in actions.items()):
                    raise ValueError("illegal recorded joint action")
                spec.step(game, actions)
            if game.outcome.terminal and data.get("rewards") is not None and list(spec.terminal_values(game)) != list(data["rewards"]):
                raise ValueError("record rewards do not match full history")
            if not engine_history_consistent(spec, game, data["joint_actions"]):
                raise ValueError("recorded actions replay to a different engine history; " + _TRANSLATION_HINT)
        else:
            for action in data["moves"]:
                actors = spec.actors(game)
                if len(actors) != 1 or type(action) is not int or action not in spec.legal_actions(game, actors[0]):
                    raise ValueError("illegal recorded action")
                spec.step(game, {actors[0]: action})
            if not _replays_to(game, data["moves"]):
                raise ValueError("version-1 moves replay to a different engine history; " + _TRANSLATION_HINT)
        if game.outcome.winner != data["winner"] or game.outcome.reason != data["reason"]:
            raise ValueError("record outcome does not match full history")
        # Version-1 records of seeded games keep their metadata too: an engine may accept a seed
        # without exposing game.seed, and dataset extraction / resaving must reproduce the deal.
        if data["version"] == 2 or data.get("seed") is not None:
            return RecordedGame(game, data)
        return game
    except BaseException:
        game.close()
        raise


class RecordedGame:
    """Keep replay metadata alongside an engine without modifying its state."""

    def __init__(self, game, record):
        self.game = game
        self.record = record

    def clone(self):
        """Clone the engine and carry the replay metadata (seed, truncation) with it."""
        return RecordedGame(self.game.clone(), dict(self.record))

    def __getattr__(self, name):
        return getattr(self.game, name)

    def __setattr__(self, name, value):
        # Adapters may assign engine fields (game.scores = ...); keep all game state in the engine.
        if name in ("game", "record"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.game, name, value)

    def __delattr__(self, name):
        if name in ("game", "record"):
            object.__delattr__(self, name)
        else:
            delattr(self.game, name)
