"""Single-writer, capped self-play archives with an append-only recovery journal.

Caps cover committed game files, excluding the journal and in-flight temporary
files. An oversized game is journaled and immediately evicted, so its returned
path may no longer exist. Readers ignore torn journal lines; recovery appends
missing records for fully committed archives without rewriting old entries.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator
from zipfile import BadZipFile

import numpy as np

from .games import default_rules, get_spec
from .games.base import as_v2, encode_player
from hasami.model import require_int
from .selfplay import Trajectory
from .runlock import cleanup_temps, plain_directory, run_lock, writer_path


def _sync_directory(path: Path) -> None:
    """Persist directory entries after rename/unlink where directory fsync is supported."""
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class SelfplayStore:
    def __init__(self, root, max_games: int = 10000, max_bytes: int = 30 * 2**30, *, spec=None, game_params: dict | None = None) -> None:
        for name, cap in (("max_games", max_games), ("max_bytes", max_bytes)):
            require_int(cap, name)
            if cap < 0:
                raise ValueError(f"{name} must be nonnegative")
        self.spec = as_v2(spec or get_spec())
        # Variant identity: archives written under another parameterisation or rule set of the same
        # game id must never be replayed here (their targets and histories would not match).
        self.game_params = dict(game_params or {})
        self.ruleset_id = getattr(self.spec, "ruleset_id", "")
        self.rules = default_rules(self.spec)
        self.root = writer_path(root)
        self.games_dir = writer_path(self.root / "games")
        self.index_path = writer_path(self.root / "index.jsonl")
        self.max_games, self.max_bytes = max_games, max_bytes
        self.truncation_value = 0.0
        self._entries: list[dict] = []
        self._seq = 0
        with run_lock(self.root, "store.lock"):
            self.games_dir.mkdir(parents=True, exist_ok=True)
            if not plain_directory(self.games_dir):
                raise ValueError("self-play games directory must not contain symlinks")
            _sync_directory(self.root)
            self._recover()
            self._revision = self._disk_revision()

    def _disk_revision(self):
        return tuple((p.stat().st_ino, p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
                     for p in (self.games_dir, self.index_path))

    def _append_index(self, entry: dict) -> None:
        # Separate a torn final line from the new record, preserving all old bytes.
        with self.index_path.open("ab+") as output:
            if output.tell():
                output.seek(-1, os.SEEK_END)
                if output.read(1) != b"\n":
                    output.write(b"\n")
            output.write((json.dumps(entry, allow_nan=False) + "\n").encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        _sync_directory(self.root)

    def _check_variant(self, metadata: dict) -> None:
        """Reject archives from another game id, parameterisation, or rule set."""
        if metadata.get("game_id", "hasami") != self.spec.game_id:
            raise ValueError("Self-play archive game does not match selected spec")
        if metadata.get("num_players", 2) != self.spec.num_players:
            raise ValueError("Self-play archive player count does not match selected spec")
        saved_params = metadata.get("game_params")
        if saved_params is None:
            saved_params = {}  # archives written before variant identity existed are default-parameter archives
        if dict(saved_params) != self.game_params:
            raise ValueError(f"Self-play archive game_params {saved_params} do not match the store's {self.game_params}")
        saved_rules = metadata.get("rules")
        if saved_rules is not None and saved_rules != self.rules:
            raise ValueError("Self-play archive rules do not match the selected game rules")
        saved_ruleset = metadata.get("ruleset_id")
        if saved_ruleset is not None and self.ruleset_id and saved_ruleset != self.ruleset_id:
            raise ValueError("Self-play archive ruleset does not match the selected game ruleset")

    def _entry(self, path: Path, metadata: dict) -> dict:
        self._check_variant(metadata)
        return {"game_id": self.spec.game_id, "num_players": self.spec.num_players, "file": f"games/{path.name}", "iteration": metadata["iteration"],
                "plies": metadata["plies"], "bytes": path.stat().st_size,
                "winner": metadata["winner"], "reason": metadata["reason"],
                "created_at": metadata["created_at"]}

    def _recover(self) -> None:
        indexed = set()
        if self.index_path.exists():
            with self.index_path.open("rb") as source:
                for line in source:
                    try:
                        entry = json.loads(line)
                        filename = entry["file"]
                        stem = Path(filename).stem
                        self._seq = max(self._seq, int(stem.split("_")[1]) + 1)
                    except (ValueError, KeyError, TypeError, IndexError):
                        continue
                    indexed.add(filename)
        recovered = []
        for path in self.games_dir.glob("*.npz"):
            if path.is_symlink():
                raise ValueError(f"Self-play archive must not be a symlink: {path}")
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"].item()))
                self._seq = max(self._seq, int(path.stem.split("_")[1]) + 1)
                entry = self._entry(path, metadata)
            except (OSError, ValueError, KeyError, IndexError, BadZipFile):
                # A committed archive must be readable; corruption is not a training sample.
                raise ValueError(f"Corrupt self-play archive: {path}") from None
            recovered.append(entry)
        self._entries = sorted(recovered, key=lambda entry: int(Path(entry["file"]).stem.split("_")[1]))
        for entry in self._entries:
            if entry["file"] not in indexed:
                self._append_index(entry)
        # Only our staging namespace is cleaned, under the single-writer contract.
        cleanup_temps(self.games_dir, files=(r"\.selfplay-[A-Za-z0-9_-]+\.tmp",))
        self._evict()

    def _evict(self) -> None:
        total = sum(entry["bytes"] for entry in self._entries)
        removed = False
        while self._entries and (len(self._entries) > self.max_games or total > self.max_bytes):
            entry = self._entries[0]
            (self.root / entry["file"]).unlink(missing_ok=True)
            self._entries.pop(0)
            total -= entry["bytes"]
            removed = True
        if removed:
            _sync_directory(self.games_dir)

    def add(self, trajectory: Trajectory, iteration: int) -> Path | None:
        """Commit a trajectory; move prefixes support opponent-only decision recording.

        A trajectory without any learner decision (for example an opponent game capped before
        the learner moved) carries no training targets and no move history, so it is skipped
        and ``None`` is returned instead of raising.
        """
        self.games_dir = writer_path(self.games_dir)
        self.index_path = writer_path(self.index_path)
        with run_lock(self.root, "store.lock"):
            if self._disk_revision() != self._revision:
                self._recover()
            try:
                return self._add(trajectory, iteration)
            finally:
                self._revision = self._disk_revision()

    def _add(self, trajectory: Trajectory, iteration: int) -> Path | None:
        if trajectory.game_id != self.spec.game_id or trajectory.num_players != self.spec.num_players:
            raise ValueError("Trajectory game does not match store")
        require_int(iteration, "iteration")
        count = len(trajectory)
        if any(len(items) != count for items in (trajectory.players, trajectory.policies, trajectory.extras)):
            raise ValueError("Trajectory decision fields have inconsistent lengths")
        if count == 0:
            return None
        if len(trajectory.rewards) != self.spec.num_players or any(
                type(p) is not int or not 0 <= p < self.spec.num_players for p in trajectory.players):
            raise ValueError("Invalid trajectory player or reward count")
        if trajectory.joint_actions:
            joints = trajectory.joint_actions
            if any(not isinstance(joint, dict) or any(
                    type(p) is not int or not 0 <= p < self.spec.num_players
                    for p in joint) for joint in joints):
                raise ValueError("Invalid joint history seats")
            if len(joints) != trajectory.plies:
                raise ValueError("A stored trajectory requires full move history (joint steps)")
            moves = [action for joint in joints for action in joint.values()]
            decision_plies = [extra["decision_ply"] for extra in trajectory.extras]
            keys = [(ply, trajectory.players[i]) for i, ply in enumerate(decision_plies)]
            if (len(set(keys)) != count or decision_plies != sorted(decision_plies) or any(
                    type(ply) is not int or not 0 <= ply < len(joints) or
                    joints[ply].get(trajectory.players[i]) != trajectory.actions[i]
                    for i, ply in enumerate(decision_plies))):
                raise ValueError("Decision seats do not match joint history")
        else:
            history = trajectory.extras[0].get("game_moves")
            moves = list(history) if history is not None else list(trajectory.actions)
            decision_plies = ([extra["decision_ply"] for extra in trajectory.extras]
                             if history is not None else list(range(count)))
            if len(moves) != trajectory.plies:
                raise ValueError("A stored trajectory requires full move history")
            if (decision_plies != sorted(set(decision_plies)) or any(
                    ply < 0 or ply >= len(moves) or moves[ply] != trajectory.actions[i]
                    for i, ply in enumerate(decision_plies))):
                raise ValueError("Decision plies do not match the move history")
        if any(type(action) is not int or not 0 <= action < self.spec.action_size for action in moves):
            raise ValueError("Invalid move action ID")
        indices, probs, offsets = [], [], [0]
        for policy in trajectory.policies:
            policy = np.asarray(policy, dtype=np.float32)
            if (policy.shape != (self.spec.action_size,) or not np.isfinite(policy).all() or
                    (policy < 0).any() or not np.isclose(policy.sum(), 1, atol=1e-5)):
                raise ValueError("Stored trajectories require normalized finite policy targets")
            active = np.flatnonzero(policy)
            indices.extend(active)
            probs.extend(policy[active])
            offsets.append(len(indices))
        metadata = {"game_id": self.spec.game_id, "num_players": self.spec.num_players,
                    "seed": trajectory.seed, "joint_actions": trajectory.joint_actions, "game_params": dict(self.game_params), "ruleset_id": self.ruleset_id, "rules": dict(self.rules),
                    "iteration": iteration, "plies": trajectory.plies,
                    "winner": trajectory.winner, "reason": trajectory.reason,
                    "truncated": trajectory.truncated, "rewards": trajectory.rewards,
                    "draw_value": trajectory.draw_value, "draw_reasons": trajectory.draw_reasons,
                    "truncation_value": self.truncation_value, "created_at": time.time()}
        arrays = dict(moves=np.asarray(moves, dtype=np.int16 if self.spec.action_size <= np.iinfo(np.int16).max + 1 else np.int64),
                      players=np.asarray(trajectory.players, dtype=np.int64),
                      decision_plies=np.asarray(decision_plies, dtype=np.int64),
                      winner=np.asarray(trajectory.winner), reason=np.asarray(trajectory.reason),
                      truncated=np.asarray(trajectory.truncated),
                      policy_index=np.asarray(indices, dtype=np.int16 if self.spec.action_size <= np.iinfo(np.int16).max + 1 else np.int64),
                      policy_prob=np.asarray(probs, dtype=np.float16),
                      policy_offsets=np.asarray(offsets, dtype=np.int64),
                      root_value=np.asarray([extra.get("root_value", 0.0) for extra in trajectory.extras],
                                            dtype=np.float16),
                      metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
        path = writer_path(self.games_dir / f"{iteration:06d}_{self._seq:012d}.npz")
        self._seq += 1
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.games_dir, prefix=".selfplay-", suffix=".tmp",
                                             delete=False) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, **arrays)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            _sync_directory(self.games_dir)
            entry = self._entry(path, metadata)
            self._entries.append(entry)
            try:
                self._append_index(entry)
            finally:
                self._evict()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def stats(self) -> dict[str, int]:
        return {"games": len(self._entries), "plies": sum(e["plies"] for e in self._entries),
                "bytes": sum(e["bytes"] for e in self._entries)}

    def iter_games(self, limit: int | None = None, newest_first: bool = True) -> Iterator[dict]:
        """Yield metadata and decoded numpy arrays, normalizing float16 policy rounding."""
        if limit is not None:
            require_int(limit, "limit")
            if limit < 0:
                raise ValueError("limit must be nonnegative")
        entries = list(reversed(self._entries)) if newest_first else list(self._entries)
        for entry in entries[:limit]:
            with np.load(self.root / entry["file"], allow_pickle=False) as archive:
                game = {key: archive[key] for key in archive.files if key != "metadata"}
                game.update(json.loads(str(archive["metadata"].item())))
            self._check_variant(game)
            game.setdefault("game_id", "hasami")
            game.setdefault("num_players", 2)
            count = len(game["players"])
            policy = np.zeros((count, self.spec.action_size), dtype=np.float32)
            offsets = game["policy_offsets"]
            for i in range(count):
                start, end = offsets[i:i + 2]
                policy[i, game["policy_index"][start:end]] = game["policy_prob"][start:end]
                total = policy[i].sum()
                if not np.isfinite(total) or total <= 0:
                    raise ValueError("Invalid sparse policy target")
                policy[i] /= total
            game["policy"] = policy
            trajectory = Trajectory(actions=game["moves"].tolist(), players=game["players"].tolist(),
                                    rewards=list(game["rewards"]), num_players=game.get("num_players", 2), truncated=game["truncated"],
                                    winner=game["winner"], reason=game["reason"],
                                    draw_value=game["draw_value"], draw_reasons=tuple(game["draw_reasons"]))
            game["value"] = np.asarray([trajectory.value_for(i, game["truncation_value"])
                                        for i in range(count)], dtype=np.float32)
            yield game

    def to_dataset(self, encoder, backend: str = "python") -> dict[str, np.ndarray]:
        """Replay full histories to regenerate planes and masks for recorded decisions."""
        data = {"planes": [], "mask": [], "policy": [], "value": []}
        for saved in self.iter_games():
            game = self.spec.create_game(backend, seed=saved.get("seed"))
            try:
                joints = saved.get("joint_actions")
                steps = joints if joints else [{-1: int(a)} for a in saved["moves"]]
                decisions = {}
                for i, ply in enumerate(saved["decision_plies"]):
                    decisions.setdefault(int(ply), []).append(i)
                for ply, joint in enumerate(steps):
                    actions = {int(p): int(a) for p, a in joint.items()}
                    if -1 in actions:
                        actions = {self.spec.actors(game)[0]: actions[-1]}
                    for i in decisions.get(ply, []):
                        player = int(saved["players"][i])
                        # Copy on return (reused mask buffers) and before encoding, which may query masks.
                        mask = np.array(self.spec.legal_mask(game, player), copy=True)
                        policy = saved["policy"][i]
                        if player not in actions or np.any(policy[mask == 0]):
                            raise ValueError("Stored player or policy does not match replayed history")
                        # Retained until final assembly: copy, since encoders may return a reused buffer.
                        data["planes"].append(np.array(encode_player(encoder, game, player), copy=True))
                        data["mask"].append(mask)
                        data["policy"].append(policy)
                        data["value"].append(saved["value"][i])
                    self.spec.step(game, actions)
                if game.outcome.terminal and not np.allclose(self.spec.terminal_values(game), saved["rewards"]):
                    raise ValueError("Stored rewards do not match replayed history")
            finally:
                game.close()
        shapes = {"planes": (encoder.num_planes, *self.spec.board_shape), "mask": (self.spec.action_size,),
                  "policy": (self.spec.action_size,), "value": ()}
        return {key: np.asarray(values, dtype=np.float32).reshape((-1, *shapes[key]))
                for key, values in data.items()}
