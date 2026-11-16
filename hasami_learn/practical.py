"""Level-conditioned proxies and practical advice, with no training dependencies."""

from __future__ import annotations

import json
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import numpy as np
import torch

from hasami.agents import Decision, SearchLimits, check_cancel
from .checkpoint_bundle import engine_dependency, sha256_file
from .devices import resolve_device, resolve_inference_batch
from .games import get_spec
from .games.base import encode_player, require_search_game, terminal_rewards
from .manifest import check_manifest_matches_spec, load_checkpoint
from .search.mcts import MCTS, MctsConfig, batched_evaluator

TRAIN_STYLES = ("qre", "temperature", "checkpoint")
HELDOUT_STYLES = ("alphabeta",)


@dataclass(frozen=True)
class LevelInfo:
    level: int
    elo: float | None
    elo_ci: tuple[float, float] | None
    styles: tuple[str, ...]


@dataclass(frozen=True)
class Advice:
    action: int
    move: int
    text: str
    practical_score: float
    engine_value: float
    value_loss: float
    difficulty: float | None
    naturalness: float
    guard_fallback: bool = False


def validate_artifact(data, name):
    schema = json.loads((Path(__file__).parent / "schemas" / f"practical-{name}.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(data)
    # JSON Schema's number type alone does not reject IEEE nonfinite values.
    json.dumps(data, allow_nan=False)


def validate_levels(data):
    """Validate calibration and return a mapping with the documented style defaults."""
    validate_artifact(data, "levels")
    if [row["level"] for row in data["levels"]] != list(range(1, data["k"] + 1)):
        raise ValueError("levels must be contiguous and agree with k")
    train = data.get("train_styles", list(TRAIN_STYLES))
    heldout = data.get("heldout_styles", list(HELDOUT_STYLES))
    if not train or len(set(train)) != len(train) or set(train) & set(heldout):
        raise ValueError("training and held-out styles must be nonempty/disjoint")
    previous = {}
    for row in data["levels"]:
        for style, entry in row["styles"].items():
            if style not in TRAIN_STYLES + HELDOUT_STYLES:
                raise ValueError(f"unknown proxy style: {style}")
            if not entry["available"]:
                continue
            elo, ci = entry["elo"], entry["ci"]
            if elo is None or ci is None or not ci[0] <= elo <= ci[1] or entry["games"] < 1:
                raise ValueError("available styles need measured Elo, interval and games")
            if style in previous and elo <= previous[style]:
                raise ValueError(f"non-increasing measured Elo for {style}")
            previous[style] = elo
    return {**data, "train_styles": list(train), "heldout_styles": list(heldout)}


def load_levels(path):
    return validate_levels(json.loads(Path(path).read_text(encoding="utf-8")))


def mover(spec, game):
    actors = spec.actors(game)
    if len(actors) != 1:
        raise ValueError("practical play requires one active player and a nonterminal position")
    return actors[0]


def normalize(weights):
    if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("invalid probability weights")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("probability mass must be positive")
    return {int(a): float(v / total) for a, v in sorted(weights.items())}


def top_distribution(probabilities, count=8):
    actions = sorted(probabilities, key=lambda a: (-probabilities[a], a))[:count]
    return normalize({a: probabilities[a] for a in actions})


def prior_distribution(logits, legal, temperature=1.0, nucleus=1.0):
    if not legal or not math.isfinite(temperature) or temperature <= 0 or not 0 < nucleus <= 1:
        raise ValueError("invalid policy distribution arguments")
    z = np.asarray([logits[a] for a in legal], dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    result = normalize(dict(zip(legal, z)))
    if nucleus < 1:
        selected, mass = {}, 0.0
        for action in sorted(result, key=lambda a: (-result[a], a)):
            selected[action] = result[action]
            mass += result[action]
            if mass >= nucleus:
                break
        result = normalize(selected)
    return result


def sample_distribution(distribution, rng):
    actions = sorted(distribution)
    return int(rng.choice(actions, p=[distribution[a] for a in actions]))


@dataclass
class Evaluation:
    prior: dict[int, float]
    value: float
    practical: np.ndarray | None = None
    slip: np.ndarray | None = None


class NetworkEvaluator:
    """Batch nonterminal observations; terminal values are always adapter-owned."""

    def __init__(self, network, encoder, spec, device="cpu", batch_size=64):
        self.network, self.encoder, self.spec = network, encoder, spec
        self.device = resolve_device(device)
        self.batch_size = resolve_inference_batch(batch_size, self.device)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.network.eval()

    def evaluate(self, games, cancel=None, *, players=None):
        if players is not None and (len(players) != len(games) or any(
                type(p) is not int or not 0 <= p < self.spec.num_players for p in players)):
            raise ValueError("one valid observing player per game is required")
        rows = []
        for offset in range(0, len(games), self.batch_size):
            check_cancel(cancel)
            chunk = games[offset:offset + self.batch_size]
            seats = ([mover(self.spec, g) for g in chunk] if players is None
                     else players[offset:offset + self.batch_size])
            planes = np.stack([np.array(encode_player(self.encoder, g, p), copy=True)
                               for g, p in zip(chunk, seats)])
            x = torch.as_tensor(planes, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                if getattr(self.network, "aux_heads", {}):
                    head, value, aux = self.network.forward_aux(x)
                else:
                    head, value = self.network(x)
                    aux = {}
                if value is None:
                    raise ValueError("practical play needs a policy_value engine")
                logits, values = head.float().cpu().numpy(), value.float().cpu().numpy()
                ps = torch.sigmoid(aux["practical"]).cpu().numpy() if "practical" in aux else None
                slips = torch.sigmoid(aux["slip"]).cpu().numpy() if "slip" in aux else None
            for index, (game, player) in enumerate(zip(chunk, seats)):
                legal = self.spec.legal_actions(game, player)
                rows.append(Evaluation(prior_distribution(logits[index], legal) if legal else {},
                                       float(values[index]), None if ps is None else ps[index],
                                       None if slips is None else slips[index]))
        check_cancel(cancel)
        return rows

    def one_ply_many(self, games, cancel=None):
        """Return every legal action's value in its root player's perspective."""
        require_search_game(self.spec)
        outputs = [{} for _ in games]
        with ExitStack() as stack:
            pending = []
            for index, game in enumerate(games):
                player = mover(self.spec, game)
                for action in self.spec.legal_actions(game, player):
                    check_cancel(cancel)
                    child = game.clone()
                    stack.callback(child.close)
                    outcome = self.spec.step(child, {player: action})
                    if outcome.truncated:
                        outputs[index][action] = 0.0
                    elif outcome.terminal:
                        outputs[index][action] = terminal_rewards(self.spec, child)[player]
                    else:
                        pending.append((index, action, player, child))
            for item, evaluation in zip(pending, self.evaluate([p[3] for p in pending], cancel)):
                index, action, player, child = item
                outputs[index][action] = evaluation.value * (1 if mover(self.spec, child) == player else -1)
        return outputs

    def one_ply(self, game, cancel=None):
        return self.one_ply_many([game], cancel)[0]


class ProxyAgent:
    """Pure distributions; choose advances a private RNG. Search uses calibrated fixed budgets.

    SearchLimits from the caller do not override calibration; cancellation remains cooperative.
    Classical search uses a fixed node cap and a one-hour emergency time cap.
    """

    def __init__(self, engine, style, params=None, seed=0, checkpoint_engines=None):
        self.engine, self.spec, self.style = engine, engine.spec, style
        self.params = dict(params or {})
        self.seed, self.rng = int(seed), np.random.default_rng(seed)
        self.checkpoint_engines = checkpoint_engines if checkpoint_engines is not None else {}
        if style not in TRAIN_STYLES + HELDOUT_STYLES:
            raise ValueError(f"unknown proxy style {style}")
        n = self.params.get("simulations", 0)
        if type(n) is not int or not 0 <= n <= 256:
            raise ValueError("qre simulations must be an integer in [0, 256]")
        for key, default in (("beta", 1.0), ("alpha", 0.5), ("epsilon", 0.0)):
            value = self.params.get(key, default)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if self.params.get("epsilon", 0) > 1:
            raise ValueError("epsilon must be in [0, 1]")
        if style == "temperature" and (not math.isfinite(self.params.get("temperature", 1))
                                        or self.params.get("temperature", 1) <= 0):
            raise ValueError("temperature must be positive")
        if style == "alphabeta" and (type(self.params.get("depth", 1)) is not int
                                     or not 1 <= self.params.get("depth", 1) <= 4):
            raise ValueError("alphabeta depth must be in [1, 4]")
        if style == "checkpoint" and not self.params.get("checkpoint"):
            raise ValueError("checkpoint style requires an earlier checkpoint")

    @property
    def policy_engine(self):
        if self.style != "checkpoint":
            return self.engine
        path = str(Path(self.params["checkpoint"]).resolve())
        if path not in self.checkpoint_engines:
            loaded = load_checkpoint(path, self.engine.device, expect_game_id=self.spec.game_id)
            check_manifest_matches_spec(loaded.manifest, self.spec, getattr(self.engine, "game_params", {}))
            if (loaded.manifest.encoding_id != self.engine.encoder.encoding_id
                    or loaded.manifest.num_planes != self.engine.encoder.num_planes):
                raise ValueError("earlier checkpoint must use the engine encoder")
            manifest = getattr(self.engine, "manifest", None)
            if manifest is not None and (loaded.manifest.run_id != manifest.run_id
                                          or loaded.manifest.iteration >= manifest.iteration):
                raise ValueError("checkpoint proxy must use an earlier checkpoint of the engine run")
            encoder = self.spec.make_encoder(loaded.manifest.encoding_id, **loaded.manifest.encoder_params)
            self.checkpoint_engines[path] = NetworkEvaluator(loaded.network, encoder, self.spec, self.engine.device)
        return self.checkpoint_engines[path]

    def from_prior(self, prior, q=None, best=None):
        if self.style in ("temperature", "checkpoint"):
            t = 0.5 if self.style == "checkpoint" else self.params.get("temperature", 1.0)
            # Log probabilities preserve masking, including underflowed legal logits.
            logits = {a: math.log(max(p, 1e-300)) for a, p in prior.items()}
            return prior_distribution(logits, list(prior), t, 1.0 if self.style == "checkpoint" else 0.95)
        if self.style == "qre":
            z = {a: self.params.get("alpha", 0.5) * math.log(max(p, 1e-300))
                 + self.params.get("beta", 1.0) * q[a] for a, p in prior.items()}
            peak = max(z.values())
            base = normalize({a: math.exp(v - peak) for a, v in z.items()})
        else:
            base = {a: float(a == best) for a in prior}
        lapse = top_distribution(prior)
        epsilon = self.params.get("epsilon", 0.0)
        return normalize({a: (1 - epsilon) * base[a] + epsilon * lapse.get(a, 0) for a in prior})

    def distribution(self, game, cancel=None):
        return proxy_distributions([self], [game], cancel)[0]

    def choose(self, game, limits, cancel=None, player=None):
        start = time.perf_counter()
        check_cancel(cancel)
        if player is not None and player != mover(self.spec, game):
            raise ValueError("proxy must act for the current mover")
        distribution = self.distribution(game, cancel)
        check_cancel(cancel)
        return Decision(sample_distribution(distribution, self.rng), elapsed_ms=(time.perf_counter() - start) * 1000)


def proxy_distributions(agents, games, cancel=None, priors=None, q_values=None):
    """Share root/one-ply batches and MCTS batches across independent games."""
    if len(agents) != len(games):
        raise ValueError("one proxy per game is required")
    result, rows = [None] * len(games), {}
    groups = {}
    for index, agent in enumerate(agents):
        groups.setdefault(agent.policy_engine, []).append(index)
    for evaluator, indices in groups.items():
        evaluated = evaluator.evaluate([games[i] for i in indices], cancel) if priors is None else None
        for row, index in enumerate(indices):
            rows[index] = evaluated[row].prior if evaluated is not None else priors[index]
    q_groups = {}
    for index, agent in enumerate(agents):
        if agent.style == "qre":
            q_groups.setdefault(agent.engine, []).append(index)
    qs = {}
    for evaluator, indices in q_groups.items():
        values = evaluator.one_ply_many([games[i] for i in indices], cancel) if q_values is None else None
        for row, index in enumerate(indices):
            qs[index] = dict(values[row] if values is not None else q_values[index])
        budgets = sorted({agents[i].params.get("simulations", 0) for i in indices} - {0})
        for budget in budgets:
            selected = [i for i in indices if agents[i].params.get("simulations", 0) == budget]
            search = MCTS(batched_evaluator(evaluator.network, evaluator.device, "off"), evaluator.encoder,
                          MctsConfig(simulations=budget, add_noise=False), np.random.default_rng(0), spec=evaluator.spec)
            searched = search.run_many([games[i] for i in selected], cancel=cancel)
            for index, found in zip(selected, searched):
                for a in qs[index]:
                    if found.q_values is not None and np.isfinite(found.q_values[a]):
                        qs[index][a] = float(found.q_values[a])
    for index, (agent, game) in enumerate(zip(agents, games)):
        check_cancel(cancel)
        best = None
        if agent.style == "alphabeta":
            baseline, limits = agent.spec.baseline_agent(f"alphabeta:{agent.params.get('depth', 1)}", agent.seed)
            snapshot = game.clone()
            try:
                depth = agent.params.get("depth", 1)
                fixed_limits = SearchLimits(depth, 3600000, (limits or SearchLimits()).node_limit)
                best = baseline.choose(snapshot, fixed_limits, cancel).action
            finally:
                snapshot.close()
            if best not in rows[index]:
                raise ValueError("baseline produced an illegal action")
        result[index] = agent.from_prior(rows[index], qs.get(index), best)
    return result


class MixtureProxy:
    def __init__(self, agents, seed=0):
        if not agents:
            raise ValueError("no available proxy styles at this level")
        self.agents = agents
        self.rng = np.random.default_rng(seed)

    def distribution(self, game, cancel=None):
        values = proxy_distributions(self.agents, [game] * len(self.agents), cancel)
        result = {}
        for distribution in values:
            for a, p in distribution.items():
                result[a] = result.get(a, 0.0) + p / len(values)
        return normalize(result)

    def choose(self, game, limits, cancel=None, player=None):
        check_cancel(cancel)
        return Decision(sample_distribution(self.distribution(game, cancel), self.rng))


class PracticalAdvisor:
    method = "learned"

    def __init__(self, engine, practical, levels, *, checkpoint_root=None):
        self.engine, self.practical, self.spec = engine, practical, engine.spec
        self.level_data = validate_levels(levels)
        if levels["game_id"] != self.spec.game_id:
            raise ValueError("levels belong to another game")
        self.checkpoint_root = Path(checkpoint_root or ".").resolve()
        self.checkpoint_engines = {}
        self.levels = []
        for row in levels["levels"]:
            entries = [v for v in row["styles"].values() if v["available"]]
            interval = (min(v["ci"][0] for v in entries), max(v["ci"][1] for v in entries)) if entries else None
            self.levels.append(LevelInfo(row["level"], row["elo_center"], interval,
                                         tuple(s for s, v in row["styles"].items() if v["available"])))

    def _level(self, level):
        if type(level) is not int or not 1 <= level <= len(self.levels):
            raise ValueError("level is outside calibrated range")
        return self.level_data["levels"][level - 1]

    def proxy_agent(self, level, *, style="mixture", seed=0):
        row = self._level(level)
        styles = self.level_data["train_styles"] if style == "mixture" else [style]
        agents = []
        for name in styles:
            entry = row["styles"].get(name)
            if entry is None or not entry["available"]:
                if style != "mixture":
                    raise ValueError(f"style {name} is unavailable at level {level}")
                continue
            params = dict(entry["params"])
            if "checkpoint" in params:
                params["checkpoint"] = str((self.checkpoint_root / params["checkpoint"]).resolve())
            agents.append(ProxyAgent(self.engine, name, params, seed, self.checkpoint_engines))
        return MixtureProxy(agents, seed) if style == "mixture" else agents[0]

    def _advice(self, game, action, score, value, loss, difficulty, naturalness, fallback=False):
        move_hook = getattr(self.spec, "action_to_move", None)
        text_hook = getattr(self.spec, "move_text", None)
        return Advice(int(action), int(move_hook(game, action) if move_hook else action),
                      str(text_hook(game, action) if text_hook else self.spec.action_name(action)),
                      float(score), float(value), float(max(0, loss)),
                      None if difficulty is None else float(difficulty), float(naturalness), fallback)

    def engine_best(self, game, cancel=None):
        values = self.engine.one_ply(game, cancel)
        best = min(values, key=lambda a: (-values[a], a))
        return self._advice(game, best, (values[best] + 1) / 2, values[best], 0, None,
                            self.engine.evaluate([game], cancel)[0].prior[best])

    def advise(self, game, level, opponent_level=None, *, top_k=3, guard=0.3, depth=1, cancel=None):
        check_cancel(cancel)
        opponent_level = level if opponent_level is None else opponent_level
        self._level(level)
        self._level(opponent_level)
        if type(top_k) is not int or top_k < 1 or depth not in (1, 2) or not math.isfinite(guard) or guard < 0:
            raise ValueError("invalid advice top_k, depth or guard")
        player = mover(self.spec, game)
        prior = self.engine.evaluate([game], cancel)[0].prior
        candidates = set(sorted(prior, key=lambda a: (-prior[a], a))[:16]) | {a for a in prior if prior[a] >= 0.01}
        values = self.engine.one_ply(game, cancel)
        best = min(values, key=lambda a: (-values[a], a))
        candidates = sorted(a for a in candidates if values[best] - values[a] <= guard)
        fallback = not candidates
        if fallback:
            candidates = [best]
        naturalness = self.proxy_agent(level).distribution(game, cancel)
        scores, difficulties = {}, {}
        with ExitStack() as stack:
            children = []
            for a in candidates:
                check_cancel(cancel)
                child = game.clone()
                stack.callback(child.close)
                outcome = self.spec.step(child, {player: a})
                if outcome.truncated or outcome.terminal:
                    scores[a] = 0.5 if outcome.truncated else (terminal_rewards(self.spec, child)[player] + 1) / 2
                    difficulties[a] = 0.0
                else:
                    children.append((a, child))
            predictions = self.practical.evaluate([g for _, g in children], cancel)
            grandchildren = []
            awaiting_own_turn = set()
            for (a, child), prediction in zip(children, predictions):
                child_player = mover(self.spec, child)
                own_turn = child_player == player
                i, j = (level - 1, opponent_level - 1) if own_turn else (opponent_level - 1, level - 1)
                scores[a] = float(prediction.practical[i, j] if own_turn else 1 - prediction.practical[i, j])
                difficulties[a] = float(prediction.slip[level - 1]) if own_turn else 0.0
                if not own_turn:
                    awaiting_own_turn.add(a)
                elif depth == 1:
                    continue
                reply = top_distribution(self.proxy_agent(level if own_turn else opponent_level).distribution(child, cancel))
                if depth == 2:
                    scores[a] = 0.0
                for b, probability in reply.items():
                    grandchild = child.clone()
                    stack.callback(grandchild.close)
                    outcome = self.spec.step(grandchild, {child_player: b})
                    if outcome.truncated or outcome.terminal:
                        score = 0.5 if outcome.truncated else (terminal_rewards(self.spec, grandchild)[player] + 1) / 2
                        if depth == 2:
                            scores[a] += probability * score
                    else:
                        grandchildren.append((a, probability, grandchild))
            predictions = self.practical.evaluate([g for _, _, g in grandchildren], cancel)
            for (a, probability, grandchild), prediction in zip(grandchildren, predictions):
                own_turn = mover(self.spec, grandchild) == player
                if depth == 2:
                    score = (prediction.practical[level - 1, opponent_level - 1] if own_turn
                             else 1 - prediction.practical[opponent_level - 1, level - 1])
                    scores[a] += probability * float(score)
                if a in awaiting_own_turn and difficulties[a] is not None and probability > 0:
                    # An opponent's extra turn can put the next own decision beyond this horizon.
                    difficulties[a] = (difficulties[a] + probability * float(prediction.slip[level - 1])
                                       if own_turn else None)
        check_cancel(cancel)
        ranked = sorted(candidates, key=lambda a: (-scores[a], -values[a], a))
        return [self._advice(game, a, scores[a], values[a], values[best] - values[a], difficulties[a],
                             naturalness.get(a, 0.0), fallback) for a in ranked[:top_k]]


class PracticalAgent:
    """Advisor agent with an optional search budget; SearchLimits do not cap forwards."""

    def __init__(self, advisor, level=3, opponent_level=None, guard=0.3, depth=None, time_ms=None):
        advisor._level(level)
        if opponent_level is not None:
            advisor._level(opponent_level)
        self.advisor, self.level, self.opponent_level = advisor, level, opponent_level
        self.guard = guard
        self.depth = (1 if getattr(advisor, "general", False) else 2) if depth is None else depth
        if time_ms is not None and advisor.method != "search":
            raise ValueError("time_ms is supported only by method search")
        if time_ms is not None and getattr(advisor, "general", False):
            raise ValueError("general search uses fixed sampling budgets; time_ms is unsupported")
        self.time_ms = time_ms

    def choose(self, game, limits, cancel=None, player=None):
        general = getattr(self.advisor, "general", False)
        if not general and player is not None and player != mover(self.advisor.spec, game):
            raise ValueError("agent must act for the current mover")
        start = time.perf_counter()
        options = {} if self.time_ms is None else {"time_ms": self.time_ms}
        if general:
            options["player"] = player
        advice = self.advisor.advise(game, self.level, self.opponent_level, top_k=1,
                                     guard=self.guard, depth=self.depth, cancel=cancel, **options)[0]
        return Decision(advice.action, score=round(1000 * (2 * advice.practical_score - 1)),
                        depth=(getattr(self.advisor, "last_search", None) or {}).get("depth", self.depth),
                        elapsed_ms=(time.perf_counter() - start) * 1000)


def create_advisor(checkpoint: str, device: str = "auto", method: str = "search") -> PracticalAdvisor:
    if method not in ("search", "auto", "learned"):
        raise ValueError("method must be search, auto or learned")
    device = resolve_device(device)
    loaded = load_checkpoint(checkpoint, device)
    spec = get_spec(loaded.manifest.game_id, loaded.manifest.game_params)
    general = spec.num_players != 2 or not spec.perfect_information or not getattr(spec, "zero_sum", True)
    if method == "learned":
        require_search_game(spec)
    hint = loaded.manifest.inference_hint or {}
    if hint.get("mode") == "practical":
        dependency = engine_dependency(loaded.directory)
        if dependency is None:
            raise ValueError("practical checkpoint is missing its engine reference")
        engine = load_checkpoint(dependency, device)
    else:
        if method == "learned":
            raise ValueError("learned advice requires a practical checkpoint")
        engine, dependency = loaded, loaded.directory
    if engine.manifest.network["head"] != "policy_value":
        raise ValueError("practical play needs a policy_value engine")
    check_manifest_matches_spec(engine.manifest, spec, loaded.manifest.game_params)
    levels_path = loaded.directory / "levels.json"
    levels = load_levels(levels_path) if levels_path.exists() or hint.get("mode") == "practical" else None
    if levels is not None:
        if levels["engine_sha256"] != sha256_file(dependency / "weights.pt"):
            raise ValueError("levels engine_sha256 differs from engine checkpoint")
        if levels.get("game_params", {}) != engine.manifest.game_params:
            raise ValueError("levels and engine game parameters differ")
        if hint.get("mode") == "practical" and levels["k"] != hint["levels"]:
            raise ValueError("levels and checkpoint level count differ")
    def evaluator(model):
        encoder = spec.make_encoder(model.manifest.encoding_id, **model.manifest.encoder_params)
        result = NetworkEvaluator(model.network, encoder, spec, device)
        result.game_params = model.manifest.game_params
        result.manifest = model.manifest
        return result
    if method in ("search", "auto"):
        if general:
            from .practical_general import GeneralSearchAdvisor
            advisor = GeneralSearchAdvisor(evaluator(engine), levels, checkpoint_root=loaded.directory)
        else:
            from .practical_search import SearchAdvisor
            advisor = SearchAdvisor(evaluator(engine), levels, checkpoint_root=loaded.directory)
    else:
        k = levels["k"]
        expected = {"practical": {"shape": [k, k], "activation": "sigmoid"},
                    "slip": {"shape": [k], "activation": "sigmoid"}}
        if any(getattr(loaded.network, "aux_heads", {}).get(n) != declaration for n, declaration in expected.items()):
            raise ValueError("practical auxiliary heads do not match calibrated levels")
        advisor = PracticalAdvisor(evaluator(engine), evaluator(loaded), levels, checkpoint_root=loaded.directory)
    advisor.manifest, advisor.checkpoint_dir = loaded.manifest, str(loaded.directory)
    advisor.engine_checkpoint = dependency
    advisor.levels_path = levels_path if levels is not None else None
    return advisor


def create_practical_agent(checkpoint: str, device: str = "auto", level: int = 3,
                           opponent_level: int | None = None, guard: float = 0.3, depth: int | None = None,
                           method: str = "search", time_ms: float | None = None):
    advisor = create_advisor(checkpoint, device, method)
    agent = PracticalAgent(advisor, level, opponent_level, guard, depth, time_ms)
    agent.manifest, agent.checkpoint_dir = advisor.manifest, advisor.checkpoint_dir
    return agent


def create_proxy_agent(checkpoint: str, device: str = "auto", level: int = 3, style: str = "mixture", seed: int = 0,
                       method: str = "search"):
    return create_advisor(checkpoint, device, method).proxy_agent(level, style=style, seed=seed)
