"""Inference agents that satisfy hasami.agents.Agent, plus the agents-config factory."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch

from hasami.agents import Decision, SearchLimits, check_cancel
from hasami.model import GameLike
from .devices import autocast_context, resolve_device
from .games import Encoder
from .games.base import as_v2, encode_player
from .encoding import sample_from_policy
from .manifest import ModelManifest, load_checkpoint
from .games import get_spec
from .networks.base import activate_aux


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    legal = mask > 0
    if not legal.any():
        return np.zeros_like(logits, dtype=np.float64)
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z -= z[legal].max()
    p = np.exp(z)
    p /= p.sum()
    return p


def policy_action(logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
    """The PolicyAgent decision rule: masked softmax, then argmax (temperature 0) or tempered sampling.

    Batched evaluators must call this per decision so that batching cannot change the rule.
    """
    return sample_from_policy(_softmax_masked(logits, mask), mask, rng, temperature)


def decision_extra(aux: dict[str, np.ndarray]) -> dict[str, Any]:
    """JSON-serialisable belief diagnostics for ``Decision.extra``: ``{"aux": {name: nested lists}}``."""
    if not aux:
        return {}
    return {"aux": {name: np.asarray(value, dtype=np.float64).tolist() for name, value in aux.items()}}


class _NetworkAgent:
    def __init__(self, network: torch.nn.Module, encoder: Encoder, device: torch.device | str, rng: np.random.Generator | None = None, amp: str = "auto", spec=None, aux_outputs: bool = True) -> None:
        self.spec = as_v2(spec or get_spec())
        self.network = network
        self.encoder = encoder
        self.device = resolve_device(device)
        self.rng = rng or np.random.default_rng(0)
        self.amp = amp
        self.aux_outputs = bool(aux_outputs)
        self.manifest: ModelManifest | None = None
        self.network.eval()

    @property
    def aux_heads(self) -> dict[str, Any]:
        """Auxiliary heads the network declares (see networks.base); empty without ``forward_aux``."""
        specs = getattr(self.network, "aux_heads", None) or {}
        return dict(specs) if hasattr(self.network, "forward_aux") else {}

    @torch.no_grad()
    def evaluate(self, game: GameLike, player: int | None = None) -> tuple[np.ndarray, float | None, np.ndarray]:
        out, val, mask, _ = self.evaluate_aux(game, player, aux=False)
        return out, val, mask

    @torch.no_grad()
    def evaluate_aux(self, game: GameLike, player: int | None = None, aux: bool | None = None
                     ) -> tuple[np.ndarray, float | None, np.ndarray, dict[str, np.ndarray]]:
        """Head output, value, legal mask and the activated auxiliary outputs (``{}`` when disabled)."""
        player = self._player(game, player)
        # Copy on return: adapters may reuse observation/mask buffers, and computing the mask may
        # query other seats' observations through the same buffer.
        planes = np.array(encode_player(self.encoder, game, player), dtype=np.float32, copy=True)
        mask = np.array(self.spec.legal_mask(game, player), copy=True)
        x = torch.as_tensor(planes, dtype=torch.float32, device=self.device).unsqueeze(0)
        want_aux = self.aux_outputs if aux is None else bool(aux)
        raw: dict[str, torch.Tensor] = {}
        with autocast_context(self.device, self.amp):
            if want_aux and self.aux_heads:
                head, value, raw = self.network.forward_aux(x)
            else:
                head, value = self.network(x)
        out = head.float().cpu().numpy()[0]
        val = None if value is None else float(value.float().cpu().numpy()[0])
        activated = {name: tensor.float().cpu().numpy()[0]
                     for name, tensor in activate_aux(self.aux_heads, raw).items()}
        return out, val, mask, activated

    def _player(self, game, player):
        if player is None:
            actors = self.spec.actors(game)
            if len(actors) != 1:
                raise ValueError("An explicit player is required for joint decisions")
            player = actors[0]
        return player

    def _require_legal(self, game: GameLike, player=None) -> list[int]:
        legal = self.spec.legal_actions(game, self._player(game, player))
        if not legal:
            raise ValueError("No legal action is available")
        return legal


class PolicyAgent(_NetworkAgent):
    """Sample or argmax from the masked policy head."""

    def __init__(self, network, encoder, device, temperature: float = 0.0, rng=None, amp: str = "auto", spec=None, aux_outputs: bool = True) -> None:
        super().__init__(network, encoder, device, rng, amp, spec, aux_outputs)
        self.temperature = float(temperature)

    def choose(self, game: GameLike, limits: SearchLimits, cancel: threading.Event | None = None, player: int | None = None) -> Decision:
        check_cancel(cancel)
        start = time.perf_counter()
        self._require_legal(game, player)
        logits, value, mask, aux = self.evaluate_aux(game, player)
        action = policy_action(logits, mask, self.rng, self.temperature)
        check_cancel(cancel)
        score = int(round((value or 0.0) * 1000))
        return Decision(int(action), score, 1, 1, (time.perf_counter() - start) * 1000, True,
                        extra=decision_extra(aux))


class QAgent(_NetworkAgent):
    """Greedy over masked Q-values (optionally epsilon-greedy for exploration)."""

    def __init__(self, network, encoder, device, epsilon: float = 0.0, rng=None, amp: str = "auto", spec=None, aux_outputs: bool = True) -> None:
        super().__init__(network, encoder, device, rng, amp, spec, aux_outputs)
        self.epsilon = float(epsilon)

    def choose(self, game: GameLike, limits: SearchLimits, cancel: threading.Event | None = None, player: int | None = None) -> Decision:
        check_cancel(cancel)
        start = time.perf_counter()
        legal = self._require_legal(game, player)
        q, _, mask, aux = self.evaluate_aux(game, player)
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(legal))
        else:
            masked = np.where(mask > 0, q, -np.inf)
            action = int(np.argmax(masked))
        check_cancel(cancel)
        return Decision(int(action), int(round(float(q[action]) * 1000)), 1, 1, (time.perf_counter() - start) * 1000, True,
                        extra=decision_extra(aux))


class MctsAgent(_NetworkAgent):
    """PUCT/Gumbel inference; contempt is zero unless mcts_params explicitly configures it.

    GUI, USI, evaluation, ladder and practical advisers use this default and never
    read the trainer's annealed draw state. Checkpoint selfplay settings do not leak here.
    """

    def __init__(self, network, encoder, device, simulations: int = 100, mcts_params: dict[str, Any] | None = None, rng=None, temperature: float = 0.0, amp: str = "auto", spec=None) -> None:
        super().__init__(network, encoder, device, rng, amp, spec)
        self.simulations = int(simulations)
        self.temperature = float(temperature)
        try:
            from .search import mcts as mcts_module
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"MCTS is not available: {exc}") from exc
        params = dict(mcts_params or {})
        params.setdefault("add_noise", False)
        if params.get("root_selection") == "gumbel":
            params["add_noise"] = False
        self.config = mcts_module.MctsConfig(**params)
        evaluate_fn = mcts_module.batched_evaluator(self.network, self.device, amp=self.amp)
        self.search = mcts_module.MCTS(evaluate_fn, self.encoder, self.config, self.rng, spec=self.spec)

    def choose(self, game: GameLike, limits: SearchLimits, cancel: threading.Event | None = None, player: int | None = None,
               *, use_time_limit: bool = True) -> Decision:
        check_cancel(cancel)
        start = time.perf_counter()
        self._require_legal(game, player)
        budget = max(1, min(self.simulations, int(limits.node_limit)))
        deadline = start + limits.time_ms / 1000.0 if use_time_limit else None
        result = self.search.run(game, simulations=budget, cancel=cancel, deadline=deadline)
        visits = np.asarray(result.policy_target if self.config.root_selection == "gumbel"
                            else result.visits, dtype=np.float64)
        mask = np.array(self.spec.legal_mask(game, self._player(game, player)), copy=True)
        if self.temperature > 0 and self.config.root_selection != "gumbel":
            action = sample_from_policy(visits / max(visits.sum(), 1.0), mask, self.rng, self.temperature)
        else:
            action = int(np.argmax(np.where(mask > 0, visits, -1.0)))
        check_cancel(cancel)
        return Decision(int(action), int(round(float(result.root_value) * 1000)), 0, int(result.simulations),
                        (time.perf_counter() - start) * 1000, int(result.simulations) >= budget)


MCTS_PARAM_KEYS = ("c_puct", "dirichlet_alpha", "dirichlet_epsilon", "batch_size", "virtual_loss",
                  "root_selection", "gumbel_m", "gumbel_c_visit", "gumbel_c_scale", "draw_value", "draw_reasons")


def _mcts_params_from_manifest(algorithm_params) -> dict[str, Any]:
    """Collect search parameters saved either flat (AlphaZero params) or under an ``mcts`` key."""
    if not isinstance(algorithm_params, dict):
        return {}
    params = {key: algorithm_params[key] for key in MCTS_PARAM_KEYS if key in algorithm_params}
    nested = algorithm_params.get("mcts")
    if isinstance(nested, dict):
        params.update(nested)
    return params


def agent_from_network(network, head: str, encoder: Encoder, device, mode: str = "auto", simulations: int = 0,
                       temperature: float = 0.0, rng: np.random.Generator | None = None, mcts_params: dict[str, Any] | None = None, amp: str = "auto", spec=None,
                       aux_outputs: bool = True):
    if mode == "auto":
        mode = "q" if head == "q" else ("mcts" if simulations > 0 else "policy")
    if mode == "q":
        if head != "q":
            raise ValueError("mode 'q' requires a network with a q head")
        return QAgent(network, encoder, device, rng=rng, amp=amp, spec=spec, aux_outputs=aux_outputs)
    if head != "policy_value":
        raise ValueError(f"mode '{mode}' requires a policy_value head")
    if mode == "policy":
        return PolicyAgent(network, encoder, device, temperature=temperature, rng=rng, amp=amp, spec=spec, aux_outputs=aux_outputs)
    if mode == "mcts":
        return MctsAgent(network, encoder, device, simulations=max(1, simulations), mcts_params=mcts_params, rng=rng, temperature=temperature, amp=amp, spec=spec)
    raise ValueError(f"unknown inference mode {mode}; expected auto, policy, mcts, q or practical")


def create_agent(checkpoint: str, device: str = "auto", mode: str = "auto", simulations: int | None = None,
                 temperature: float = 0.0, seed: int = 0, rules: dict[str, Any] | None = None,
                 mcts_params: dict[str, Any] | None = None, amp: str = "auto", game_id: str | None = None,
                 aux_outputs: bool = True):
    """Factory for agents-config JSON: ``hasami_learn.inference:create_agent``.

    The encoder is rebuilt from ``manifest.encoding_id`` and ``encoder_params`` (for example
    ``riichi-history`` with ``k``), and the network from ``manifest.network`` including any
    ``aux_heads``; policy and q agents publish the activated auxiliary outputs in
    ``Decision.extra["aux"]`` unless ``aux_outputs`` is false. Raises explicit errors when the
    checkpoint is missing, incompatible with the ruleset, or when the requested mode needs an
    unavailable component.
    """
    resolved = resolve_device(device)
    loaded = load_checkpoint(checkpoint, resolved, expect_rules=rules, expect_game_id=game_id)
    manifest = loaded.manifest
    spec = get_spec(manifest.game_id, manifest.game_params)
    encoder = spec.make_encoder(manifest.encoding_id, **(manifest.encoder_params or {}))
    if encoder.num_planes != manifest.num_planes:
        raise ValueError(f"encoder {manifest.encoding_id} has {encoder.num_planes} planes but manifest says {manifest.num_planes}")
    hint = manifest.inference_hint or {}
    if mode == "auto":
        mode = str(hint.get("mode", "auto"))
    if mode == "practical":
        # Practical-play checkpoints (hasami_learn/practical.py) carry their own agent factory.
        from .practical import create_practical_agent

        return create_practical_agent(checkpoint, device=device)
    if simulations is None:
        simulations = int(hint.get("simulations", 0))
    params = _mcts_params_from_manifest(manifest.algorithm_params)
    params.update(dict(hint.get("mcts_params") or {}))  # settings the learner resolved at training time
    if mcts_params:
        params.update(mcts_params)
    agent = agent_from_network(loaded.network, manifest.network["head"], encoder, resolved, mode=mode, simulations=int(simulations),
                               temperature=temperature, rng=np.random.default_rng(seed), mcts_params=params, amp=amp, spec=spec,
                               aux_outputs=aux_outputs)
    agent.manifest = manifest
    agent.checkpoint_dir = str(loaded.directory)
    return agent


def describe_agent(agent, compact: bool = False) -> str:
    """Human readable summary; ``compact`` returns one short line for narrow GUI labels."""
    manifest = getattr(agent, "manifest", None)
    if manifest is None:
        return type(agent).__name__
    mode = type(agent).__name__.replace("Agent", "").lower()
    sims = getattr(agent, "simulations", 0)
    if compact:
        return f"{manifest.algorithm} #{manifest.iteration} · {mode}{f' sims {sims}' if sims else ''} · {manifest.train_device or '?'}"
    aux = getattr(agent, "aux_heads", None) or {}
    return (f"{manifest.algorithm} iter {manifest.iteration} / {manifest.encoding_id} / head {manifest.network['head']} / "
            f"trained on {manifest.train_device or '?'} / mode {mode}{f' sims {sims}' if sims else ''}"
            f"{' / aux ' + ','.join(sorted(aux)) if aux else ''}")
