"""Independent fixtures for agent conformance, with no shared worker helpers."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random

from hasami import create_game
from hasami.agents import AgentRegistry, SearchLimits
from hasami.model import Position
from hasami.records import record_dict


ROOT = Path(__file__).resolve().parents[2]
CYCLE = (72 * 81 + 63, 0 * 81 + 9, 63 * 81 + 72, 9 * 81 + 0)


def position(black, white, turn=0, pending=-1):
    """Construct a position using the public absolute square mapping."""
    return Position(sum(1 << s for s in black), sum(1 << s for s in white),
                    turn, pending)


def root_state(game):
    """Take a detached snapshot including history and last-capture state."""
    keys = game.history_keys()
    return deepcopy((record_dict(game), keys, Counter(keys), game.outcome,
                     game.last_capture))


def random_records(count=30, seed=9009):
    """Generate distinct reachable positions, balanced across both turns."""
    rng = random.Random(seed)
    records = []
    seen = set()
    game = create_game("python")
    try:
        for _ in range(10000):
            if game.outcome.terminal:
                game.close()
                game = create_game("python")
            game.play(rng.choice(game.legal_actions()))
            if game.outcome.terminal or game.position.turn != len(records) % 2:
                continue
            key = game.position.key()
            if key not in seen:
                seen.add(key)
                records.append(record_dict(game))
                if len(records) == count:
                    return records
    finally:
        game.close()
    raise AssertionError("Seeded trajectories did not supply enough positions")


def classical_cases():
    """Load both trusted examples through the real registry configuration."""
    registry = AgentRegistry(seed=9009)
    registry.load_config(ROOT / "examples" / "agents.example.json")
    cases = [(name, registry.create(name), SearchLimits(1, 10, 256))
             for name in ("random", "greedy", "plugin_first")]
    cases.extend((f"alphabeta:{depth}", registry.create("alphabeta"),
                  SearchLimits(depth, 10, 256)) for depth in (1, 2, 3))
    cases.append(("external_random", registry.create("external_random"),
                  SearchLimits(1, 1000, 256)))
    return cases
