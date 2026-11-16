# Game adapters

The normative contracts are [GameSpec v2](../../docs/CODE_GUIDE_ja.md)
and the [game abstraction specification](../../docs/CODE_GUIDE_ja.md).
Adapters expose per-player observations, fixed-seat rewards, joint decisions, and
seeded chance transitions. Game mechanics and hidden-information filtering stay
in the adapter and engine. MCTS and AlphaZero require two players and perfect
information; the other learners accept any supported player count.

## Selecting an adapter

Training configs accept `"game": {"name": "hasami", "params": {}}`. Omitting this
section retains Hasami. An empty encoder name selects the adapter's default encoder;
existing explicit encoder names continue to work. Omitted self-play and evaluation
caps use `spec.default_max_plies`.

Register an adapter factory explicitly with `GAMES.register(name)`. Built-in adapters
are imported by `games/__init__.py`; add a new module there when introducing another
production game. Spawned workers need the same explicit registration available in
their interpreter. No filesystem plugin discovery is performed.

`Trainer` creates the selected spec and supplies `LearnerContext.spec`. Networks
receive `board_shape` and `action_size`; replay uses `fields_for(spec, num_planes,
kind)`. Collection, search, opponents, evaluation, datasets and ladder runs delegate
game operations to that spec. `augment(spec, ...)` yields each symmetry, excluding
the original sample; an adapter without symmetries returns an empty list.

`GameSpec.make_encoder` accepts the configured name or an empty default name.
Optional adapter attributes include `action_encoding` (default `index`). Adapters
can provide `save_record` and `load_record` hooks to preserve an existing record
format; otherwise the shared helper stores the initial state, rules, complete move
history, game identity and outcome, and validates legality and outcome on replay.
Hasami keeps its existing record format and horizontal mirror implementation.

## Compatibility

New manifests store `game_id`, `board_shape`, and constructor `game_params`. Loading
an older manifest without game identity defaults to Hasami. The loader checks the
registered game's ruleset, rules, board dimensions, action encoding and encoder
plane count before building the network. Callers can require a game with
`load_checkpoint(..., expect_game_id=...)` or `create_agent(..., game_id=...)`.

Legacy Hasami encoder exports, network defaults, replay field constants, baseline
factory and horizontal augmentation entry points remain available. New consumers
should use the explicit spec. Self-play archives carry game identity in their
metadata and journal rows and refuse a different game's trajectories or archives.

## Validation (2026-09-15)

`tests/learn/test_game_spec.py` covers adapter registration, Hasami symmetry round
trips, rectangular MLP/ResNet outputs, replay shapes, legacy checkpoint loading,
identity/shape refusals, and a test-local 4x4 engine. The toy engine exercises all
six learners, checkpoint inference, MCTS worker job execution, datastore replay,
supervised records and ladder evaluation. Existing Hasami tests additionally cover
actual spawned CPU workers, native parity, rules and checkpoint resilience.

CPU validation passed for all six existing tiny configurations. The full suite
passed 311 tests with 32 opt-in GUI cases skipped; native validation was required.
No CUDA or MPS execution was performed. The local native build passed its CTest;
both backends returned 3,717 leaves at depth two in the five-round benchmark.


## GameSpec v2

Implement `num_players`, `perfect_information`, `actors(game)`,
`legal_actions(game, player)`, `observe(game, player)`, `legal_mask(game, player)`,
`step(game, actions)`, and `terminal_values(game)`. `create_game` accepts a seed
that determines the initial deal and subsequent chance events. A joint step
receives exactly the current actors' actions; an empty actor list advances a
chance-only phase. `StepOutcome` carries `terminal`, `truncated`, optional final
rewards, and an information dictionary. Terminal values are finite, bounded by
[-1, 1], and need not sum to zero.

Encoders accept `encode(game, player=None)` and must expose only that player's
information. `run_table_game({seat: callback_or_agent}, encoder, spec=...)` records
callbacks and plays agent seats without recording their decisions. Callbacks may
accept four arguments or append the player as a fifth. Agents may add an optional
`player` keyword to `choose`; policy and Q agents support it. Each decision gets a
separate full-history clone, including separate clones during simultaneous phases.
Configured agents and callbacks remain trusted code: a full game clone is not a
security boundary against reading hidden state.

A trajectory stores the observing player per sample, a reward list, the game seed,
and every joint step. Its `plies` counts joint steps, not decisions. PPO and DQN
connect each player's successive decisions, including when other players act
between them. Draw contempt is applied only to reasons declared by the adapter.
Optional `draw_reasons`, `default_baselines`, and `default_teacher` provide game
specific defaults. Omitted/null config lists follow the selected adapter; explicit
lists retain their values.

`play_matches(primary, opponents, games, ...)` accepts one repeated opponent or a
list containing the other seats. `games` remains the total number of games, and
seat assignments rotate cyclically. With a seed, each complete rotation cycle
uses the same deal seed. Completed games contribute competition-rank placements
and mean terminal points; caps are reported separately and contribute neither.
The two-player `MatchResult.to_dict()` keys remain unchanged. For larger tables,
the dictionary additionally contains placements and mean points. N-player ladders
require a game count divisible by the number of seats and use Plackett-Luce
ratings, marginalizing tied orders with a weak symmetric prior. Tie computation
scales factorially with the size of a tied group; ordinary three/four-player
rankings are the intended workload. Two-player ladders retain Bradley-Terry.

New generic records and archives retain joint actions and the seed so replay
regenerates the same chance outcomes and per-seat observations. Hasami keeps its
existing engine record format. Old records, archives, datasets, worker jobs, and
manifests default to two players. Explicitly registered v1 specs are wrapped at
the adapter boundary; no filesystem discovery is added. The existing metrics
schema is preserved: N-player win counts use `extra.seat_wins`.

### V2 verification (2026-09-15, macOS arm64)

- `HASAMI_REQUIRE_NATIVE=1 .venv/bin/python -m unittest discover -s tests -v`:
  371 tests, passed; 32 opt-in Tk tests and one optional Mahjong reference test skipped.
- `/opt/homebrew/bin/python3 -m unittest discover -s tests -v`: 371 tests,
  passed; 200 optional dependency/GUI tests skipped. Both commands also passed
  with `-t .` to make the project import root explicit.
- `tests.learn.test_multiplayer`: 11 tests passed, covering per-player observations,
  joint/chance steps, clone isolation, rewards, duplicate deals, seat rotations,
  archive/dataset replay, DQN same-seat transitions, game-derived defaults,
  MCTS/AlphaZero refusals, ladder resume/ties, and CPU training/checkpoint inference
  for REINFORCE, PPO, DQN, supervised, and adversarial on both toy games.
- All six `configs/tiny/*.json` Hasami runs completed with run IDs `v2-<algorithm>`
  under `/tmp/hasami-smoke`; every checkpoint and saved config reloads.
- `runs/alphazero-mps-demo` iteration 10 loaded on CPU and selected legal action 5859.
- Schema validation passed (32 schemas, seven examples); the existing model-manifest
  and training-config examples remain valid. Four unrelated schemas lack examples,
  which the validator reports as warnings.
- Native Release and sanitizer builds in `/tmp/hasami-v2-build` and
  `/tmp/hasami-v2-sanitize` passed CTest. The depth-two, five-round benchmark returned
  3,717 leaves for both engines (median C++ 0.002287 s, Python 0.077116 s).

These checks establish compatibility and functioning toy training, not playing
strength. GPU training, Windows execution, and real Tk events were not run for
this change. Build products, smoke checkpoints, and logs remain outside the source
bundle. Initial discovery errors from concurrently added Riichi tests were absent
in the final full-suite runs; this task did not edit those files.
