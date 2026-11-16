"""Command line entry: python -m hasami_learn.cli {train,eval,devices,list,dataset,export-agents-config,config-template}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import USE_DEFAULT, ConfigError, TrainingConfig, tiny_config
from .runlock import writer_path


def _parse_set(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        key, sep, raw = item.partition("=")
        if not sep or "." not in key:
            raise ConfigError(f"--set expects section.key=value, got {item}")
        section, name = key.split(".", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if value is None:
            value = USE_DEFAULT  # --set selfplay.max_plies=null resets the cap to the game default
        overrides[f"{section}__{name}"] = value
    return overrides


def resolve_train_request(args: argparse.Namespace) -> tuple[TrainingConfig, Path | None]:
    """Resolve training; auto recovery holds the same lock as the trainer."""
    from .manifest import ManifestError, find_resume_checkpoint, recover_retired_checkpoints, resume_run_dir
    from .runlock import run_lock

    if args.config:
        config = TrainingConfig.load(args.config)
    elif args.algorithm:
        config = tiny_config(args.algorithm) if args.tiny else TrainingConfig.from_dict({**TrainingConfig().to_dict(), "algorithm": {"name": args.algorithm, "params": {}}})
    elif args.resume and args.resume != "auto":
        config = TrainingConfig.load(resume_run_dir(args.resume) / "config.json")
    else:
        raise ConfigError("pass --config <file.json> or --algorithm <name>")
    overrides = _parse_set(args.set)
    if args.device:
        overrides["device__device"] = args.device
    if args.iterations is not None:
        overrides["run__iterations"] = args.iterations
    if args.run_id:
        overrides["run__run_id"] = args.run_id
    if args.output_dir:
        overrides["run__output_dir"] = args.output_dir
    if getattr(args, "init_checkpoint", None) is not None:
        overrides["run__init_checkpoint"] = args.init_checkpoint
    if args.seed is not None:
        overrides["run__seed"] = args.seed
    if args.backend:
        overrides["run__backend"] = args.backend
    if args.games is not None:
        overrides["selfplay__games_per_iteration"] = args.games
    config = config.with_overrides(**overrides)
    resume = Path(args.resume) if args.resume and args.resume != "auto" else None
    if args.resume == "auto":
        candidate = writer_path(Path(config.run.output_dir) / config.run.run_id)
        try:
            # A crash between the two renames of a checkpoint replacement leaves the only valid
            # generation retired under .tmp_replaced_*; put it back before deciding on a fresh start.
            with run_lock(candidate):
                recover_retired_checkpoints(candidate)
                find_resume_checkpoint(candidate, cleanup=False)
        except ManifestError:
            pass
        else:
            resume = candidate
    if resume is None:
        return config, None

    saved_path = resume_run_dir(resume) / "config.json"
    saved = TrainingConfig.load(saved_path)
    # Compare *resolved* values: older saved configs spell game defaults out (max_plies=300,
    # draw_reasons=["repetition"]) while newer ones store null for game-derived fields.
    original = saved.to_dict(preserve_defaults=False)
    requested = config.to_dict(preserve_defaults=False)
    conflicts = []
    from .games import get_spec

    default_encoder = getattr(get_spec(saved.game.name, saved.game.params), "default_encoder", "")

    def resolved_default(section: str, key: str):
        """The value a null (game-derived) field resolves to under the saved game selection."""
        data = saved.to_dict(preserve_defaults=False)
        data[section][key] = None
        return getattr(getattr(TrainingConfig.from_dict(data), section), key)

    def same_value(section: str, key: str, value, expected) -> bool:
        if (section, key) == ("run", "init_checkpoint") and value is None:
            return True
        if (section, key) == ("run", "output_dir"):
            return Path(value).resolve() == Path(expected).resolve()
        if (section, key) == ("encoder", "name"):
            # "" means the game's default encoder; saved configs may spell the default out.
            return (value or default_encoder) == (expected or default_encoder)
        return value == expected
    if args.config or args.tiny:
        for section, values in requested.items():
            if not isinstance(values, dict):
                if values != original[section]:
                    conflicts.append(section)
                continue
            for key, value in values.items():
                if (section, key) in (("run", "iterations"), ("device", "device")):
                    continue
                if not same_value(section, key, value, original[section][key]):
                    conflicts.append(f"{section}.{key}")
    if args.algorithm and args.algorithm != saved.algorithm.name:
        conflicts.append("algorithm.name")
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        if value is USE_DEFAULT:
            value = resolved_default(section, key)  # --set x=null means the game default
        if dotted == "device__device" and args.device is not None:
            continue
        if dotted == "run__iterations" and args.iterations is not None:
            if value < saved.run.iterations:
                conflicts.append("run.iterations (--iterations can only extend the saved target)")
            continue
        if not same_value(section, key, value, original[section][key]):
            conflicts.append(f"{section}.{key}")
    if conflicts:
        raise ConfigError(f"resume uses saved config {saved_path}; conflicting overrides: "
                          + ", ".join(sorted(set(conflicts)))
                          + "; only --device, --iterations to extend, and --quiet may change")
    saved = saved.with_overrides(device__device=args.device, run__iterations=args.iterations)
    return saved, resume


def cmd_train(args: argparse.Namespace) -> int:
    config, resume = resolve_train_request(args)
    from .trainer import train_from_config
    from .runlock import run_lock

    with run_lock(Path(config.run.output_dir) / config.run.run_id):
        # Recheck under the lifetime lock: another invocation may have completed
        # between the first config/probe pass and acquisition.
        config, resume = resolve_train_request(args)
        final = train_from_config(config, resume=resume, echo=not args.quiet)
    print(f"final checkpoint: {final}")
    return 0


def check_compatible_manifests(primary, other) -> None:
    """Refuse to play two checkpoints that belong to different game variants."""
    for field in ("game_id", "num_players", "game_params", "rules", "ruleset_id", "board_shape", "action_size", "action_encoding"):
        left, right = getattr(primary, field, None), getattr(other, field, None)
        if list(left) != list(right) if isinstance(left, (list, tuple)) else left != right:
            raise ConfigError(f"checkpoints belong to different game variants: {field} {left!r} != {right!r}")


def cmd_eval(args: argparse.Namespace) -> int:
    from hasami.agents import SearchLimits
    from .evaluation import play_matches
    from .inference import create_agent, describe_agent

    output = writer_path(args.out) if args.out else None
    records = writer_path(args.records_dir) if args.records_dir else None
    agent = create_agent(args.checkpoint, device=args.device, mode=args.mode, simulations=args.simulations, seed=args.seed)
    print(describe_agent(agent))
    game_spec = agent.spec
    if not args.opponent:
        # Default opponents come from the checkpoint's game (Hasami: random, greedy; others declare their own).
        args.opponent = list(getattr(game_spec, "default_baselines", ("random",)))
    limits = SearchLimits(max_depth=args.depth, time_ms=args.time_ms, node_limit=args.simulations or 200000)
    results = {}
    for spec in args.opponent:
        if spec.startswith("checkpoint:"):
            opponent = create_agent(spec.split(":", 1)[1], device=args.device, simulations=args.simulations, seed=args.seed + 1,
                                    game_id=game_spec.game_id, rules=dict(agent.manifest.rules))
            check_compatible_manifests(agent.manifest, opponent.manifest)
            opp_limits = limits
        else:
            opponent, opp_limits = game_spec.baseline_agent(spec, seed=args.seed + 1)
        match = play_matches(agent, opponent, args.games, backend=args.backend, max_plies=args.max_plies, limits=limits,
                             opponent_limits=opp_limits or limits, swap_sides=not args.no_swap, spec=game_spec, seed=args.seed,
                             records_dir=writer_path(records / spec.replace(":", "_")) if records else None,
                             opening_plies=args.opening_plies, opening_seed=args.opening_seed)
        results[spec] = match.to_dict()
        print(json.dumps({"opponent": spec, **results[spec]}, ensure_ascii=False), flush=True)
    if args.out:
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def cmd_devices(_args: argparse.Namespace) -> int:
    from .devices import device_report

    print(json.dumps(device_report(), indent=2))
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    from .catalog import describe_all

    print(json.dumps(describe_all(), indent=2))
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    # Validate the delegated writer destinations before the dataset CLI starts
    # generating games. Its input records remain read-only.
    outputs = argparse.ArgumentParser(add_help=False)
    outputs.add_argument("--out")
    outputs.add_argument("--out-games")
    selected, rest = outputs.parse_known_args(args.rest)
    for option, path in (("--out", selected.out), ("--out-games", selected.out_games)):
        if path is not None:
            rest.extend([option, str(writer_path(path))])
    try:
        from . import datasets
    except Exception as exc:  # noqa: BLE001
        print(f"dataset tooling unavailable: {exc}", file=sys.stderr)
        return 1
    return int(datasets.main(rest))


def cmd_export_agents_config(args: argparse.Namespace) -> int:
    from .manifest import MANIFEST_FILE, resolve_checkpoint_dir

    output = writer_path(args.out) if args.out else None
    agents = []
    for item in args.checkpoint:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = None, item
        directory = resolve_checkpoint_dir(path)
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        agent_id = name or f"model:{manifest.get('run_id') or directory.parent.parent.name}"
        options: dict[str, Any] = {"checkpoint": str(directory), "device": args.device, "mode": args.mode}
        if args.simulations is not None:
            options["simulations"] = args.simulations
        agents.append({"id": agent_id, "type": "python", "family": manifest.get("algorithm", "learned"),
                       "factory": "hasami_learn.inference:create_agent", "options": options,
                       "description": f"{manifest.get('algorithm')} iteration {manifest.get('iteration')} ({manifest.get('encoding_id')})"})
    payload = {"agents": agents}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        output.write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text, end="")
    return 0


def cmd_config_template(args: argparse.Namespace) -> int:
    output = writer_path(args.out) if args.out else None
    config = tiny_config(args.algorithm) if args.tiny else TrainingConfig.from_dict({**TrainingConfig().to_dict(), "algorithm": {"name": args.algorithm, "params": {}}})
    text = json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        output.write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hasami_learn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="run a training configuration")
    p.add_argument("--config", help="training config JSON (hasami_learn/schemas/training-config.schema.json)")
    p.add_argument("--algorithm", help="algorithm name when no --config is given")
    p.add_argument("--tiny", action="store_true", help="use the CPU smoke preset for --algorithm")
    p.add_argument("--device", help="auto | cpu | mps | cuda | cuda:N")
    p.add_argument("--iterations", type=int)
    p.add_argument("--run-id")
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--backend", choices=("auto", "python", "cpp"))
    p.add_argument("--games", type=int, help="self-play games per iteration")
    p.add_argument("--init-checkpoint", help="initialize only compatible network weights on a fresh run")
    p.add_argument("--resume", help="run/checkpoint dir, or auto to recover output-dir/run-id when valid")
    p.add_argument("--set", action="append", default=[], help="override section.key=value (JSON value)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate a checkpoint against baselines or other checkpoints")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", action="append", default=[], help="random | greedy | alphabeta:<depth> | checkpoint:<dir>")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--mode", default="auto")
    p.add_argument("--simulations", type=int)
    p.add_argument("--backend", default="python")
    p.add_argument("--max-plies", type=int)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--time-ms", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opening-plies", type=int, default=0)
    p.add_argument("--opening-seed", type=int)
    p.add_argument("--no-swap", action="store_true")
    p.add_argument("--records-dir")
    p.add_argument("--out")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("devices", help="report available devices")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("list", help="list registered algorithms, networks, encoders, strategies")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("dataset", help="build supervised datasets (delegates to hasami_learn.datasets)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("export-agents-config", help="write an agents-config JSON for the GUI/arena")
    p.add_argument("--checkpoint", action="append", required=True, help="[name=]path (repeatable)")
    p.add_argument("--device", default="auto")
    p.add_argument("--mode", default="auto")
    p.add_argument("--simulations", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_export_agents_config)

    p = sub.add_parser("config-template", help="print a config for an algorithm")
    p.add_argument("--algorithm", required=True)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_config_template)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # ``dataset`` forwards every following argument untouched to hasami_learn.datasets.main,
    # so options such as --generate never reach the parent parser.
    if argv and argv[0] == "dataset":
        rest = argv[1:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        return cmd_dataset(argparse.Namespace(rest=rest))
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
