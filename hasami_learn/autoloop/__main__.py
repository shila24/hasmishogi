"""CLI for local, resumable subprocess plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .plan import Plan
from .runner import Controller
from .state import atomic_json, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        if name == "run":
            sub.add_argument("--allow-unvalidated", action="store_true")
            sub.add_argument("--accept-config-change", action="store_true")
    for name in ("status", "stop", "resume-check"):
        sub = commands.add_parser(name)
        sub.add_argument("--loop", required=True)
        if name == "status":
            sub.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            plan = Plan(args.config)
            print(json.dumps({"valid": True, "loop_id": plan.data["loop_id"], "warnings": plan.warnings}, indent=2))
            return 0
        if args.command == "run":
            plan = Plan(args.config, running=True)
            return Controller(plan).run(allow_unvalidated=args.allow_unvalidated, accept_config_change=args.accept_config_change)
        directory = Path(args.loop).expanduser()
        if args.command == "status":
            data = read_json(directory / "status.json")
            print(json.dumps(data, indent=2) if args.json else f"{data['status']} cycle={data['cycle']} stage={data['stage']} iteration={data['latest_iteration']}")
        else:
            atomic_json(directory / "control" / args.command, {"requested": True})
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"autoloop: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
