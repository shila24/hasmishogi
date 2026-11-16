"""Resolve the evaluation device inside the stage and run the existing CLI."""
import subprocess
import sys

from .evidence import utc_now
from .state import atomic_json
from ..runlock import writer_path


def main(argv=None):
    # Learning imports remain confined to this supervised stage subprocess.
    import torch
    from hasami_learn.cli import build_parser
    from hasami_learn.devices import resolve_device

    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(["eval", *arguments])
    output = writer_path(args.out)
    metadata = writer_path(output.with_name("results-device.json"))
    if args.records_dir:
        writer_path(args.records_dir)
    device = resolve_device(args.device)
    actual = str(device)
    if device.type == "cuda" and device.index is None:
        actual = f"cuda:{torch.cuda.current_device()}"
    # The final explicit option wins over any requested 'auto' option.
    code = subprocess.run([sys.executable, "-m", "hasami_learn.cli", "eval",
                           *arguments, "--device", actual]).returncode
    if code == 0:
        atomic_json(metadata,
                    {"device": actual, "created_at": utc_now()})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
