"""Run bounded gate/export work in a supervised standard-library subprocess."""
import sys

from .plan import Plan
from .stages import Stages
from .state import LoopError


def main():
    path, digest, stage_id, cycle = sys.argv[1:]
    plan = Plan(path, running=True)
    if plan.digest != digest:
        raise LoopError("plan changed while a stage was running")
    index = next(i for i, stage in enumerate(plan.data["stages"]) if stage["id"] == stage_id)
    stage = plan.stage(index, int(cycle))
    if stage["kind"] not in ("gate", "export"):
        raise LoopError("local stage must be gate or export")
    operations = Stages(plan)
    if not operations.complete(stage, int(cycle)):
        operations.local(stage, int(cycle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
