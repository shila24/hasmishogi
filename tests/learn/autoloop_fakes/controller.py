"""Real controller process with a fake command adapter; no production hooks."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from hasami_learn.autoloop.plan import Plan
from hasami_learn.autoloop.runner import Controller
from hasami_learn.autoloop.stages import Stages


class FakeStages(Stages):
    def command(self, stage, cycle):
        return [sys.executable, str(Path(__file__).with_name("stage.py")), stage["args"][0]]


controller = Controller(Plan(sys.argv[1], running=True), health_probe=lambda *_: {"ok": True})
controller.stages = FakeStages(controller.plan)
raise SystemExit(controller.run())
