"""Render user service launchers and optionally install them on WSL2."""
import argparse
import configparser
import getpass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hasami_learn.autoloop.plan import Plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply-wsl-conf", action="store_true")
    args = parser.parse_args()
    plan = Plan(args.config)
    identity = plan.data["loop_id"]
    settings = Path.home() / ".config" / "hasami-autoloop" / identity
    units = Path.home() / ".config" / "systemd" / "user"
    start = "#!/usr/bin/env bash\nset -Eeuo pipefail\ncd -- " + shlex.quote(str(ROOT)) + "\nexec " + shlex.join(
        [sys.executable, "-m", "hasami_learn.autoloop", "run", "--config", str(plan.path)]) + "\n"
    # ExecCondition prevents Restart=always from undoing an intentional stop,
    # or repeatedly relaunching a completed or deterministically blocked plan.
    ready = "#!/usr/bin/env bash\nset -Eeuo pipefail\ncd -- " + shlex.quote(str(ROOT)) + "\nexec " + shlex.join([sys.executable, "-c", '''import sys
from pathlib import Path
from hasami_learn.autoloop.state import LoopError, StateStore
p = Path(sys.argv[1])
if (p / "control/stop").exists():
    sys.exit(1)
try:
    state = StateStore(p).load() or {}
except LoopError:
    # Let the controller record the failure when neither copy is usable.
    state = {}
sys.exit(1 if state.get("status") in ("complete", "blocked") else 0)
''', str(plan.loop_dir)]) + "\n"
    commands = [["loginctl", "enable-linger", getpass.getuser()], ["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now", f"hasami-autoloop@{identity}.service"]]
    if args.dry_run:
        print("Check /etc/wsl.conf: [boot] systemd=true")
        print(f"Install {units / 'hasami-autoloop@.service'}")
        print(f"Write {settings / 'start.sh'} and ready.sh")
        print(start, end="")
        for command in commands:
            print(shlex.join(command))
        for warning in plan.warnings:
            print("WARNING: " + warning)
        return 0
    if sys.platform != "linux" or "microsoft" not in os.uname().release.lower():
        parser.error("real installation requires WSL2; use --dry-run elsewhere")
    conf = configparser.ConfigParser()
    conf.read("/etc/wsl.conf")
    if not conf.getboolean("boot", "systemd", fallback=False):
        if not args.apply_wsl_conf:
            parser.error("set [boot] systemd=true in /etc/wsl.conf, restart WSL, then retry; --apply-wsl-conf explicitly permits editing it")
        # Privilege escalation is explicit and limited to this one requested file.
        subprocess.run(["sudo", sys.executable, "-c", '''import configparser
p = configparser.ConfigParser()
p.read('/etc/wsl.conf')
if not p.has_section('boot'): p.add_section('boot')
p.set('boot', 'systemd', 'true')
with open('/etc/wsl.conf', 'w') as f: p.write(f)
'''], check=True)
        print("Updated /etc/wsl.conf. Restart WSL yourself, then run this installer again.")
        return 0
    settings.mkdir(parents=True, exist_ok=True)
    units.mkdir(parents=True, exist_ok=True)
    (settings / "start.sh").write_text(start)
    (settings / "ready.sh").write_text(ready)
    shutil.copyfile(Path(__file__).with_name("hasami-autoloop@.service"), units / "hasami-autoloop@.service")
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
