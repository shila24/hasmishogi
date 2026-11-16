"""Small idempotent subprocess used by the autoloop crash tests."""
import json
import os
from pathlib import Path
import sys
import time

root = Path(os.environ["HASAMI_AUTOLOOP_STAGE_DIR"])
done = Path(os.environ["HASAMI_AUTOLOOP_DONE_FILE"])
mode = sys.argv[1]
count = root / "launches"
count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))
if mode == "crash":
    sys.exit(7)
if mode == "temporary" and int(count.read_text()) < 3:
    sys.exit(75)
if mode == "hang":
    while True:
        time.sleep(1)
for step in range(10):
    (root / "progress").write_text(str(step))
    time.sleep(0.02)
if not done.exists():
    done.parent.mkdir(parents=True, exist_ok=True)
    temporary = done.with_suffix('.tmp')
    temporary.write_text(json.dumps({"status": "complete", "value": 42}))
    temporary.replace(done)
