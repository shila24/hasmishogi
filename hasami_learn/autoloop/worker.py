"""Keep the loop lock until its stage stops, including after parent SIGKILL.

The controller owns stdin's write end. EOF asks this guardian to terminate the
stage's separate process group and wait for checkpointing. The inherited lock
prevents another controller from starting a conflicting writer during cleanup.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time


def adopt_orphans():
    """Reap orphaned stage workers on Linux, including under a non-reaping PID 1."""
    if sys.platform == "linux":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
        libc.prctl.restype = ctypes.c_int
        # PR_SET_CHILD_SUBREAPER: orphaned descendants are reparented here.
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    except PermissionError:
        # An orphaned zombie group can report EPERM until init reaps it.
        # Permission denial does not prove the group is gone: retain the lock.
        return True
    return True


def reap_group(pgid):
    """Called only after Popen has collected the direct child's exit status."""
    while True:
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def main():
    grace = float(sys.argv[1])
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())

    def watch_parent():
        while os.read(0, 4096):
            pass
        stopping.set()

    adopt_orphans()
    threading.Thread(target=watch_parent, daemon=True).start()
    # Keep the guardian outside the stage group so SIGKILL can stop all stage
    # workers without killing the lock holder before it finishes cleanup.
    child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = None
    killed = False
    while True:
        code = child.poll()
        if (stopping.is_set() or code is not None) and deadline is None:
            deadline = time.monotonic() + grace
            signal_group(child.pid, signal.SIGTERM)
        if deadline is not None and not killed and time.monotonic() >= deadline:
            signal_group(child.pid, signal.SIGKILL)
            killed = True
        if code is not None:
            reap_group(child.pid)
            if not signal_group(child.pid, 0):
                return code if code >= 0 else 128 - code
        time.sleep(0.02)


if __name__ == "__main__":
    raise SystemExit(main())
