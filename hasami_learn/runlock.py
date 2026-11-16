"""Standard-library writer locks and narrowly scoped staging-file cleanup.

Locks cover recovery as well as publication. They are kernel owned, so SIGKILL
does not require a stale-PID heuristic. Never unlink a lock file: other processes
may already have opened that inode. Reentrancy is limited to one process/thread.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import re
import shutil
import stat
import threading


_held: dict[tuple[int, int, str, bool], int] = {}
RANDOM_SUFFIX = r"[a-z0-9_]{8}"
GENERATION = r"(?:iter_[0-9]{6,}|exploiter_[0-9]{6,}_[0-9]{6,}(?:_(?:defensive|aggressive|fast))?)"
PROGRESS_LOCK = ".progress.lock" if os.name == "nt" else None
# All short JSON/report publications share the same directory lock. Keep the
# Windows filename compatible with reporters from earlier versions.
FILE_LOCK = PROGRESS_LOCK


class WriterBusyError(RuntimeError):
    """A live writer owns the requested kernel lock."""


def writer_path(path: str | Path, *, allow_best_alias: bool = False) -> Path:
    """Validate every original component, then return the sole path used for I/O.

    Plain ``..`` remains supported, but no link may be hidden by normalization.
    Autoloop metadata alone may traverse its intentional ``models/best`` alias:
    a direct sibling directory is validated and returned before selecting a lock.
    This checks stable paths; callers must still hold their writer's kernel lock.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    # Check each component before collapsing a following '..'. Checking only
    # raw prefixes misses links after missing/..; normalizing up front instead
    # hides links before /... Neither may choose a different lock from the I/O.
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        if part == "..":
            current = current.parent
            continue
        current /= part
        if not current.is_symlink():
            continue
        if not (allow_best_alias and index < len(path.parts) - 1
                and current.name == "best" and current.parent.name == "models"
                and ".." not in path.parts):
            raise RuntimeError(f"writer path contains a symlink: {current}")
        reference = Path(os.readlink(current))
        if reference.is_absolute() or len(reference.parts) != 1 or reference.name == "..":
            raise RuntimeError(f"writer symlink must name a sibling directory: {current}")
        current = writer_path(current.parent / reference)
        allow_best_alias = False
    return current


def writer_directory(path: str | Path) -> Path:
    """Compatibility name for the shared writer-path validator."""
    return writer_path(path)


def plain_directory(path: str | Path) -> bool:
    """Return false if any directory component is a link or is missing."""
    try:
        path = writer_directory(path)
    except RuntimeError:
        return False
    return all(p.is_dir() and not p.is_symlink() for p in (path, *path.parents))


@contextmanager
def run_lock(directory: str | Path, filename: str | None = "run.lock", *, blocking: bool = False):
    """Lock and yield the validated directory; filename=None locks its POSIX inode."""
    directory = writer_path(directory)
    path = writer_path(directory / filename) if filename is not None else directory
    directory.mkdir(parents=True, exist_ok=True)
    if not plain_directory(directory):
        raise RuntimeError(f"writer directory contains a symlink: {directory}")
    if filename is None and os.name == "nt":
        raise RuntimeError("directory writer locks require POSIX")
    key = (os.getpid(), threading.get_ident(), str(path), filename is None)
    if key in _held:
        yield directory
        return
    if path.is_symlink():
        raise RuntimeError(f"writer lock must not be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) if filename is None else os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        expected_type = stat.S_ISDIR if filename is None else stat.S_ISREG
        if not expected_type(os.fstat(fd).st_mode):
            raise RuntimeError(f"writer lock has the wrong file type: {path}")
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise WriterBusyError(f"another writer owns {path}") from exc
            raise RuntimeError(f"another writer owns {path}") from exc
        _held[key] = fd
        try:
            yield directory
        finally:
            del _held[key]
    finally:
        os.close(fd)


def cleanup_temps(directory: str | Path, *, files=(), directories=(), links=()) -> list[dict[str, str]]:
    """Remove exact writer namespaces; the caller must hold the owning lock.

    Inspect immediate children only. Directory removal never follows links;
    symlinks are otherwise preserved unless explicitly owned (the best pointer).
    """
    try:
        directory = writer_path(directory)
    except RuntimeError:
        return []
    if not plain_directory(directory):
        return []
    removed = []
    for path in sorted(directory.iterdir()):
        if not any(re.fullmatch(pattern, path.name) for pattern in (*files, *directories, *links)):
            continue
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            patterns = links
        elif stat.S_ISREG(mode):
            patterns = files
        elif stat.S_ISDIR(mode):
            patterns = directories
        else:
            continue
        if not any(re.fullmatch(pattern, path.name) for pattern in patterns):
            continue
        if stat.S_ISDIR(mode):
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append({"path": str(path), "reason": "removed unfinished writer temporary"})
    return removed


def cleanup_progress_files(directory: str | Path) -> list[dict[str, str]]:
    """Recover dead reporters without waiting for a live file publication.

    Reporters own the directory inode only during their atomic write on POSIX,
    or a dedicated file lock on Windows. Neither path takes the training lock.
    """
    if not plain_directory(directory):
        return []
    try:
        with run_lock(directory, PROGRESS_LOCK) as directory:
            return cleanup_temps(directory, files=(rf"\.progress-{RANDOM_SUFFIX}\.tmp",))
    except WriterBusyError:
        return []


def cleanup_json_files(directory: str | Path, names, *, legacy: bool = False) -> list[dict[str, str]]:
    """Exclude both JSON writers while removing only declared output staging."""
    try:
        directory = writer_path(directory)
    except RuntimeError:
        return []
    if not plain_directory(directory):
        return []
    patterns = [rf"\.tmp_{re.escape(name)}_{RANDOM_SUFFIX}" for name in names]
    if legacy:
        patterns.append(rf"\.tmp_{RANDOM_SUFFIX}")
    # An unrelated live reporter must not prevent startup when there is no
    # JSON staging to remove. The actual deletion still rescans under the lock.
    if not any(any(re.fullmatch(pattern, path.name) for pattern in patterns)
               for path in directory.iterdir()):
        return []
    with run_lock(directory, FILE_LOCK) as directory:
        return cleanup_temps(directory, files=patterns)


def cleanup_training_files(run: str | Path) -> list[dict[str, str]]:
    """Discard uncommitted root files; retired checkpoint recovery is separate."""
    with run_lock(run) as run:
        removed = cleanup_json_files(run, ("config.json", "trainer_state.json", "smoke-levels.json"))
        removed += cleanup_temps(run, files=(rf"\.tmp_replay_{RANDOM_SUFFIX}\.npz",))
        removed += cleanup_json_files(run / "checkpoints", ("latest.json",))
        removed += cleanup_progress_files(run)
        store = run / "selfplay"
        if plain_directory(store):
            with run_lock(store, "store.lock"):
                removed += cleanup_temps(store / "games", files=(r"\.selfplay-[A-Za-z0-9_-]+\.tmp",))
        return removed


def checkpoint_boundary(name: str, iteration: int) -> None:
    """Fault-injection seam; normal training does nothing at these boundaries."""
