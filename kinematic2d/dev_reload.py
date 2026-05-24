"""Auto-restart the visualizer when Python source files change."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kinematic2d import viz_shutdown

ROOT = Path(__file__).resolve().parent.parent
WATCH_DIRS = [ROOT / "kinematic2d"]
POLL_SEC = 0.5
SELF_PATH = Path(__file__).resolve()
LAUNCHER_PATH = ROOT / "run_visualizer.py"


def _snapshot_mtimes() -> dict[Path, float]:
    mtimes: dict[Path, float] = {}
    for watch_dir in WATCH_DIRS:
        if not watch_dir.is_dir():
            continue
        for path in watch_dir.rglob("*.py"):
            try:
                mtimes[path] = path.stat().st_mtime
            except OSError:
                pass
    for path in (SELF_PATH, LAUNCHER_PATH):
        try:
            mtimes[path] = path.stat().st_mtime
        except OSError:
            pass
    return mtimes


def _reexec_launcher() -> None:
    extra = [arg for arg in sys.argv[1:] if arg != "--reload"]
    argv = [sys.executable, str(LAUNCHER_PATH), "--reload", *extra]
    os.execv(sys.executable, argv)


def _terminate_proc(proc: subprocess.Popen, *, force: bool = False) -> None:
    if proc.poll() is not None:
        return
    if force:
        proc.kill()
    else:
        proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3.0)


def _should_stop_after_exit(returncode: int | None) -> bool:
    user_closed = viz_shutdown.was_user_close()
    viz_shutdown.clear()
    if user_closed:
        return True
    return returncode == 0


def run_with_reload() -> None:
    cmd = [sys.executable, "-m", "kinematic2d.visualizer"]
    viz_shutdown.clear()
    mtimes = _snapshot_mtimes()
    proc = subprocess.Popen(cmd, cwd=ROOT)

    try:
        while True:
            try:
                returncode = proc.wait(timeout=POLL_SEC)
            except subprocess.TimeoutExpired:
                returncode = None

            if returncode is not None:
                if _should_stop_after_exit(returncode):
                    break
                sys.exit(returncode)

            if viz_shutdown.was_user_close():
                _terminate_proc(proc, force=True)
                viz_shutdown.clear()
                break

            current = _snapshot_mtimes()
            if current == mtimes:
                continue

            wrapper_changed = (
                current.get(SELF_PATH) != mtimes.get(SELF_PATH)
                or current.get(LAUNCHER_PATH) != mtimes.get(LAUNCHER_PATH)
            )
            mtimes = current

            if wrapper_changed:
                _terminate_proc(proc)
                _reexec_launcher()

            if viz_shutdown.was_user_close():
                _terminate_proc(proc, force=True)
                viz_shutdown.clear()
                break

            if proc.poll() is not None:
                if _should_stop_after_exit(proc.returncode):
                    break
                sys.exit(proc.returncode or 1)

            _terminate_proc(proc)
            proc = subprocess.Popen(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        _terminate_proc(proc)
        viz_shutdown.clear()
