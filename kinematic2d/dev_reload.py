"""Auto-restart the visualizer when Python source files change."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCH_DIRS = [ROOT / "kinematic2d"]
POLL_SEC = 0.5


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
    return mtimes


def run_with_reload() -> None:
    cmd = [sys.executable, "-m", "kinematic2d.visualizer"]
    mtimes = _snapshot_mtimes()
    proc: subprocess.Popen | None = None

    try:
        while True:
            if proc is None or proc.poll() is not None:
                if proc is not None and proc.returncode not in (0, None):
                    sys.exit(proc.returncode or 1)
                proc = subprocess.Popen(cmd, cwd=ROOT)

            time.sleep(POLL_SEC)
            current = _snapshot_mtimes()
            if current != mtimes:
                mtimes = current
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                proc = None
    except KeyboardInterrupt:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=3.0)
