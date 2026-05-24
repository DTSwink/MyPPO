"""Automated close / wait / check-respawn test for the visualizer."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLL_SEC = 10


def _viz_processes() -> list[tuple[int, str]]:
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'run_visualizer|kinematic2d' } | "
                "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }",
            ],
            text=True,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError:
        return []
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid_s, _, cmd = line.partition("\t")
        rows.append((int(pid_s), cmd))
    return rows


def _close_myppo_window() -> bool:
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        title = buf.value
        if "MyPPO" in title:
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(callback, 0)
    if not found:
        return False
    user32.PostMessageW(found[0], 0x0010, 0, 0)  # WM_CLOSE
    return True


def _wait_for_exit(timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _viz_processes():
            return True
        time.sleep(0.25)
    return not _viz_processes()


def run_once(use_reload: bool) -> bool:
    cmd = [sys.executable, "run_visualizer.py"]
    if use_reload:
        cmd.append("--reload")
    proc = subprocess.Popen(cmd, cwd=ROOT)
    time.sleep(3.0)

    before = _viz_processes()
    print(f"  running PIDs: {[pid for pid, _ in before]}")

    closed = _close_myppo_window()
    if not closed:
        print("  window not found; terminating launcher")
        proc.terminate()
        proc.wait(timeout=3.0)

    if not _wait_for_exit():
        print("  FAIL: process still running after close")
        for pid, cmdline in _viz_processes():
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        return False

    print(f"  closed; waiting {POLL_SEC}s for respawn...")
    time.sleep(POLL_SEC)
    after = _viz_processes()
    if after:
        print(f"  FAIL: respawned PIDs: {[pid for pid, _ in after]}")
        for pid, cmdline in after:
            print(f"    {cmdline}")
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        return False

    print("  OK: stayed closed")
    return True


def main() -> int:
    print("=== close loop: default (no reload) ===")
    if not run_once(use_reload=False):
        return 1

    print("=== close loop: --reload ===")
    if not run_once(use_reload=True):
        return 1

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
