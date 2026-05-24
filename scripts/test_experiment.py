"""Automated experiment smoke test + timing (no pygame)."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic2d.checkpoint import load_checkpoint, load_loss_history
from kinematic2d.experiment_runner import ExperimentRunner
from kinematic2d.settings import SettingsStore


def main() -> int:
    checkpoint_dir = Path("checkpoints/_smoke_test")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    config = SettingsStore().get_experiment_config()
    stop = Event()
    runner = ExperimentRunner(
        checkpoint_dir=checkpoint_dir,
        batch_size=min(config.batch_size, 512),
        lr=config.lr,
        seed=config.seed,
    )

    t0 = time.perf_counter()
    runner.start()

    first_metrics = None
    first_trained = None
    deadline = t0 + 30.0

    while time.perf_counter() < deadline:
        if not runner.is_running:
            print("FAIL: trainer thread died")
            return 1

        points = load_loss_history(checkpoint_dir)
        if points and first_metrics is None:
            first_metrics = time.perf_counter() - t0
            print(f"first metrics point: {first_metrics:.3f}s  loss={points[0]['loss']}")

        ckpt = load_checkpoint(checkpoint_dir)
        if ckpt and int(ckpt["step"]) > 0 and first_trained is None:
            first_trained = time.perf_counter() - t0
            print(f"first trained checkpoint: {first_trained:.3f}s  step={ckpt['step']}  loss={ckpt['loss']:.6f}")

        if first_metrics is not None and first_trained is not None:
            break

        time.sleep(0.05)

    stop.set()
    runner.stop()

    if first_metrics is None:
        print("FAIL: no metrics written")
        return 1
    if first_trained is None:
        print("FAIL: training did not advance past step 0")
        return 1

    print(f"OK  metrics={first_metrics:.3f}s  trained_ckpt={first_trained:.3f}s")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
