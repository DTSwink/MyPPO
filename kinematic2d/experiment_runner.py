"""Manages the background training thread and full experiment restarts."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path


class ExperimentRunner:
    def __init__(
        self,
        checkpoint_dir: Path,
        batch_size: int = 2048,
        lr: float = 3e-3,
        seed: int | None = 0,
        loss_terms_enabled: dict[str, bool] | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.loss_terms_enabled = loss_terms_enabled
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        from kinematic2d.trainer import train_loop

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=train_loop,
            kwargs={
                "checkpoint_dir": self.checkpoint_dir,
                "stop_event": self._stop_event,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "seed": self.seed,
                "loss_terms_enabled": self.loss_terms_enabled,
            },
            daemon=True,
            name="trainer",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0, *, join: bool = True) -> None:
        self._stop_event.set()
        if join and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def reset_and_relaunch(self) -> None:
        self.stop()
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self._stop_event = threading.Event()
        self.start()
