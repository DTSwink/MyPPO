"""Launch training + live visualizer together."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from kinematic2d.settings import ExperimentConfig, SettingsStore

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/live")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with live visualization.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fresh", action="store_true", help="Clear checkpoint directory before run.")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if args.fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    config = ExperimentConfig(
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        loss_terms_enabled=SettingsStore().loss_terms_enabled(),
    )
    SettingsStore().save_experiment_config(config)

    from kinematic2d.experiment_runner import ExperimentRunner
    from kinematic2d.visualizer import main as viz_main

    runner = ExperimentRunner(
        checkpoint_dir=config.checkpoint_dir,
        batch_size=config.batch_size,
        lr=config.lr,
        seed=config.seed,
        loss_terms_enabled=config.loss_terms_enabled,
    )

    try:
        viz_main(
            training_mode=True,
            checkpoint_dir=config.checkpoint_dir,
            experiment_runner=runner,
        )
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
