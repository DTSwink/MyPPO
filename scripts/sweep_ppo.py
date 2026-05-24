"""Autonomous PPO hyperparameter sweep for Colab/GPU runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic2d.ppo_cleanrl import PPOConfig, train


def candidate_configs(args: argparse.Namespace) -> list[PPOConfig]:
    base = PPOConfig(
        total_timesteps=args.timesteps_per_trial,
        num_envs=args.num_envs,
        num_steps=180,
        eval_envs=args.eval_envs,
        target_loss=args.target_loss,
        stop_on_plateau=True,
        fresh=True,
    )
    candidates = [
        {"learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.01, "init_log_std": -1.6},
        {"learning_rate": 1e-4, "gamma": 0.997, "gae_lambda": 0.97, "ent_coef": 0.02, "init_log_std": -1.2},
        {"learning_rate": 5e-4, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.005, "init_log_std": -1.8},
        {"learning_rate": 2e-4, "gamma": 0.999, "gae_lambda": 0.98, "ent_coef": 0.01, "init_log_std": -1.4},
        {"learning_rate": 7e-5, "gamma": 0.999, "gae_lambda": 0.98, "ent_coef": 0.03, "init_log_std": -1.0},
    ]
    configs = []
    for trial, overrides in enumerate(candidates[: args.max_trials], start=1):
        cfg = replace(
            base,
            seed=args.seed + trial - 1,
            checkpoint_dir=args.output_dir / f"trial_{trial:02d}",
            **overrides,
        )
        configs.append(cfg)
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO trials until one reaches target loss.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/ppo_sweep"))
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--timesteps-per-trial", type=int, default=25_920_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--eval-envs", type=int, default=2048)
    parser.add_argument("--target-loss", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sweep_summary.jsonl"
    best: dict | None = None

    for trial, config in enumerate(candidate_configs(args), start=1):
        print(f"=== PPO trial {trial}: {config.checkpoint_dir} ===")
        print(json.dumps({**asdict(config), "checkpoint_dir": str(config.checkpoint_dir)}, indent=2))
        t0 = time.time()
        result = train(config)
        row = {
            "trial": trial,
            "seconds": time.time() - t0,
            "config": {**asdict(config), "checkpoint_dir": str(config.checkpoint_dir)},
            "result": result,
        }
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if best is None or float(result["best_eval_loss"]) < float(best["result"]["best_eval_loss"]):
            best = row

        if result["status"] == "target_loss":
            print("Target loss reached; stopping sweep.")
            return 0

    print("Sweep finished without hitting target loss.")
    if best is not None:
        print("Best trial:")
        print(json.dumps(best, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
