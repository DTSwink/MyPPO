"""CleanRL-style PPO trainer for the 2D kinematic controller."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from kinematic2d.action_codec import sanitize_action_torch
from kinematic2d.agent import HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM
from kinematic2d.batch_env import BatchEnv, batch_local_xy_to_global
from kinematic2d.checkpoint import append_loss_point, save_checkpoint
from kinematic2d.loss_terms import loss_term_keys, normalize_loss_coeffs, normalize_loss_terms_enabled
from kinematic2d.settings import SettingsStore
from kinematic2d.trainer import MAX_LIMB_RADIUS_M, PELVIS_MAX_DEVIATION_RAD


@dataclass
class PPOConfig:
    exp_name: str = "myppo_cleanrl"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 25_920_000
    learning_rate: float = 3e-4
    num_envs: int = 2048
    num_steps: int = 180
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    init_log_std: float = -1.6
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    checkpoint_dir: Path = Path("checkpoints/ppo")
    fresh: bool = False
    eval_interval: int = 5
    eval_envs: int = 2048
    target_loss: float = 1e-5
    plateau_patience: int = 30
    plateau_min_delta: float = 1e-4
    min_updates_before_plateau: int = 40
    stop_on_plateau: bool = False

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)


class ActorCritic(nn.Module):
    """Actor layout keeps the same fc1/fc2 shape as MLPAgent checkpoints."""

    def __init__(self, init_log_std: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.log_std = nn.Parameter(torch.full((OUTPUT_DIM,), init_log_std))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(INPUT_DIM, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        layer_init(self.fc1)
        layer_init(self.fc2, std=0.01)

    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        return sanitize_action_torch(self.fc2(torch.nn.functional.gelu(self.fc1(x))))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        probs = Normal(mean, std)
        raw_action = probs.sample() if action is None else action
        logprob = probs.log_prob(raw_action).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)
        return sanitize_action_torch(raw_action), logprob, entropy, self.get_value(x), raw_action

    def numpy_weights(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.fc1.weight.detach().T.contiguous().cpu().numpy(),
            "b1": self.fc1.bias.detach().cpu().numpy(),
            "w2": self.fc2.weight.detach().T.contiguous().cpu().numpy(),
            "b2": self.fc2.bias.detach().cpu().numpy(),
        }


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def pelvis_tilt_loss_per_env(pelvis_angles: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(torch.sin(pelvis_angles), torch.cos(pelvis_angles))
    excess = torch.relu(torch.abs(angles) - PELVIS_MAX_DEVIATION_RAD)
    return excess * excess


def limb_radius_loss_per_env(limbs: torch.Tensor, max_radius: float = MAX_LIMB_RADIUS_M) -> torch.Tensor:
    dist = torch.linalg.norm(limbs[..., :2], dim=-1)
    excess = torch.relu(dist - max_radius)
    return (excess * excess).mean(dim=-1)


def pinned_foot_velocity_loss_per_env(
    env: BatchEnv,
    frame_idx: torch.Tensor,
    current_limbs: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    root_curr = env.roots[frame_idx]
    next_idx = torch.clamp(frame_idx + 1, max=env.n_roots - 1)
    root_next = env.roots[next_idx]
    curr_left = current_limbs[:, 1, :2]
    curr_right = current_limbs[:, 2, :2]
    next_limbs = action[:, :9].reshape(env.batch_size, 3, 3)
    next_left = next_limbs[:, 1, :2]
    next_right = next_limbs[:, 2, :2]

    g_curr_left = batch_local_xy_to_global(root_curr, curr_left)
    g_curr_right = batch_local_xy_to_global(root_curr, curr_right)
    g_next_left = batch_local_xy_to_global(root_next, next_left)
    g_next_right = batch_local_xy_to_global(root_next, next_right)

    vel_left_sq = ((g_next_left - g_curr_left) ** 2).sum(dim=-1)
    vel_right_sq = ((g_next_right - g_curr_right) ** 2).sum(dim=-1)
    left_pinned = action[:, 9] <= action[:, 10]
    return torch.where(left_pinned, vel_left_sq, vel_right_sq)


def unpinned_foot_stride_loss_per_env(
    env: BatchEnv,
    frame_idx: torch.Tensor,
    current_limbs: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    root_curr = env.roots[frame_idx]
    next_idx = torch.clamp(frame_idx + 1, max=env.n_roots - 1)
    root_next = env.roots[next_idx]
    target_disp = 2.0 * (root_next[:, :2] - root_curr[:, :2])
    curr_left = current_limbs[:, 1, :2]
    curr_right = current_limbs[:, 2, :2]
    next_limbs = action[:, :9].reshape(env.batch_size, 3, 3)
    next_left = next_limbs[:, 1, :2]
    next_right = next_limbs[:, 2, :2]

    g_curr_left = batch_local_xy_to_global(root_curr, curr_left)
    g_curr_right = batch_local_xy_to_global(root_curr, curr_right)
    g_next_left = batch_local_xy_to_global(root_next, next_left)
    g_next_right = batch_local_xy_to_global(root_next, next_right)

    err_left_sq = ((g_next_left - g_curr_left - target_disp) ** 2).sum(dim=-1)
    err_right_sq = ((g_next_right - g_curr_right - target_disp) ** 2).sum(dim=-1)
    left_pinned = action[:, 9] <= action[:, 10]
    return torch.where(left_pinned, err_right_sq, err_left_sq)


def weighted_loss_per_env(
    env: BatchEnv,
    frame_idx: torch.Tensor,
    current_limbs: torch.Tensor,
    action: torch.Tensor,
    enabled: dict[str, bool],
    coeffs: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    limbs = action[:, :9].reshape(env.batch_size, 3, 3)
    raw_terms = {
        "pelvis_tilt": pelvis_tilt_loss_per_env(limbs[:, 0, 2]),
        "foot_pin": pinned_foot_velocity_loss_per_env(env, frame_idx, current_limbs, action),
        "limb_radius": limb_radius_loss_per_env(limbs),
        "foot_stride": unpinned_foot_stride_loss_per_env(env, frame_idx, current_limbs, action),
    }
    weighted_terms: dict[str, torch.Tensor] = {}
    total = torch.zeros(env.batch_size, device=action.device)
    for key in loss_term_keys():
        if enabled.get(key, True):
            weighted_terms[key] = coeffs[key] * raw_terms[key]
            total = total + weighted_terms[key]
        else:
            weighted_terms[key] = torch.zeros_like(total)
    return total, weighted_terms


@torch.no_grad()
def evaluate_policy(
    agent: ActorCritic,
    device: torch.device,
    enabled: dict[str, bool],
    coeffs: dict[str, float],
    num_envs: int,
    num_steps: int,
) -> tuple[float, dict[str, float]]:
    env = BatchEnv(batch_size=num_envs, device=device)
    env.reset_all()
    term_sums = {key: 0.0 for key in loss_term_keys()}
    total_sum = 0.0
    count = 0
    for _ in range(num_steps):
        obs = env.build_obs()
        action = agent.actor_mean(obs)
        losses, terms = weighted_loss_per_env(env, env.frame_idx, env.current_limbs, action, enabled, coeffs)
        total_sum += float(losses.mean().item())
        for key in term_sums:
            term_sums[key] += float(terms[key].mean().item())
        env.step(action)
        count += 1
    n = max(count, 1)
    return total_sum / n, {key: value / n for key, value in term_sums.items()}


def train(config: PPOConfig) -> dict[str, float | int | str]:
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not config.torch_deterministic

    settings = SettingsStore()
    enabled = normalize_loss_terms_enabled(settings.loss_terms_enabled())
    coeffs = normalize_loss_coeffs(settings.loss_coeffs())
    active_keys = [key for key in loss_term_keys() if enabled.get(key, True)]
    if not active_keys:
        raise ValueError("At least one loss term must be enabled for PPO reward.")

    if config.fresh and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (config.checkpoint_dir / "ppo_config.json").write_text(
        json.dumps({**asdict(config), "checkpoint_dir": str(config.checkpoint_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )

    env = BatchEnv(batch_size=config.num_envs, device=device)
    agent = ActorCritic(config.init_log_std).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs = torch.zeros((config.num_steps, config.num_envs, INPUT_DIM), device=device)
    raw_actions = torch.zeros((config.num_steps, config.num_envs, OUTPUT_DIM), device=device)
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    best_eval_loss = float("inf")
    best_update = 0
    plateau_clock = 0
    status = "max_updates"

    save_checkpoint(config.checkpoint_dir, **agent.numpy_weights(), step=0, loss=float("nan"))

    for update in range(1, config.num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / config.num_updates
            optimizer.param_groups[0]["lr"] = frac * config.learning_rate

        env.reset_all()
        next_obs = env.build_obs()
        next_done = torch.zeros(config.num_envs, device=device)
        rollout_loss_sum = 0.0
        rollout_terms = {key: 0.0 for key in loss_term_keys()}

        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, raw_action = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            losses, terms = weighted_loss_per_env(env, env.frame_idx, env.current_limbs, action, enabled, coeffs)
            rewards[step] = -losses
            raw_actions[step] = raw_action
            logprobs[step] = logprob
            rollout_loss_sum += float(losses.mean().item())
            for key in rollout_terms:
                rollout_terms[key] += float(terms[key].mean().item())

            env.step(action)
            next_obs = env.build_obs()

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = torch.zeros(config.num_envs, device=device)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, INPUT_DIM))
        b_logprobs = logprobs.reshape(-1)
        b_actions = raw_actions.reshape((-1, OUTPUT_DIM))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for _epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
                agent.log_std.data.clamp_(config.min_log_std, config.max_log_std)

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        mean_rollout_loss = rollout_loss_sum / config.num_steps
        mean_rollout_terms = {key: value / config.num_steps for key, value in rollout_terms.items()}
        sps = int(global_step / max(time.time() - start_time, 1e-6))

        should_eval = update == 1 or update % config.eval_interval == 0 or update == config.num_updates
        if should_eval:
            eval_loss, eval_terms = evaluate_policy(
                agent,
                device,
                enabled,
                coeffs,
                num_envs=config.eval_envs,
                num_steps=config.num_steps,
            )
            save_checkpoint(config.checkpoint_dir, **agent.numpy_weights(), step=global_step, loss=eval_loss)
            append_loss_point(config.checkpoint_dir, step=global_step, loss=eval_loss, terms=eval_terms)

            improved = eval_loss < best_eval_loss - config.plateau_min_delta
            if improved:
                best_eval_loss = eval_loss
                best_update = update
                plateau_clock = 0
            else:
                plateau_clock += 1

            print(
                f"update={update}/{config.num_updates} step={global_step} "
                f"rollout_loss={mean_rollout_loss:.6g} eval_loss={eval_loss:.6g} "
                f"best={best_eval_loss:.6g}@{best_update} sps={sps} "
                f"kl={float(approx_kl):.4g} clipfrac={np.mean(clipfracs):.3f}"
            )

            if eval_loss <= config.target_loss:
                status = "target_loss"
                break
            if (
                config.stop_on_plateau
                and update >= config.min_updates_before_plateau
                and plateau_clock >= config.plateau_patience
            ):
                status = "plateau"
                break
        else:
            print(
                f"update={update}/{config.num_updates} step={global_step} "
                f"rollout_loss={mean_rollout_loss:.6g} sps={sps}"
            )

        del mean_rollout_terms

    result: dict[str, float | int | str] = {
        "status": status,
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "best_update": best_update,
    }
    (config.checkpoint_dir / "ppo_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train MyPPO with a CleanRL-style PPO loop.")
    parser.add_argument("--exp-name", type=str, default=PPOConfig.exp_name)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=PPOConfig.cuda)
    parser.add_argument("--total-timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--num-steps", type=int, default=PPOConfig.num_steps)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=PPOConfig.anneal_lr)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--num-minibatches", type=int, default=PPOConfig.num_minibatches)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--clip-coef", type=float, default=PPOConfig.clip_coef)
    parser.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=PPOConfig.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    parser.add_argument("--target-kl", type=float, default=PPOConfig.target_kl)
    parser.add_argument("--init-log-std", type=float, default=PPOConfig.init_log_std)
    parser.add_argument("--checkpoint-dir", type=Path, default=PPOConfig.checkpoint_dir)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=PPOConfig.eval_interval)
    parser.add_argument("--eval-envs", type=int, default=PPOConfig.eval_envs)
    parser.add_argument("--target-loss", type=float, default=PPOConfig.target_loss)
    parser.add_argument("--stop-on-plateau", action="store_true")
    args = parser.parse_args()
    return PPOConfig(**vars(args))


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
