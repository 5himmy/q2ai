#!/usr/bin/env python3
"""
train.py -- PPO training loop for Q2AI.

Usage:
    python train.py                              # single instance
    python train.py --num-envs 4                 # 4 parallel instances
    python train.py --resume checkpoints/model_100.pt

Connects to running q2ai headless clients via shared memory.
With --num-envs N, collects rollouts from N environments in parallel
for N× data collection throughput.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from model import ActorCritic
from q2ai_env import OBS_DIM, Q2AIEnv


class RolloutBuffer:
    """Fixed-size buffer for storing one rollout of PPO transitions."""

    def __init__(self, rollout_len, obs_dim=OBS_DIM):
        self.rollout_len = rollout_len
        self.obs = np.zeros((rollout_len, obs_dim), dtype=np.float32)
        self.cont_actions = np.zeros((rollout_len, 4), dtype=np.float32)
        self.jump_actions = np.zeros(rollout_len, dtype=np.float32)
        self.crouch_actions = np.zeros(rollout_len, dtype=np.float32)
        self.attack_actions = np.zeros(rollout_len, dtype=np.float32)
        self.weapon_actions = np.zeros(rollout_len, dtype=np.int64)
        self.log_probs = np.zeros(rollout_len, dtype=np.float32)
        self.rewards = np.zeros(rollout_len, dtype=np.float32)
        self.values = np.zeros(rollout_len, dtype=np.float32)
        self.dones = np.zeros(rollout_len, dtype=np.float32)
        self.returns = np.zeros(rollout_len, dtype=np.float32)
        self.advantages = np.zeros(rollout_len, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action_dict, log_prob, reward, value, done):
        """Store one transition."""
        i = self.ptr
        self.obs[i] = obs
        self.cont_actions[i] = [
            action_dict["yaw_delta"] / 10.0,
            action_dict["pitch_delta"] / 5.0,
            action_dict["forwardmove"],
            action_dict["sidemove"],
        ]
        self.jump_actions[i] = action_dict["jump"]
        self.crouch_actions[i] = action_dict["crouch"]
        self.attack_actions[i] = action_dict["attack"]
        self.weapon_actions[i] = action_dict["weapon"]
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE advantages and returns."""
        last_gae = 0.0
        for t in reversed(range(self.rollout_len)):
            if t == self.rollout_len - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * lam * non_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size, device="cpu"):
        """Yield random mini-batches as torch tensors."""
        indices = np.random.permutation(self.rollout_len)
        for start in range(0, self.rollout_len, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield {
                "obs": torch.FloatTensor(self.obs[idx]).to(device),
                "continuous": torch.FloatTensor(self.cont_actions[idx]).to(device),
                "jump": torch.FloatTensor(self.jump_actions[idx]).to(device),
                "crouch": torch.FloatTensor(self.crouch_actions[idx]).to(device),
                "attack": torch.FloatTensor(self.attack_actions[idx]).to(device),
                "weapon": torch.LongTensor(self.weapon_actions[idx]).to(device),
                "log_probs": torch.FloatTensor(self.log_probs[idx]).to(device),
                "returns": torch.FloatTensor(self.returns[idx]).to(device),
                "advantages": torch.FloatTensor(self.advantages[idx]).to(device),
            }

    def reset(self):
        self.ptr = 0


def ppo_update(model, optimizer, buffer, clip_eps, epochs, batch_size,
               entropy_coef, value_coef, max_grad_norm, device="cpu"):
    """Run PPO policy gradient updates on the rollout buffer."""
    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0,
               "clip_fraction": 0, "approx_kl": 0, "n_updates": 0}

    for _ in range(epochs):
        for batch in buffer.get_batches(batch_size, device=device):
            action = {
                "continuous": batch["continuous"],
                "jump": batch["jump"],
                "crouch": batch["crouch"],
                "attack": batch["attack"],
                "weapon": batch["weapon"],
            }

            result = model.get_action_and_value(batch["obs"], action=action)
            new_log_prob = result["log_prob"]
            entropy = result["entropy"]
            new_value = result["value"]

            advantages = batch["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_ratio = new_log_prob - batch["log_probs"]
            ratio = log_ratio.exp()

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * (new_value - batch["returns"]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy_loss.item()
            metrics["clip_fraction"] += clip_frac
            metrics["approx_kl"] += approx_kl
            metrics["n_updates"] += 1

    n = max(metrics["n_updates"], 1)
    for k in metrics:
        if k != "n_updates":
            metrics[k] /= n

    return metrics


def train(args):
    """Main training loop with multi-instance support."""
    num_envs = args.num_envs
    print(f"Q2AI Training — KaaN[V1.0]")
    print(f"Instances: {num_envs}, LR: {args.lr}, Rollout: {args.rollout_len}")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create environments
    envs = []
    for i in range(num_envs):
        instance_id = args.instance + i
        print(f"Connecting to q2ai instance {instance_id}...")
        envs.append(Q2AIEnv(instance=instance_id, obs_timeout=args.timeout))
    print(f"Connected to {num_envs} environment(s)")

    # Model
    model = ActorCritic(obs_dim=OBS_DIM).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, weights_only=True, map_location=device))
        with torch.no_grad():
            model.actor.bias[5] = min(model.actor.bias[5].item(), -10.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Total rollout size = rollout_len * num_envs (more data per iteration)
    steps_per_env = args.rollout_len
    total_rollout_size = steps_per_env * num_envs
    buffer = RolloutBuffer(total_rollout_size)

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/train_{timestamp}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "iteration", "timesteps", "episodes", "mean_reward", "mean_frags",
        "mean_length", "policy_loss", "value_loss", "entropy",
        "clip_fraction", "approx_kl", "elapsed_s",
    ])

    # Per-env episode tracking
    total_timesteps = 0
    total_episodes = 0
    episode_rewards = []
    episode_frags = []
    episode_lengths = []
    ep_reward = [0.0] * num_envs
    ep_frags = [0] * num_envs
    ep_length = [0] * num_envs

    start_time = time.time()

    # Reset all envs
    print(f"Resetting {num_envs} environment(s)...")
    obs_list = []
    for env in envs:
        obs, info = env.reset()
        obs_list.append(obs)
    print(f"Ready! Starting training loop ({num_envs}x parallel).\n")

    for iteration in range(1, args.num_rollouts + 1):
        buffer.reset()
        rollout_rewards = []

        # Collect rollout from all envs in round-robin
        for t in range(steps_per_env):
            for e in range(num_envs):
                obs = obs_list[e]
                action, log_prob, value = model.act(obs)
                next_obs, reward, terminated, truncated, info = envs[e].step(action)
                rollout_rewards.append(reward)

                buffer.add(obs, action, log_prob, reward, value, terminated)
                total_timesteps += 1
                ep_reward[e] += reward
                ep_length[e] += 1
                ep_frags[e] = info.get("frags", ep_frags[e])

                obs_list[e] = next_obs

                if terminated:
                    episode_rewards.append(ep_reward[e])
                    episode_frags.append(ep_frags[e])
                    episode_lengths.append(ep_length[e])
                    total_episodes += 1
                    ep_reward[e] = 0.0
                    ep_frags[e] = 0
                    ep_length[e] = 0
                    obs_list[e], _ = envs[e].reset()

        # Bootstrap value (use first env's obs as representative)
        with torch.no_grad():
            last_value = model.get_value(
                torch.FloatTensor(obs_list[0]).unsqueeze(0).to(device)
            ).item()

        buffer.compute_returns(last_value, gamma=args.gamma, lam=args.lam)

        # PPO update
        metrics = ppo_update(
            model, optimizer, buffer,
            clip_eps=args.clip_eps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )

        # Log
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - start_time
            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            recent_frags = episode_frags[-10:] if episode_frags else [0]
            recent_lengths = episode_lengths[-10:] if episode_lengths else [0]

            mean_r = np.mean(recent_rewards)
            mean_f = np.mean(recent_frags)
            mean_l = np.mean(recent_lengths)
            step_r = np.mean(rollout_rewards) if rollout_rewards else 0.0

            print(f"[{iteration:>6}] steps={total_timesteps:>8} eps={total_episodes:>4} | "
                  f"R={mean_r:>7.2f} step_r={step_r:>7.4f} frags={mean_f:>5.1f} deaths={total_episodes} | "
                  f"pi={metrics['policy_loss']:>7.4f} v={metrics['value_loss']:>7.4f} "
                  f"ent={metrics['entropy']:>6.3f} kl={metrics['approx_kl']:>7.4f} | "
                  f"{elapsed:>6.0f}s")

            log_writer.writerow([
                iteration, total_timesteps, total_episodes,
                f"{mean_r:.4f}", f"{mean_f:.1f}", f"{mean_l:.0f}",
                f"{metrics['policy_loss']:.6f}", f"{metrics['value_loss']:.6f}",
                f"{metrics['entropy']:.4f}", f"{metrics['clip_fraction']:.4f}",
                f"{metrics['approx_kl']:.6f}", f"{elapsed:.1f}",
            ])
            log_file.flush()

        # Save checkpoint
        if iteration % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"model_{iteration}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Checkpoint saved: {path}")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total: {total_timesteps} steps, {total_episodes} episodes, {time.time() - start_time:.0f}s")

    log_file.close()
    for env in envs:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2AI PPO Training")
    parser.add_argument("--instance", type=int, default=0, help="Starting instance ID")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--rollout-len", type=int, default=512, help="Steps per rollout per env")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per rollout")
    parser.add_argument("--num-rollouts", type=int, default=100000, help="Total rollout iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=100, help="Save every N iterations")
    parser.add_argument("--timeout", type=float, default=5.0, help="Observation timeout (seconds)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train(parser.parse_args())
