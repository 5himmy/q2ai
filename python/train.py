#!/usr/bin/env python3
"""
train.py -- PPO training loop for Q2AI with LSTM memory.

Usage:
    python train.py                              # single instance
    python train.py --num-envs 4                 # 4 parallel instances
    python train.py --resume checkpoints/model_100.pt
    python train.py --num-envs 200 --seq-len 32  # LSTM with 32-step chunks

Connects to running q2ai headless clients via shared memory.
With --num-envs N, collects rollouts from N environments in parallel
for N× data collection throughput.

LSTM hidden states are maintained per-env during rollout and saved at
chunk boundaries for truncated BPTT during PPO updates.
"""

import argparse
import csv
import os
import socket
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from model import ActorCritic
from q2ai_env import OBS_DIM, Q2AIEnv

REWARD_SCALE = 5.0   # Scale up small rewards (~-0.1) so value function learns faster
BASE_PORT = 27910
RCON_PASSWORD = "q2ai"

# Weapon action IDs (1-indexed, 0=no switch): index in WEAPON_INDICES + 1
WEAPON_IDS = {
    "none": None,
    "rl": 7,   # rocket launcher (WEAPON_INDICES index 6)
    "rg": 9,   # railgun (WEAPON_INDICES index 8)
}


def rcon_query(port, command):
    """Send rcon command to q2pro server and return response."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        packet = b"\xff\xff\xff\xff" + f"rcon {RCON_PASSWORD} {command}\n".encode()
        sock.sendto(packet, ("127.0.0.1", port))
        data = sock.recv(4096)
        sock.close()
        text = data[4:].decode("utf-8", errors="replace")
        if text.startswith("print\n"):
            text = text[6:]
        return text.strip()
    except Exception:
        return None


class AdaptiveTuner:
    """Adjusts entropy_coef, LR, reward_scale, and bot skill during training."""

    def __init__(self, initial_entropy_coef, initial_lr, initial_reward_scale,
                 num_rollouts, num_envs, target_entropy=5.0, initial_bot_skill=0,
                 aim_assist=True, initial_assist_weight=1.0):
        self.entropy_coef = initial_entropy_coef
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.reward_scale = initial_reward_scale
        self.num_rollouts = num_rollouts
        self.num_envs = num_envs
        self.target_entropy = target_entropy
        self.bot_skill = initial_bot_skill

        # Aim assist curriculum
        self.aim_assist = aim_assist
        self.assist_weight = initial_assist_weight if aim_assist else 0.0

        # Rolling K/D tracking (last 10 logged iterations)
        self.kd_history = deque(maxlen=10)

    def update(self, iteration, metrics, rollout_frags, rollout_deaths):
        """Adjust hyperparameters based on recent metrics. Call after each PPO update."""

        # 1. Adaptive entropy coefficient
        ent = metrics["entropy"]
        if ent > self.target_entropy:
            self.entropy_coef *= 0.99
        elif ent < self.target_entropy - 1.0:
            self.entropy_coef *= 1.002
        self.entropy_coef = max(0.0005, min(0.02, self.entropy_coef))

        # 2. Linear LR decay
        frac = 1.0 - (iteration - 1) / self.num_rollouts
        self.lr = self.initial_lr * frac

        # 3. Reward scale (value-loss based)
        vl = metrics["value_loss"]
        if vl > 10.0:
            self.reward_scale *= 0.95
        elif vl < 0.1 and iteration > 20:
            self.reward_scale *= 1.05
        self.reward_scale = max(1.0, min(20.0, self.reward_scale))

        # 4. Curriculum — bot skill via rcon
        kd = rollout_frags / max(rollout_deaths, 1)
        self.kd_history.append(kd)
        if len(self.kd_history) >= 10:
            rolling_kd = sum(self.kd_history) / len(self.kd_history)
            if rolling_kd > 0.5 and self.bot_skill < 10:
                self.bot_skill += 1
                print(f"  [TUNER] Graduating bot skill to {self.bot_skill} (rolling K/D={rolling_kd:.2f})")
                for port in range(BASE_PORT, BASE_PORT + self.num_envs):
                    rcon_query(port, f"g_bot_skill {self.bot_skill}")
                self.kd_history.clear()
            elif rolling_kd < 0.05 and self.bot_skill > 0:
                self.bot_skill -= 1
                print(f"  [TUNER] Dropping bot skill to {self.bot_skill} (rolling K/D={rolling_kd:.2f})")
                for port in range(BASE_PORT, BASE_PORT + self.num_envs):
                    rcon_query(port, f"g_bot_skill {self.bot_skill}")
                self.kd_history.clear()

        # 5. Aim assist decay — one-way curriculum (never re-enables once reduced)
        if self.aim_assist and self.assist_weight > 0 and len(self.kd_history) >= 5:
            recent_kd = sum(list(self.kd_history)[-5:]) / 5
            if recent_kd > 0.5:
                self.assist_weight = 0.0
                print(f"  [TUNER] Aim assist disabled (recent K/D={recent_kd:.2f} > 0.5)")
            elif recent_kd > 0.3:
                old_w = self.assist_weight
                self.assist_weight = max(0.0, self.assist_weight * 0.95)
                if abs(old_w - self.assist_weight) > 0.01:
                    print(f"  [TUNER] Aim assist decayed {old_w:.3f} → {self.assist_weight:.3f} (recent K/D={recent_kd:.2f})")

        return {
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "reward_scale": self.reward_scale,
            "bot_skill": self.bot_skill,
            "assist_weight": self.assist_weight,
        }


class RunningNormalizer:
    """Running mean/std observation normalizer (Welford's algorithm)."""

    def __init__(self, shape, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch):
        """Update running stats with a batch of observations."""
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, obs):
        """Normalize observation(s) using running stats."""
        return np.clip(
            (np.asarray(obs) - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self.clip, self.clip
        ).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        saved_mean = d["mean"]
        saved_var = d["var"]
        cur_dim = self.mean.shape[0]
        saved_dim = saved_mean.shape[0]
        if saved_dim != cur_dim:
            # Obs dim changed — copy what fits, leave new dims at defaults
            copy_len = min(saved_dim, cur_dim)
            self.mean[:copy_len] = saved_mean[:copy_len]
            self.var[:copy_len] = saved_var[:copy_len]
            print(f"  Normalizer resized: {saved_dim} → {cur_dim} (new dims init fresh)")
        else:
            self.mean = saved_mean
            self.var = saved_var
        self.count = d["count"]


class RolloutBuffer:
    """Per-env rollout buffer for LSTM-based PPO.

    Stores data as (num_envs, steps_per_env, ...) so GAE is computed
    correctly per environment (no cross-env contamination).
    """

    def __init__(self, num_envs, steps_per_env, seq_len, obs_dim=OBS_DIM, hidden_size=256):
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_chunks_per_env = steps_per_env // seq_len

        # Per-env, per-step storage
        self.obs = np.zeros((num_envs, steps_per_env, obs_dim), dtype=np.float32)
        self.cont_actions = np.zeros((num_envs, steps_per_env, 4), dtype=np.float32)
        self.jump_actions = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.crouch_actions = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.attack_actions = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.weapon_actions = np.zeros((num_envs, steps_per_env), dtype=np.int64)
        self.log_probs = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.rewards = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.values = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.dones = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.returns = np.zeros((num_envs, steps_per_env), dtype=np.float32)
        self.advantages = np.zeros((num_envs, steps_per_env), dtype=np.float32)

        # Aim supervision targets: [yaw_target, pitch_target] + mask
        self.aim_targets = np.zeros((num_envs, steps_per_env, 2), dtype=np.float32)
        self.aim_masks = np.zeros((num_envs, steps_per_env), dtype=np.float32)

        # Hidden states at chunk boundaries (for LSTM replay during PPO)
        # Shape: (num_envs, num_chunks, 1, 1, hidden_size) for each of h_a, c_a, h_c, c_c
        nc = self.num_chunks_per_env
        self.chunk_h_actor = np.zeros((num_envs, nc, hidden_size), dtype=np.float32)
        self.chunk_c_actor = np.zeros((num_envs, nc, hidden_size), dtype=np.float32)
        self.chunk_h_critic = np.zeros((num_envs, nc, hidden_size), dtype=np.float32)
        self.chunk_c_critic = np.zeros((num_envs, nc, hidden_size), dtype=np.float32)

        self.ptrs = np.zeros(num_envs, dtype=np.int64)  # per-env step pointer

    def add(self, env_id, obs, action_dict, log_prob, reward, value, done,
            aim_target=None, aim_mask=0.0):
        """Store one transition for a specific env."""
        t = self.ptrs[env_id]
        self.obs[env_id, t] = obs
        self.cont_actions[env_id, t] = action_dict["_raw_continuous"]
        self.jump_actions[env_id, t] = action_dict["jump"]
        self.crouch_actions[env_id, t] = action_dict["crouch"]
        self.attack_actions[env_id, t] = action_dict["attack"]
        self.weapon_actions[env_id, t] = action_dict["weapon"]
        self.log_probs[env_id, t] = log_prob
        self.rewards[env_id, t] = reward
        self.values[env_id, t] = value
        self.dones[env_id, t] = float(done)
        if aim_target is not None:
            self.aim_targets[env_id, t] = aim_target
        self.aim_masks[env_id, t] = aim_mask
        self.ptrs[env_id] = t + 1

    def save_hidden(self, env_id, chunk_idx, hidden_state):
        """Save LSTM hidden state at a chunk boundary.

        Args:
            hidden_state: (h_a, c_a, h_c, c_c) each (1, 1, hidden_size) tensor
        """
        h_a, c_a, h_c, c_c = hidden_state
        self.chunk_h_actor[env_id, chunk_idx] = h_a.squeeze().cpu().numpy()
        self.chunk_c_actor[env_id, chunk_idx] = c_a.squeeze().cpu().numpy()
        self.chunk_h_critic[env_id, chunk_idx] = h_c.squeeze().cpu().numpy()
        self.chunk_c_critic[env_id, chunk_idx] = c_c.squeeze().cpu().numpy()

    def compute_returns(self, last_values, gamma=0.99, lam=0.95):
        """Compute GAE advantages and returns per-env.

        Args:
            last_values: (num_envs,) array — bootstrap value for each env
        """
        for e in range(self.num_envs):
            last_gae = 0.0
            for t in reversed(range(self.steps_per_env)):
                if t == self.steps_per_env - 1:
                    next_value = last_values[e]
                else:
                    next_value = self.values[e, t + 1]
                non_terminal = 1.0 - self.dones[e, t]
                delta = self.rewards[e, t] + gamma * next_value * non_terminal - self.values[e, t]
                self.advantages[e, t] = last_gae = delta + gamma * lam * non_terminal * last_gae
            self.returns[e] = self.advantages[e] + self.values[e]

    def get_sequence_batches(self, batch_chunks, device="cpu"):
        """Yield mini-batches of sequence chunks for LSTM PPO.

        Each env's steps are split into (steps_per_env / seq_len) chunks.
        Chunks are shuffled and yielded in mini-batches of batch_chunks.

        Yields dicts with:
            obs: (batch_chunks, seq_len, obs_dim)
            continuous, jump, crouch, attack: (batch_chunks, seq_len, ...)
            weapon: (batch_chunks, seq_len)
            log_probs, returns, advantages: (batch_chunks, seq_len)
            h_actor, c_actor, h_critic, c_critic: (1, batch_chunks, hidden_size)
        """
        S = self.seq_len
        nc = self.num_chunks_per_env

        # Build list of (env_id, chunk_idx) pairs
        chunk_indices = []
        for e in range(self.num_envs):
            for c in range(nc):
                chunk_indices.append((e, c))

        # Shuffle
        perm = np.random.permutation(len(chunk_indices))

        for start in range(0, len(chunk_indices), batch_chunks):
            end = min(start + batch_chunks, len(chunk_indices))
            batch_idx = perm[start:end]
            B = len(batch_idx)

            # Gather chunks
            obs_batch = np.zeros((B, S, self.obs.shape[2]), dtype=np.float32)
            cont_batch = np.zeros((B, S, 4), dtype=np.float32)
            jump_batch = np.zeros((B, S), dtype=np.float32)
            crouch_batch = np.zeros((B, S), dtype=np.float32)
            attack_batch = np.zeros((B, S), dtype=np.float32)
            weapon_batch = np.zeros((B, S), dtype=np.int64)
            logp_batch = np.zeros((B, S), dtype=np.float32)
            ret_batch = np.zeros((B, S), dtype=np.float32)
            adv_batch = np.zeros((B, S), dtype=np.float32)
            aim_t_batch = np.zeros((B, S, 2), dtype=np.float32)
            aim_m_batch = np.zeros((B, S), dtype=np.float32)
            h_a_batch = np.zeros((B, self.hidden_size), dtype=np.float32)
            c_a_batch = np.zeros((B, self.hidden_size), dtype=np.float32)
            h_c_batch = np.zeros((B, self.hidden_size), dtype=np.float32)
            c_c_batch = np.zeros((B, self.hidden_size), dtype=np.float32)

            for i, idx in enumerate(batch_idx):
                e, c = chunk_indices[idx]
                t0 = c * S
                t1 = t0 + S
                obs_batch[i] = self.obs[e, t0:t1]
                cont_batch[i] = self.cont_actions[e, t0:t1]
                jump_batch[i] = self.jump_actions[e, t0:t1]
                crouch_batch[i] = self.crouch_actions[e, t0:t1]
                attack_batch[i] = self.attack_actions[e, t0:t1]
                weapon_batch[i] = self.weapon_actions[e, t0:t1]
                logp_batch[i] = self.log_probs[e, t0:t1]
                ret_batch[i] = self.returns[e, t0:t1]
                adv_batch[i] = self.advantages[e, t0:t1]
                aim_t_batch[i] = self.aim_targets[e, t0:t1]
                aim_m_batch[i] = self.aim_masks[e, t0:t1]
                h_a_batch[i] = self.chunk_h_actor[e, c]
                c_a_batch[i] = self.chunk_c_actor[e, c]
                h_c_batch[i] = self.chunk_h_critic[e, c]
                c_c_batch[i] = self.chunk_c_critic[e, c]

            yield {
                "obs": torch.FloatTensor(obs_batch).to(device),
                "continuous": torch.FloatTensor(cont_batch).to(device),
                "jump": torch.FloatTensor(jump_batch).to(device),
                "crouch": torch.FloatTensor(crouch_batch).to(device),
                "attack": torch.FloatTensor(attack_batch).to(device),
                "weapon": torch.LongTensor(weapon_batch).to(device),
                "log_probs": torch.FloatTensor(logp_batch).to(device),
                "returns": torch.FloatTensor(ret_batch).to(device),
                "advantages": torch.FloatTensor(adv_batch).to(device),
                "aim_targets": torch.FloatTensor(aim_t_batch).to(device),
                "aim_masks": torch.FloatTensor(aim_m_batch).to(device),
                # Hidden states: (1, B, hidden) for LSTM
                "h_actor": torch.FloatTensor(h_a_batch).unsqueeze(0).to(device),
                "c_actor": torch.FloatTensor(c_a_batch).unsqueeze(0).to(device),
                "h_critic": torch.FloatTensor(h_c_batch).unsqueeze(0).to(device),
                "c_critic": torch.FloatTensor(c_c_batch).unsqueeze(0).to(device),
            }

    def reset(self):
        self.ptrs[:] = 0


def ppo_update(model, optimizer, buffer, clip_eps, epochs, batch_chunks,
               entropy_coef, value_coef, max_grad_norm, target_kl, aim_coef=1.0, device="cpu",
               disable_crouch=False, fixed_weapon=None):
    """Run PPO updates on sequence chunks with LSTM unrolling."""
    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0,
               "clip_fraction": 0, "approx_kl": 0, "aim_loss": 0, "n_updates": 0}

    early_stop = False
    for epoch in range(epochs):
        if early_stop:
            break
        for batch in buffer.get_sequence_batches(batch_chunks, device=device):
            # batch["obs"] is (B, seq_len, obs_dim) — full sequence chunk
            action = {
                "continuous": batch["continuous"],   # (B, seq_len, 4)
                "jump": batch["jump"],               # (B, seq_len)
                "crouch": batch["crouch"],
                "attack": batch["attack"],
                "weapon": batch["weapon"],
            }

            # Forward through LSTM with saved initial hidden states
            hidden_actor = (batch["h_actor"], batch["c_actor"])
            hidden_critic = (batch["h_critic"], batch["c_critic"])

            result = model.get_action_and_value(
                batch["obs"], hidden_actor, hidden_critic, action=action,
                disable_crouch=disable_crouch, fixed_weapon=fixed_weapon,
            )

            # result tensors are (B, seq_len) — flatten for loss computation
            new_log_prob = result["log_prob"].reshape(-1)
            entropy = result["entropy"].reshape(-1)
            new_value = result["value"].reshape(-1)

            old_log_prob = batch["log_probs"].reshape(-1)
            returns = batch["returns"].reshape(-1)
            advantages = batch["advantages"].reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if metrics["n_updates"] == 0:
                    print(f"    batch0: ratio_mean={ratio.mean().item():.4f}, ratio_std={ratio.std().item():.4f}, "
                          f"kl0={approx_kl:.4f}, logprob_diff={log_ratio.abs().mean().item():.4f}")
            if target_kl > 0 and metrics["n_updates"] >= 1 and approx_kl > target_kl:
                early_stop = True
                break

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * (new_value - returns).pow(2).mean()
            entropy_loss = entropy.mean()

            # Auxiliary aim supervision: teach yaw/pitch to match ideal aim deltas
            aim_loss = torch.tensor(0.0, device=device)
            if aim_coef > 0:
                cont_means = result["cont_means"].reshape(-1, 4)  # (B*S, 4)
                aim_targets = batch["aim_targets"].reshape(-1, 2)  # (B*S, 2)
                aim_masks = batch["aim_masks"].reshape(-1)          # (B*S,)
                # MSE on yaw and pitch means, masked to frames with visible enemy
                aim_err = (cont_means[..., :2] - aim_targets).pow(2).sum(dim=-1)  # (B*S,)
                mask_sum = aim_masks.sum().clamp(min=1.0)
                aim_loss = (aim_err * aim_masks).sum() / mask_sum

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss + aim_coef * aim_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy_loss.item()
            metrics["aim_loss"] += aim_loss.item()
            metrics["clip_fraction"] += clip_frac
            metrics["approx_kl"] += approx_kl
            metrics["n_updates"] += 1

    n = max(metrics["n_updates"], 1)
    for k in metrics:
        if k != "n_updates":
            metrics[k] /= n
    print(f"  [PPO] n_updates={metrics['n_updates']:.0f}, avg_kl={metrics['approx_kl']:.4f}, "
          f"pi={metrics['policy_loss']:.4f}, aim={metrics['aim_loss']:.4f}")

    return metrics


def train(args):
    """Main training loop with LSTM memory and multi-instance support."""
    num_envs = args.num_envs
    seq_len = args.seq_len
    steps_per_env = args.rollout_len

    # Ensure rollout_len is divisible by seq_len
    assert steps_per_env % seq_len == 0, \
        f"rollout_len ({steps_per_env}) must be divisible by seq_len ({seq_len})"

    num_chunks_per_env = steps_per_env // seq_len
    total_chunks = num_envs * num_chunks_per_env

    # Resolve fixed weapon ID
    fixed_weapon = WEAPON_IDS.get(args.single_weapon, None)
    disable_crouch = args.disable_crouch

    print(f"Q2AI Training — KaaN[V3.1]")
    print(f"Instances: {num_envs}, LR: {args.lr}, Rollout: {steps_per_env}, "
          f"Seq: {seq_len}, Reward scale: {REWARD_SCALE}x, Target KL: {args.target_kl}")
    print(f"Chunks per env: {num_chunks_per_env}, Total chunks: {total_chunks}, "
          f"Batch chunks: {args.batch_chunks}")
    if disable_crouch:
        print("  Action: crouch DISABLED (always 0, excluded from entropy)")
    if fixed_weapon is not None:
        print(f"  Action: weapon FIXED to {args.single_weapon} (id={fixed_weapon}, excluded from entropy)")
        # Send rcon give command to all servers so weapon is available on spawn
        give_cmds = {
            "rl": ["sv_cheats 1", "give rocket_launcher", "give rockets 200"],
            "rg": [],  # instagib mode gives railgun automatically
        }
        for cmd in give_cmds.get(args.single_weapon, []):
            for port in range(BASE_PORT, BASE_PORT + num_envs):
                rcon_query(port, cmd)

    # Adaptive hyperparameter tuner
    tuner = AdaptiveTuner(
        initial_entropy_coef=args.entropy_coef,
        initial_lr=args.lr,
        initial_reward_scale=REWARD_SCALE,
        num_rollouts=args.num_rollouts,
        num_envs=num_envs,
        initial_bot_skill=args.bot_skill,
        aim_assist=args.aim_assist,
        initial_assist_weight=args.aim_assist_weight,
    )

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
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)

        # Support both old (state_dict only) and new (dict with normalizer) formats
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Handle old checkpoint formats → V2.5 grouped encoder

        if "trunk.0.weight" in state_dict:
            # V1.x shared-trunk MLP → V2.5: skip encoder, keep heads
            print("  Converting V1.x shared-trunk model to V2.5...")
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith("trunk."):
                    print(f"    Skipping {k} (old trunk → grouped encoders init fresh)")
                elif k == "actor.weight" and v.shape[1] != 256:
                    print(f"    Skipping {k} (shape mismatch: {v.shape})")
                elif k == "critic.weight" and v.shape[1] != 256:
                    print(f"    Skipping {k} (shape mismatch: {v.shape})")
                else:
                    new_sd[k] = v
            state_dict = new_sd

        elif "actor_trunk.0.weight" in state_dict:
            # V1.x separate-trunk MLP → V2.5: skip encoder, keep heads
            print("  Converting V1.x separate-trunk MLP to V2.5...")
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith("actor_trunk.") or k.startswith("critic_trunk."):
                    print(f"    Skipping {k} (old trunk → grouped encoders init fresh)")
                elif k == "actor.weight" and v.shape[1] != 256:
                    print(f"    Skipping {k} (shape mismatch: {v.shape})")
                elif k == "critic.weight" and v.shape[1] != 256:
                    print(f"    Skipping {k} (shape mismatch: {v.shape})")
                else:
                    new_sd[k] = v
            state_dict = new_sd

        elif "actor_encoder.0.weight" in state_dict and "self_encoder.0.weight" not in state_dict:
            # V2.0 LSTM (actor_encoder/critic_encoder 91→256) → V2.5 grouped encoders
            # LSTM weights transfer directly (same 256→256)
            # Actor/critic heads transfer directly (same 256→18 / 256→1)
            # Old encoder weights can't map to new group encoders (different input dims)
            print("  Converting V2.0 LSTM checkpoint to V2.5 grouped encoder...")
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith("actor_encoder.") or k.startswith("critic_encoder."):
                    print(f"    Skipping {k} (old 91→256 encoder → grouped encoders init fresh)")
                else:
                    new_sd[k] = v
            state_dict = new_sd

        # Handle any remaining size mismatches (e.g. fusion layer after obs dim change)
        model_sd = model.state_dict()
        keys_to_skip = []
        for k, v in state_dict.items():
            if k in model_sd and v.shape != model_sd[k].shape:
                print(f"    Skipping {k} (shape mismatch: checkpoint {v.shape} vs model {model_sd[k].shape})")
                keys_to_skip.append(k)
        for k in keys_to_skip:
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)

        # Load normalizer if present
        if isinstance(ckpt, dict) and "normalizer" in ckpt:
            print("  Loaded observation normalizer state")

        print(f"  Attack bias: {model.actor.bias[6].item():.2f}, Crouch bias: {model.actor.bias[5].item():.2f}")
        print(f"  Log_std: {model.action_log_std.data.tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Observation normalizer
    obs_normalizer = RunningNormalizer(OBS_DIM)
    if args.resume and isinstance(ckpt, dict) and "normalizer" in ckpt:
        obs_normalizer.load_state_dict(ckpt["normalizer"])

    # Buffer: per-env storage with sequence chunk support
    buffer = RolloutBuffer(num_envs, steps_per_env, seq_len)

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/train_{timestamp}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "iteration", "timesteps", "episodes", "mean_reward", "step_reward",
        "frags", "deaths", "mean_length", "policy_loss", "value_loss",
        "entropy", "clip_fraction", "approx_kl", "aim_loss", "elapsed_s",
        "entropy_coef", "lr", "reward_scale", "bot_skill", "assist_weight",
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
    prev_frags = [0] * num_envs
    prev_health = [100] * num_envs
    rollout_frags = 0
    rollout_deaths = 0

    start_time = time.time()
    executor = ThreadPoolExecutor(max_workers=num_envs)

    # Initialize LSTM hidden states for all envs (on CPU — moved to GPU in act_batch)
    hidden_states = [model.init_hidden(1) for _ in range(num_envs)]
    # Move to CPU for storage
    hidden_states = [
        tuple(h.cpu() for h in hs) for hs in hidden_states
    ]

    # Reset all envs
    print(f"Resetting {num_envs} environment(s)...")
    raw_obs_list = list(executor.map(lambda env: env.reset()[0], envs))
    obs_normalizer.update(raw_obs_list)
    obs_list = [obs_normalizer.normalize(o) for o in raw_obs_list]
    raw_obs_list = [o.copy() for o in raw_obs_list]  # keep raw copy for aim targets
    print(f"Ready! Starting training loop ({num_envs}x parallel, LSTM seq={seq_len}).\n")

    for iteration in range(1, args.num_rollouts + 1):
        buffer.reset()
        rollout_rewards = []
        rollout_frags = 0
        rollout_deaths = 0

        # Drain stale semaphore posts
        for env in envs:
            env._drain_semaphore(env._sem_obs)

        # Resync with fresh obs
        noop = {"yaw_delta": 0, "pitch_delta": 0, "forwardmove": 0,
                "sidemove": 0, "jump": 0, "crouch": 0, "attack": 0, "weapon": 0}
        for env in envs:
            env.send_action(noop)
        fresh = list(executor.map(lambda env: env.recv_obs(), envs))
        raw_obs_batch = [fresh[e][0] for e in range(num_envs)]
        obs_normalizer.update(raw_obs_batch)
        for e in range(num_envs):
            raw_obs_list[e] = raw_obs_batch[e].copy()
            obs_list[e] = obs_normalizer.normalize(raw_obs_batch[e])

        # Collect rollout
        for t in range(steps_per_env):
            # Save hidden states at chunk boundaries (before forward pass)
            if t % seq_len == 0:
                chunk_idx = t // seq_len
                for e in range(num_envs):
                    buffer.save_hidden(e, chunk_idx, hidden_states[e])

            # Compute aim targets from raw obs (before normalization)
            # Target: clamp(aim_delta_degrees / max_delta_per_frame, -1, 1)
            # yaw: raw[51] = aim_yaw_deg/180, NN output*10=degrees → target = raw*18
            # pitch: raw[52] = aim_pitch_deg/90, NN output*5=degrees → target = raw*18
            aim_targets_t = []
            aim_masks_t = []
            for e in range(num_envs):
                raw = raw_obs_list[e]
                aim_targets_t.append([
                    np.clip(raw[51] * 18.0, -1.0, 1.0),
                    np.clip(raw[52] * 18.0, -1.0, 1.0),
                ])
                aim_masks_t.append(1.0 if raw[41] > 0.5 else 0.0)

            # Batch inference with LSTM hidden states
            actions, log_probs, values, new_hidden_states = model.act_batch(
                obs_list, hidden_states,
                disable_crouch=disable_crouch, fixed_weapon=fixed_weapon,
            )
            hidden_states = new_hidden_states

            # Aim assist: blend model yaw/pitch with ideal script aim (enemy visible only).
            # sent_actions has blended yaw/pitch; buffer.add() uses original actions
            # so policy gradient is computed on unmodified model output.
            if args.aim_assist and tuner.assist_weight > 0:
                sent_actions = []
                for e in range(num_envs):
                    if raw_obs_list[e][41] > 0.5:  # enemy visible
                        w = tuner.assist_weight
                        sa = dict(actions[e])
                        sa["yaw_delta"] = w * raw_obs_list[e][51] * 180.0 + (1.0 - w) * actions[e]["yaw_delta"]
                        sa["pitch_delta"] = w * raw_obs_list[e][52] * 90.0 + (1.0 - w) * actions[e]["pitch_delta"]
                        sent_actions.append(sa)
                    else:
                        sent_actions.append(actions[e])
            else:
                sent_actions = actions

            # Send all actions
            for e in range(num_envs):
                envs[e].send_action(sent_actions[e])

            # Receive observations in parallel
            results = list(executor.map(lambda env: env.recv_obs(), envs))

            # Process results
            reset_envs = []
            raw_obs_batch = [results[e][0] for e in range(num_envs)]
            obs_normalizer.update(raw_obs_batch)
            for e in range(num_envs):
                next_obs_raw, reward, terminated, truncated, info = results[e]
                rollout_rewards.append(reward)
                scaled_reward = reward * tuner.reward_scale

                buffer.add(e, obs_list[e], actions[e], log_probs[e], scaled_reward, values[e], terminated,
                           aim_target=aim_targets_t[e], aim_mask=aim_masks_t[e])
                total_timesteps += 1
                ep_reward[e] += reward
                ep_length[e] += 1
                ep_frags[e] = info.get("frags", ep_frags[e])

                cur_frags = info.get("frags", 0)
                cur_health = info.get("health", 100)
                if cur_frags > prev_frags[e]:
                    rollout_frags += cur_frags - prev_frags[e]
                if cur_health <= 0 and prev_health[e] > 0:
                    rollout_deaths += 1
                prev_frags[e] = cur_frags
                prev_health[e] = cur_health

                raw_obs_list[e] = next_obs_raw.copy()
                obs_list[e] = obs_normalizer.normalize(next_obs_raw)

                if terminated:
                    episode_rewards.append(ep_reward[e])
                    episode_frags.append(ep_frags[e])
                    episode_lengths.append(ep_length[e])
                    total_episodes += 1
                    ep_reward[e] = 0.0
                    ep_frags[e] = 0
                    ep_length[e] = 0
                    reset_envs.append(e)
                    # Reset LSTM hidden state for terminated env
                    h = model.init_hidden(1)
                    hidden_states[e] = tuple(x.cpu() for x in h)

            # Reset terminated envs
            if reset_envs:
                reset_results = list(executor.map(lambda e: (e, envs[e].reset()), reset_envs))
                for e, (obs_raw, _info) in reset_results:
                    raw_obs_list[e] = obs_raw.copy()
                    obs_list[e] = obs_normalizer.normalize(obs_raw)

        # Bootstrap ALL envs (fix: old code only bootstrapped env 0)
        with torch.no_grad():
            last_values = np.zeros(num_envs, dtype=np.float32)
            for e in range(num_envs):
                obs_t = torch.FloatTensor(obs_list[e]).unsqueeze(0).to(device)
                h_c = hidden_states[e][2].to(device)
                c_c = hidden_states[e][3].to(device)
                val, _ = model.get_value(obs_t, (h_c, c_c))
                last_values[e] = val.item()

        buffer.compute_returns(last_values, gamma=args.gamma, lam=args.lam)

        # Apply adaptive LR before PPO update
        optimizer.param_groups[0]['lr'] = tuner.lr

        # PPO update with sequence chunks
        metrics = ppo_update(
            model, optimizer, buffer,
            clip_eps=args.clip_eps,
            epochs=args.epochs,
            batch_chunks=args.batch_chunks,
            entropy_coef=tuner.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            aim_coef=args.aim_coef,
            device=device,
            disable_crouch=disable_crouch,
            fixed_weapon=fixed_weapon,
        )

        # Adaptive tuning
        tuner_state = tuner.update(iteration, metrics, rollout_frags, rollout_deaths)

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

            print(f"[{iteration:>6}] steps={total_timesteps:>8} | "
                  f"step_r={step_r:>7.4f} frags={rollout_frags:>3} deaths={rollout_deaths:>3} | "
                  f"pi={metrics['policy_loss']:>7.4f} v={metrics['value_loss']:>7.4f} "
                  f"aim={metrics['aim_loss']:>6.4f} ent={metrics['entropy']:>6.3f} kl={metrics['approx_kl']:>7.4f} | "
                  f"{elapsed:>6.0f}s")
            print(f"  [TUNER] ent_c={tuner_state['entropy_coef']:.4f} "
                  f"lr={tuner_state['lr']:.2e} "
                  f"rscale={tuner_state['reward_scale']:.1f} "
                  f"bot_skill={tuner_state['bot_skill']} "
                  f"assist={tuner_state['assist_weight']:.3f}")

            log_writer.writerow([
                iteration, total_timesteps, total_episodes,
                f"{mean_r:.4f}", f"{step_r:.4f}",
                rollout_frags, rollout_deaths, f"{mean_l:.0f}",
                f"{metrics['policy_loss']:.6f}", f"{metrics['value_loss']:.6f}",
                f"{metrics['entropy']:.4f}", f"{metrics['clip_fraction']:.4f}",
                f"{metrics['approx_kl']:.6f}", f"{metrics['aim_loss']:.6f}", f"{elapsed:.1f}",
                f"{tuner_state['entropy_coef']:.6f}", f"{tuner_state['lr']:.2e}",
                f"{tuner_state['reward_scale']:.2f}", tuner_state['bot_skill'],
                f"{tuner_state['assist_weight']:.4f}",
            ])
            log_file.flush()

        # Save checkpoint (model + normalizer)
        if iteration % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"model_{iteration}.pt")
            torch.save({
                "model": model.state_dict(),
                "normalizer": obs_normalizer.state_dict(),
                "iteration": iteration,
                "timesteps": total_timesteps,
            }, path)
            print(f"  Checkpoint saved: {path}")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    torch.save({
        "model": model.state_dict(),
        "normalizer": obs_normalizer.state_dict(),
        "iteration": args.num_rollouts,
        "timesteps": total_timesteps,
    }, final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total: {total_timesteps} steps, {total_episodes} episodes, {time.time() - start_time:.0f}s")

    log_file.close()
    executor.shutdown(wait=False)
    for env in envs:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2AI PPO Training (LSTM)")
    parser.add_argument("--instance", type=int, default=0, help="Starting instance ID")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.005, help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=1.0, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--rollout-len", type=int, default=256, help="Steps per rollout per env")
    parser.add_argument("--seq-len", type=int, default=32, help="LSTM sequence length (TBPTT chunk size)")
    parser.add_argument("--batch-chunks", type=int, default=16, help="Number of sequence chunks per mini-batch")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per rollout")
    parser.add_argument("--target-kl", type=float, default=0.05, help="KL early stopping threshold (0=disabled)")
    parser.add_argument("--num-rollouts", type=int, default=100000, help="Total rollout iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=100, help="Save every N iterations")
    parser.add_argument("--timeout", type=float, default=5.0, help="Observation timeout (seconds)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--bot-skill", type=int, default=1, help="Initial bot skill level (0-10)")
    parser.add_argument("--aim-coef", type=float, default=1.0, help="Auxiliary aim supervision coefficient")
    parser.add_argument("--aim-assist", dest="aim_assist", action="store_true", default=True,
                        help="Enable aim assist curriculum (default: on)")
    parser.add_argument("--no-aim-assist", dest="aim_assist", action="store_false",
                        help="Disable aim assist curriculum")
    parser.add_argument("--aim-assist-weight", type=float, default=1.0,
                        help="Initial aim assist blend weight (1.0=full script, 0.0=pure model)")
    parser.add_argument("--disable-crouch", action="store_true", default=False,
                        help="Always output crouch=0, exclude from entropy")
    parser.add_argument("--single-weapon", choices=["none", "rl", "rg"], default="none",
                        help="Fix weapon output: none=model decides, rl=rocket launcher, rg=railgun")
    train(parser.parse_args())
