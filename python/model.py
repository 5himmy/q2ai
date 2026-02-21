"""
model.py -- Actor-Critic MLP for PPO training.

Separate actor and critic networks (no shared trunk).
Actor: 256-256-128 trunk → 18 outputs (4 continuous + 3 binary + 11 weapon).
Critic: 256-256-128 trunk → 1 scalar value.

Continuous actions use state-independent log_std (learned parameter).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal

from q2ai_env import OBS_DIM


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (standard for PPO)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """PPO Actor-Critic with hybrid action space (separate networks)."""

    # Action dimensions
    CONTINUOUS_DIM = 4   # yaw_delta, pitch_delta, forwardmove, sidemove
    BINARY_DIM = 3       # jump, crouch, attack
    WEAPON_DIM = 11      # weapon selection (0=no switch, 1-10)
    ACTOR_OUT = CONTINUOUS_DIM + BINARY_DIM + WEAPON_DIM  # 18

    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()

        # Actor network (own trunk + head)
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, self.ACTOR_OUT), std=0.01)

        # Bias continuous actions: [yaw, pitch, forward, side]
        with torch.no_grad():
            self.actor.bias[2] = 0.3    # forward: slight tendency to move
            self.actor.bias[4] = -2.2   # jump: ~10%
            self.actor.bias[5] = -10.0  # crouch: effectively disabled (~0.005%)
            self.actor.bias[6] = -4.0   # attack: ~2% (avoid rocket suicide)
            self.actor.bias[7] = 2.0    # weapon 0: no switch (strong preference)

        # Critic network (own trunk + head)
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

        # State-independent log_std for continuous actions
        self.action_log_std = nn.Parameter(torch.tensor([-0.5, -0.5, 0.0, 0.0]))

        # Keep legacy 'trunk' attribute pointing to actor_trunk for BC model loading
        self.trunk = self.actor_trunk

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log_prob, entropy, and value.

        If action is None: sample new action (for rollout collection).
        If action is provided: evaluate given action (for PPO update).
        """
        actor_features = self.actor_trunk(obs)
        actor_out = self.actor(actor_features)

        critic_features = self.critic_trunk(obs)
        value = self.critic(critic_features).squeeze(-1)

        # Split actor output
        cont_means = actor_out[:, :self.CONTINUOUS_DIM]
        binary_logits = actor_out[:, self.CONTINUOUS_DIM:self.CONTINUOUS_DIM + self.BINARY_DIM]
        weapon_logits = actor_out[:, self.CONTINUOUS_DIM + self.BINARY_DIM:]

        # Distributions
        cont_std = torch.clamp(self.action_log_std, -2.0, 0.5).exp()
        cont_dist = Normal(cont_means, cont_std)
        jump_dist = Bernoulli(logits=binary_logits[:, 0])
        crouch_dist = Bernoulli(logits=binary_logits[:, 1])
        attack_dist = Bernoulli(logits=binary_logits[:, 2])
        weapon_dist = Categorical(logits=weapon_logits)

        if action is None:
            # Sample
            cont_action = cont_dist.sample()
            jump_action = jump_dist.sample()
            crouch_action = crouch_dist.sample()
            attack_action = attack_dist.sample()
            weapon_action = weapon_dist.sample()
        else:
            cont_action = action["continuous"]
            jump_action = action["jump"]
            crouch_action = action["crouch"]
            attack_action = action["attack"]
            weapon_action = action["weapon"]

        # Log probabilities (sum across sub-distributions)
        log_prob = (
            cont_dist.log_prob(cont_action).sum(-1)
            + jump_dist.log_prob(jump_action)
            + crouch_dist.log_prob(crouch_action)
            + attack_dist.log_prob(attack_action)
            + weapon_dist.log_prob(weapon_action)
        )

        # Entropy (sum across sub-distributions)
        entropy = (
            cont_dist.entropy().sum(-1)
            + jump_dist.entropy()
            + crouch_dist.entropy()
            + attack_dist.entropy()
            + weapon_dist.entropy()
        )

        return {
            "continuous": cont_action,
            "jump": jump_action,
            "crouch": crouch_action,
            "attack": attack_action,
            "weapon": weapon_action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }

    def get_value(self, obs):
        """Get state value only (for bootstrapping)."""
        features = self.critic_trunk(obs)
        return self.critic(features).squeeze(-1)

    def act(self, obs_np):
        """Convenience: numpy obs in, action dict out (for env.step)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            result = self.get_action_and_value(obs_t)

            cont = result["continuous"][0].cpu().numpy()

            return {
                "yaw_delta": float(np.clip(cont[0] * 10.0, -10.0, 10.0)),
                "pitch_delta": float(np.clip(cont[1] * 5.0, -5.0, 5.0)),
                "forwardmove": float(np.clip(cont[2], -1.0, 1.0)),
                "sidemove": float(np.clip(cont[3], -1.0, 1.0)),
                "jump": int(result["jump"][0].item()),
                "crouch": int(result["crouch"][0].item()),
                "attack": int(result["attack"][0].item()),
                "weapon": int(result["weapon"][0].item()),
            }, float(result["log_prob"][0].item()), float(result["value"][0].item())

    def act_batch(self, obs_list):
        """Batched inference for all N envs in one GPU round trip."""
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.FloatTensor(np.stack(obs_list)).to(device)
            result = self.get_action_and_value(obs_t)

            cont = result["continuous"].cpu().numpy()
            jumps = result["jump"].cpu().numpy()
            crouches = result["crouch"].cpu().numpy()
            attacks = result["attack"].cpu().numpy()
            weapons = result["weapon"].cpu().numpy()
            log_probs = result["log_prob"].cpu().numpy()
            vals = result["value"].cpu().numpy()

            actions = []
            for i in range(len(obs_list)):
                actions.append({
                    # Clipped values for sending to game
                    "yaw_delta": float(np.clip(cont[i, 0] * 10.0, -10.0, 10.0)),
                    "pitch_delta": float(np.clip(cont[i, 1] * 5.0, -5.0, 5.0)),
                    "forwardmove": float(np.clip(cont[i, 2], -1.0, 1.0)),
                    "sidemove": float(np.clip(cont[i, 3], -1.0, 1.0)),
                    "jump": int(jumps[i]),
                    "crouch": int(crouches[i]),
                    "attack": int(attacks[i]),
                    "weapon": int(weapons[i]),
                    # Raw (unclipped) model outputs for PPO buffer
                    "_raw_continuous": cont[i].copy(),
                })

            return actions, log_probs.tolist(), vals.tolist()
