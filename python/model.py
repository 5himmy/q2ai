"""
model.py -- Actor-Critic MLP for PPO training.

Shared trunk (256-256-128) with separate actor and critic heads.
Actor outputs: 4 continuous means + 3 binary logits + 11 weapon logits.
Critic outputs: scalar state value.

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
    """PPO Actor-Critic with hybrid action space."""

    # Action dimensions
    CONTINUOUS_DIM = 4   # yaw_delta, pitch_delta, forwardmove, sidemove
    BINARY_DIM = 3       # jump, crouch, attack
    WEAPON_DIM = 11      # weapon selection (0=no switch, 1-10)
    ACTOR_OUT = CONTINUOUS_DIM + BINARY_DIM + WEAPON_DIM  # 18

    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )

        # Actor head: continuous means + binary logits + weapon logits
        self.actor = layer_init(nn.Linear(128, self.ACTOR_OUT), std=0.01)

        # Bias continuous actions: [yaw, pitch, forward, side]
        # Bias forwardmove positive so agent actually explores the map
        # Binary logits at indices [4, 5, 6] = [jump, crouch, attack]
        with torch.no_grad():
            self.actor.bias[2] = 0.3    # forward: slight tendency to move
            self.actor.bias[4] = -2.2   # jump: ~10%
            self.actor.bias[5] = -10.0  # crouch: effectively disabled (~0.005%)
            self.actor.bias[6] = -4.0   # attack: ~2% (avoid rocket suicide)
            # Bias weapon logits toward index 0 (no switch)
            self.actor.bias[7] = 2.0    # weapon 0: no switch (strong preference)

        # Critic head: scalar value
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

        # State-independent log_std for continuous actions
        # Higher std for movement (indices 2,3) so agent actually moves
        # Yaw/pitch std ~0.6, movement std ~1.0
        self.action_log_std = nn.Parameter(torch.tensor([-0.5, -0.5, 0.0, 0.0]))

    def forward(self, obs):
        """Forward pass through shared trunk."""
        return self.trunk(obs)

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log_prob, entropy, and value.

        If action is None: sample new action (for rollout collection).
        If action is provided: evaluate given action (for PPO update).

        Args:
            obs: [batch, obs_dim] tensor
            action: dict with 'continuous', 'binary', 'weapon' tensors, or None

        Returns:
            dict with 'action', 'log_prob', 'entropy', 'value' (and sub-actions)
        """
        features = self.forward(obs)
        actor_out = self.actor(features)
        value = self.critic(features).squeeze(-1)

        # Split actor output
        cont_means = actor_out[:, :self.CONTINUOUS_DIM]
        binary_logits = actor_out[:, self.CONTINUOUS_DIM:self.CONTINUOUS_DIM + self.BINARY_DIM]
        weapon_logits = actor_out[:, self.CONTINUOUS_DIM + self.BINARY_DIM:]

        # Distributions
        cont_std = torch.clamp(self.action_log_std, -20, 2).exp()
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
        features = self.forward(obs)
        return self.critic(features).squeeze(-1)

    def act(self, obs_np):
        """
        Convenience: numpy obs in, action dict out (for env.step).

        Returns dict compatible with Q2AIEnv.step().
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            result = self.get_action_and_value(obs_t)

            cont = result["continuous"][0].cpu().numpy()

            return {
                # Scale continuous actions to game ranges
                "yaw_delta": float(np.clip(cont[0] * 10.0, -10.0, 10.0)),
                "pitch_delta": float(np.clip(cont[1] * 5.0, -5.0, 5.0)),
                "forwardmove": float(np.clip(cont[2], -1.0, 1.0)),
                "sidemove": float(np.clip(cont[3], -1.0, 1.0)),
                "jump": int(result["jump"][0].item()),
                "crouch": int(result["crouch"][0].item()),
                "attack": int(result["attack"][0].item()),
                "weapon": int(result["weapon"][0].item()),
            }, float(result["log_prob"][0].item()), float(result["value"][0].item())
