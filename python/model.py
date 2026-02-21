"""
model.py -- Actor-Critic with grouped encoder + entity attention + LSTM memory.

V2.5 Architecture:
  obs (91) → slice into groups → group encoders → concat → fusion → LSTM → head

Group encoders:
  - Self: Fourier(pos) + vel + angles + stats + alive → Linear(38→96)
  - Inventory: weapon_1hot + owned + ammo → Linear(27→64)
  - Enemy: visible + rel_xyz + dist + angle → Linear(7→64)
  - Entities: 8×[xyz+active+type] → shared Linear(5→64) + attention pool → 64
  - Game: frags + enemy_frags + match_time → 3 (passthrough)

Fusion: Linear(291→256) → ReLU → Linear(256→256) → ReLU (shared)
Then: separate Actor LSTM(256→256) + Critic LSTM(256→256)

Actor head: Linear(256→18) — 4 continuous + 3 binary + 11 weapon.
Critic head: Linear(256→1) — scalar value.

Continuous actions use state-independent log_std (learned parameter).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from q2ai_env import OBS_DIM


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (standard for PPO)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class FourierEncoding(nn.Module):
    """Fixed sinusoidal encoding for position coordinates.

    Maps 3D position to higher-dimensional space so the network can
    distinguish map locations (e.g. mega health room vs RA corridor).

    Input: (..., 3) → Output: (..., 27)  [3 × (2×L + 1) where L=4]
    """

    def __init__(self, input_dim=3, num_bands=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        # Output: input_dim * (2 * num_bands + 1) = 3 * 9 = 27
        self.output_dim = input_dim * (2 * num_bands + 1)

        # Frequencies: [1, 2, 4, 8] * pi — fixed, not learned
        freqs = torch.tensor([2**i for i in range(num_bands)], dtype=torch.float32) * np.pi
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """x: (..., 3) → (..., 27)"""
        # x is already normalized (position / 4096 from obs normalizer)
        # Apply each frequency: sin(freq * x), cos(freq * x)
        # x: (..., 3), freqs: (L,) → outer product over last dims
        x_expanded = x.unsqueeze(-1) * self.freqs  # (..., 3, L)
        sin_enc = torch.sin(x_expanded)  # (..., 3, L)
        cos_enc = torch.cos(x_expanded)  # (..., 3, L)
        # Flatten: (..., 3, 2*L) then concat with original
        encoded = torch.cat([
            x.unsqueeze(-1),   # (..., 3, 1) — original values
            sin_enc,           # (..., 3, L)
            cos_enc,           # (..., 3, L)
        ], dim=-1)            # (..., 3, 2*L+1)
        # Reshape to (..., 3 * (2*L+1)) = (..., 27)
        return encoded.flatten(-2)


class EntityAttention(nn.Module):
    """Permutation-invariant entity encoder with attention pooling.

    Shared linear applied to each entity, then attention-weighted sum.
    Entities at any slot produce the same representation.

    Input: (..., 8, 5) → Output: (..., 64)
    """

    def __init__(self, entity_dim=5, hidden_dim=64):
        super().__init__()
        self.entity_encoder = nn.Sequential(
            layer_init(nn.Linear(entity_dim, hidden_dim)),
            nn.ReLU(),
        )
        # Learned query vector for attention
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.1)

    def forward(self, entities, active_mask=None):
        """
        Args:
            entities: (..., 8, 5) — [x, y, z, is_active, type] per entity
            active_mask: (..., 8) bool — which entities are active.
                         If None, derived from entities[..., 3] > 0.5

        Returns:
            pooled: (..., 64)
        """
        if active_mask is None:
            active_mask = entities[..., 3] > 0.5  # (..., 8)

        # Encode each entity identically
        encoded = self.entity_encoder(entities)  # (..., 8, 64)

        # Attention scores: dot product with query
        scores = (encoded * self.query).sum(dim=-1)  # (..., 8)

        # Mask inactive entities with -inf before softmax
        scores = scores.masked_fill(~active_mask, float('-inf'))

        # Handle case where all entities are inactive
        all_inactive = ~active_mask.any(dim=-1, keepdim=True)  # (..., 1)
        # Replace -inf with 0 when all inactive (softmax of zeros = uniform, but result is zero anyway)
        scores = scores.masked_fill(all_inactive.expand_as(scores), 0.0)

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (..., 8, 1)

        # Weighted sum
        pooled = (encoded * weights).sum(dim=-2)  # (..., 64)

        # Zero out when all entities inactive
        pooled = pooled * (~all_inactive.squeeze(-1)).unsqueeze(-1).float()

        return pooled


class ActorCritic(nn.Module):
    """PPO Actor-Critic with grouped encoder, entity attention, and LSTM memory."""

    # Action dimensions
    CONTINUOUS_DIM = 4   # yaw_delta, pitch_delta, forwardmove, sidemove
    BINARY_DIM = 3       # jump, crouch, attack
    WEAPON_DIM = 11      # weapon selection (0=no switch, 1-10)
    ACTOR_OUT = CONTINUOUS_DIM + BINARY_DIM + WEAPON_DIM  # 18

    HIDDEN_SIZE = 256

    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()

        # --- Shared Group Encoders ---
        self.fourier = FourierEncoding(input_dim=3, num_bands=4)  # 3 → 27

        # Self state: fourier(27) + vel(3) + angles(4) + stats(3) + alive(1) = 38
        self.self_encoder = nn.Sequential(
            layer_init(nn.Linear(38, 96)),
            nn.ReLU(),
        )

        # Inventory: weapon_1hot(11) + owned(10) + ammo(6) = 27
        self.inv_encoder = nn.Sequential(
            layer_init(nn.Linear(27, 64)),
            nn.ReLU(),
        )

        # Enemy: visible(1) + rel_xyz(3) + dist(1) + angle(2) = 7
        self.enemy_encoder = nn.Sequential(
            layer_init(nn.Linear(7, 64)),
            nn.ReLU(),
        )

        # Entity attention: 8 × 5 → 64
        self.entity_attn = EntityAttention(entity_dim=5, hidden_dim=64)

        # --- Shared Fusion ---
        # Input: self(96) + inv(64) + enemy(64) + entities(64) + game(3) = 291
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(291, self.HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)),
            nn.ReLU(),
        )

        # --- Separate Actor/Critic LSTMs + Heads ---
        self.actor_lstm = nn.LSTM(self.HIDDEN_SIZE, self.HIDDEN_SIZE, batch_first=True)
        self.actor = layer_init(nn.Linear(self.HIDDEN_SIZE, self.ACTOR_OUT), std=0.01)

        # Bias continuous actions: [yaw, pitch, forward, side]
        with torch.no_grad():
            self.actor.bias[2] = 0.3    # forward: slight tendency to move
            self.actor.bias[4] = -2.2   # jump: ~10%
            self.actor.bias[5] = -10.0  # crouch: effectively disabled (~0.005%)
            self.actor.bias[6] = -4.0   # attack: ~2% (avoid rocket suicide)
            self.actor.bias[7] = 2.0    # weapon 0: no switch (strong preference)

        self.critic_lstm = nn.LSTM(self.HIDDEN_SIZE, self.HIDDEN_SIZE, batch_first=True)
        self.critic = layer_init(nn.Linear(self.HIDDEN_SIZE, 1), std=1.0)

        # State-independent log_std for continuous actions
        self.action_log_std = nn.Parameter(torch.tensor([-0.5, -0.5, 0.0, 0.0]))

        # Initialize LSTM weights with orthogonal init
        for lstm in [self.actor_lstm, self.critic_lstm]:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    # Set forget gate bias to 1.0 (standard practice)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)

    def _encode_obs(self, obs):
        """Slice obs into groups, encode each, and fuse.

        Works for both (batch, 91) and (batch, seq, 91).

        Returns: (..., 256) fused features
        """
        # --- Slice obs into groups ---
        pos = obs[..., 0:3]         # 3  → Fourier encode
        vel = obs[..., 3:6]         # 3
        angles = obs[..., 6:10]     # 4  sin/cos yaw/pitch
        stats = obs[..., 10:13]     # 3  health, armor, ammo
        weapon_1hot = obs[..., 13:24]  # 11
        alive = obs[..., 24:25]     # 1
        weapons_owned = obs[..., 25:35]  # 10
        ammo = obs[..., 35:41]      # 6
        enemy = obs[..., 41:48]     # 7
        ent_flat = obs[..., 48:88]  # 40 → reshape to (..., 8, 5)
        game = obs[..., 88:91]      # 3  frags, enemy_frags, match_time

        # --- Self state group ---
        pos_fourier = self.fourier(pos)  # (..., 27)
        self_input = torch.cat([pos_fourier, vel, angles, stats, alive], dim=-1)  # (..., 38)
        self_features = self.self_encoder(self_input)  # (..., 96)

        # --- Inventory group ---
        inv_input = torch.cat([weapon_1hot, weapons_owned, ammo], dim=-1)  # (..., 27)
        inv_features = self.inv_encoder(inv_input)  # (..., 64)

        # --- Enemy group ---
        enemy_features = self.enemy_encoder(enemy)  # (..., 64)

        # --- Entity attention group ---
        # Reshape flat entities to (..., 8, 5)
        ent_shape = ent_flat.shape[:-1] + (8, 5)
        entities = ent_flat.reshape(ent_shape)
        entity_features = self.entity_attn(entities)  # (..., 64)

        # --- Fuse all groups ---
        fused = torch.cat([self_features, inv_features, enemy_features,
                          entity_features, game], dim=-1)  # (..., 291)
        return self.fusion(fused)  # (..., 256)

    def init_hidden(self, batch_size=1):
        """Return zero hidden states for actor and critic LSTMs.

        Returns:
            (h_actor, c_actor, h_critic, c_critic) — each shape (1, batch, 256)
        """
        device = next(self.parameters()).device
        zeros = lambda: torch.zeros(1, batch_size, self.HIDDEN_SIZE, device=device)
        return zeros(), zeros(), zeros(), zeros()

    def _forward_actor(self, obs, hidden):
        """Encode obs through shared encoder + actor LSTM.

        Args:
            obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: (h, c) each (1, batch, 256), or None

        Returns:
            features: (batch, seq_len, 256) or (batch, 256)
            new_hidden: (h, c)
        """
        squeeze = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
            squeeze = True

        encoded = self._encode_obs(obs)  # (batch, seq, 256)
        if hidden is None:
            lstm_out, new_hidden = self.actor_lstm(encoded)
        else:
            lstm_out, new_hidden = self.actor_lstm(encoded, hidden)

        if squeeze:
            lstm_out = lstm_out.squeeze(1)  # (batch, 256)
        return lstm_out, new_hidden

    def _forward_critic(self, obs, hidden):
        """Encode obs through shared encoder + critic LSTM."""
        squeeze = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze = True

        encoded = self._encode_obs(obs)  # (batch, seq, 256)
        if hidden is None:
            lstm_out, new_hidden = self.critic_lstm(encoded)
        else:
            lstm_out, new_hidden = self.critic_lstm(encoded, hidden)

        if squeeze:
            lstm_out = lstm_out.squeeze(1)
        return lstm_out, new_hidden

    def get_action_and_value(self, obs, hidden_actor=None, hidden_critic=None, action=None):
        """
        Get action, log_prob, entropy, value, and updated hidden states.

        Args:
            obs: (batch, obs_dim) for rollout, (batch, seq_len, obs_dim) for training
            hidden_actor: (h, c) tuple or None
            hidden_critic: (h, c) tuple or None
            action: dict of actions to evaluate (for PPO update), or None to sample

        Returns:
            dict with actions, log_prob, entropy, value, hidden_actor, hidden_critic
        """
        actor_features, new_hidden_actor = self._forward_actor(obs, hidden_actor)
        actor_out = self.actor(actor_features)

        critic_features, new_hidden_critic = self._forward_critic(obs, hidden_critic)
        value = self.critic(critic_features).squeeze(-1)

        # Split actor output — works for both (batch, 18) and (batch, seq, 18)
        cont_means = actor_out[..., :self.CONTINUOUS_DIM]
        binary_logits = actor_out[..., self.CONTINUOUS_DIM:self.CONTINUOUS_DIM + self.BINARY_DIM]
        weapon_logits = actor_out[..., self.CONTINUOUS_DIM + self.BINARY_DIM:]

        # Distributions
        cont_std = torch.clamp(self.action_log_std, -2.0, 0.5).exp()
        cont_dist = Normal(cont_means, cont_std)
        jump_dist = Bernoulli(logits=binary_logits[..., 0])
        crouch_dist = Bernoulli(logits=binary_logits[..., 1])
        attack_dist = Bernoulli(logits=binary_logits[..., 2])
        weapon_dist = Categorical(logits=weapon_logits)

        if action is None:
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

        log_prob = (
            cont_dist.log_prob(cont_action).sum(-1)
            + jump_dist.log_prob(jump_action)
            + crouch_dist.log_prob(crouch_action)
            + attack_dist.log_prob(attack_action)
            + weapon_dist.log_prob(weapon_action)
        )

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
            "hidden_actor": new_hidden_actor,
            "hidden_critic": new_hidden_critic,
        }

    def get_value(self, obs, hidden_critic=None):
        """Get state value only (for bootstrapping).

        Returns:
            value: scalar or (batch,) tensor
            new_hidden_critic: (h, c) tuple
        """
        features, new_hidden = self._forward_critic(obs, hidden_critic)
        return self.critic(features).squeeze(-1), new_hidden

    def act_batch(self, obs_list, hidden_states=None):
        """Batched inference for all N envs in one GPU round trip.

        Args:
            obs_list: list of N numpy obs arrays
            hidden_states: list of N tuples (h_a, c_a, h_c, c_c) each (1,1,256),
                          or None to use zero states

        Returns:
            actions: list of N action dicts
            log_probs: list of N floats
            values: list of N floats
            new_hidden_states: list of N tuples (h_a, c_a, h_c, c_c)
        """
        N = len(obs_list)
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.FloatTensor(np.stack(obs_list)).to(device)  # (N, obs_dim)

            # Stack hidden states: (1,1,256) per env → (1,N,256) batched
            if hidden_states is not None:
                h_a = torch.cat([h[0] for h in hidden_states], dim=1).to(device)
                c_a = torch.cat([h[1] for h in hidden_states], dim=1).to(device)
                h_c = torch.cat([h[2] for h in hidden_states], dim=1).to(device)
                c_c = torch.cat([h[3] for h in hidden_states], dim=1).to(device)
                ha_tuple = (h_a, c_a)
                hc_tuple = (h_c, c_c)
            else:
                ha_tuple = None
                hc_tuple = None

            result = self.get_action_and_value(obs_t, ha_tuple, hc_tuple)

            cont = result["continuous"].cpu().numpy()
            jumps = result["jump"].cpu().numpy()
            crouches = result["crouch"].cpu().numpy()
            attacks = result["attack"].cpu().numpy()
            weapons = result["weapon"].cpu().numpy()
            log_probs = result["log_prob"].cpu().numpy()
            vals = result["value"].cpu().numpy()

            # Unbatch hidden states: (1,N,256) → list of N (1,1,256)
            new_ha, new_ca = result["hidden_actor"]
            new_hc, new_cc = result["hidden_critic"]
            new_hidden_states = []
            for i in range(N):
                new_hidden_states.append((
                    new_ha[:, i:i+1, :].cpu(),
                    new_ca[:, i:i+1, :].cpu(),
                    new_hc[:, i:i+1, :].cpu(),
                    new_cc[:, i:i+1, :].cpu(),
                ))

            actions = []
            for i in range(N):
                actions.append({
                    "yaw_delta": float(np.clip(cont[i, 0] * 10.0, -10.0, 10.0)),
                    "pitch_delta": float(np.clip(cont[i, 1] * 5.0, -5.0, 5.0)),
                    "forwardmove": float(np.clip(cont[i, 2], -1.0, 1.0)),
                    "sidemove": float(np.clip(cont[i, 3], -1.0, 1.0)),
                    "jump": int(jumps[i]),
                    "crouch": int(crouches[i]),
                    "attack": int(attacks[i]),
                    "weapon": int(weapons[i]),
                    "_raw_continuous": cont[i].copy(),
                })

            return actions, log_probs.tolist(), vals.tolist(), new_hidden_states
