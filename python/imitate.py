"""
imitate.py -- Behavior cloning from MVD demo training data.

Reads (observation, action) pairs from demo_training_data.bin,
converts them to the same format used by the RL pipeline, and
trains the ActorCritic network via supervised learning.

Usage:
    python imitate.py --data /path/to/demo_training_data.bin --epochs 50
    python imitate.py --data /path/to/demo_training_data.bin --epochs 50 --resume model.pt
"""

import argparse
import ctypes
import math
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

from model import ActorCritic
from q2ai_structs import (
    AiAction,
    AiEntityObs,
    AiObservation,
    AMMO_INDICES,
    AMMO_MAX,
    ENT_TYPE_ITEM,
    ENT_TYPE_PLAYER,
    PM_NORMAL,
    WEAPON_INDICES,
)
from q2ai_env import OBS_DIM


def load_demo_data(path):
    """Load (observation, action) pairs from binary demo file."""
    with open(path, "rb") as f:
        obs_size, act_size = struct.unpack("ii", f.read(8))
        print(f"obs_size={obs_size}, act_size={act_size}")

        # Verify sizes match our structs
        expected_obs = ctypes.sizeof(AiObservation)
        expected_act = ctypes.sizeof(AiAction)
        assert obs_size == expected_obs, f"obs size mismatch: file={obs_size}, expected={expected_obs}"
        assert act_size == expected_act, f"act size mismatch: file={act_size}, expected={expected_act}"

        data = f.read()

    pair_size = obs_size + act_size
    n_pairs = len(data) // pair_size
    print(f"Loaded {n_pairs} observation-action pairs")

    observations = []
    actions = []

    for i in range(n_pairs):
        offset = i * pair_size
        obs_raw = data[offset : offset + obs_size]
        act_raw = data[offset + obs_size : offset + pair_size]

        obs = AiObservation.from_buffer_copy(obs_raw)
        act = AiAction.from_buffer_copy(act_raw)

        observations.append(obs)
        actions.append(act)

    return observations, actions


def obs_to_vector(o):
    """Convert AiObservation to the same normalized vector as Q2AIEnv._extract_obs."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Self position and velocity (normalized)
    obs[0] = o.origin[0] / 4096.0
    obs[1] = o.origin[1] / 4096.0
    obs[2] = o.origin[2] / 4096.0
    obs[3] = o.velocity[0] / 400.0
    obs[4] = o.velocity[1] / 400.0
    obs[5] = o.velocity[2] / 400.0

    # View angles as sin/cos
    yaw_rad = math.radians(o.viewangles[1])
    pitch_rad = math.radians(o.viewangles[0])
    obs[6] = math.sin(yaw_rad)
    obs[7] = math.cos(yaw_rad)
    obs[8] = math.sin(pitch_rad)
    obs[9] = math.cos(pitch_rad)

    # Stats
    obs[10] = o.health / 100.0
    obs[11] = o.armor / 200.0
    obs[12] = o.ammo / 50.0

    # Weapon one-hot
    weapon_idx = o.current_weapon
    if 0 <= weapon_idx <= 10:
        obs[13 + weapon_idx] = 1.0

    # Alive flag (in demo, PM_FREEZE is also alive since it's MVD chase)
    obs[24] = 1.0 if o.health > 0 else 0.0

    # Weapons owned
    for i, inv_idx in enumerate(WEAPON_INDICES):
        obs[25 + i] = 1.0 if o.inventory[inv_idx] > 0 else 0.0

    # Ammo counts
    for i, (inv_idx, max_val) in enumerate(zip(AMMO_INDICES, AMMO_MAX)):
        obs[35 + i] = o.inventory[inv_idx] / max_val

    # Enemy player
    self_origin = np.array([o.origin[0], o.origin[1], o.origin[2]])
    if o.enemy_visible:
        obs[41] = 1.0
        enemy_pos = np.array([o.enemy_origin[0], o.enemy_origin[1], o.enemy_origin[2]])
        rel_enemy = enemy_pos - self_origin
        obs[42] = rel_enemy[0] / 4096.0
        obs[43] = rel_enemy[1] / 4096.0
        obs[44] = rel_enemy[2] / 4096.0
        dist = np.linalg.norm(rel_enemy)
        obs[45] = min(dist / 4096.0, 1.0)
        angle_to_enemy = math.atan2(rel_enemy[1], rel_enemy[0]) - yaw_rad
        obs[46] = math.sin(angle_to_enemy)
        obs[47] = math.cos(angle_to_enemy)

    # Entities: 8 nearest
    entities = []
    for j in range(o.num_entities):
        e = o.entities[j]
        if not e.active:
            continue
        e_origin = np.array([e.origin[0], e.origin[1], e.origin[2]])
        rel = e_origin - self_origin
        dist = np.linalg.norm(rel)
        etype = e.entity_type / 3.0
        entities.append((dist, rel, etype))

    entities.sort(key=lambda x: x[0])
    for i, (dist, rel, etype) in enumerate(entities[:8]):
        base = 48 + i * 5
        obs[base] = rel[0] / 4096.0
        obs[base + 1] = rel[1] / 4096.0
        obs[base + 2] = rel[2] / 4096.0
        obs[base + 3] = 1.0
        obs[base + 4] = etype

    # Game state
    obs[88] = o.frags / 20.0

    return obs


def action_to_targets(act):
    """Convert AiAction to target tensors for supervised loss."""
    # Continuous targets: scale to network output space
    # Network outputs raw values, then scaled: yaw*10, pitch*5, fwd clip, side clip
    cont = np.array([
        act.yaw_delta / 10.0,   # network space
        act.pitch_delta / 5.0,  # network space
        act.forwardmove,        # already [-1, 1]
        act.sidemove,           # already [-1, 1]
    ], dtype=np.float32)

    return {
        "continuous": cont,
        "jump": float(act.jump),
        "crouch": float(act.crouch),
        "attack": float(act.attack),
        "weapon": 0,  # no weapon switch info from demo
    }


def prepare_dataset(observations, actions):
    """Convert raw C structs to torch tensors."""
    obs_vecs = np.array([obs_to_vector(o) for o in observations], dtype=np.float32)
    targets = [action_to_targets(a) for a in actions]

    cont_targets = np.array([t["continuous"] for t in targets], dtype=np.float32)
    jump_targets = np.array([t["jump"] for t in targets], dtype=np.float32)
    crouch_targets = np.array([t["crouch"] for t in targets], dtype=np.float32)
    attack_targets = np.array([t["attack"] for t in targets], dtype=np.float32)

    return {
        "obs": torch.FloatTensor(obs_vecs),
        "continuous": torch.FloatTensor(cont_targets),
        "jump": torch.FloatTensor(jump_targets),
        "crouch": torch.FloatTensor(crouch_targets),
        "attack": torch.FloatTensor(attack_targets),
    }


def train(args):
    # Load data
    print(f"Loading demo data from {args.data}")
    observations, actions = load_demo_data(args.data)

    # Filter out frames with no meaningful action (warmup standing still)
    filtered_obs = []
    filtered_act = []
    for obs, act in zip(observations, actions):
        speed = (obs.velocity[0]**2 + obs.velocity[1]**2)**0.5
        has_movement = abs(act.forwardmove) > 0.05 or abs(act.sidemove) > 0.05
        has_aim = abs(act.yaw_delta) > 0.1 or abs(act.pitch_delta) > 0.1
        has_action = act.jump or act.attack
        if has_movement or has_aim or has_action or speed > 50:
            filtered_obs.append(obs)
            filtered_act.append(act)

    print(f"Filtered to {len(filtered_obs)} active frames (from {len(observations)} total)")

    if len(filtered_obs) < 10:
        print("Too few active frames! Check demo extraction.")
        sys.exit(1)

    dataset = prepare_dataset(filtered_obs, filtered_act)
    n = len(filtered_obs)

    # Print action distribution stats
    cont = dataset["continuous"].numpy()
    print(f"\nAction statistics:")
    print(f"  yaw_delta (scaled):  mean={cont[:,0].mean():.3f} std={cont[:,0].std():.3f}")
    print(f"  pitch_delta (scaled): mean={cont[:,1].mean():.3f} std={cont[:,1].std():.3f}")
    print(f"  forwardmove:         mean={cont[:,2].mean():.3f} std={cont[:,2].std():.3f}")
    print(f"  sidemove:            mean={cont[:,3].mean():.3f} std={cont[:,3].std():.3f}")
    print(f"  jump rate:           {dataset['jump'].mean():.3f}")
    print(f"  attack rate:         {dataset['attack'].mean():.3f}")

    # Model
    model = ActorCritic(obs_dim=OBS_DIM)

    if args.resume:
        print(f"\nLoading pretrained weights from {args.resume}")
        model.load_state_dict(torch.load(args.resume, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Dataset: {n} samples")

    for epoch in range(args.epochs):
        # Shuffle
        perm = torch.randperm(n)
        total_loss = 0.0
        total_cont_loss = 0.0
        total_binary_loss = 0.0
        n_batches = 0

        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            idx = perm[start:end]

            obs = dataset["obs"][idx]
            cont_target = dataset["continuous"][idx]
            jump_target = dataset["jump"][idx]
            attack_target = dataset["attack"][idx]

            # Forward pass through trunk + actor
            features = model.forward(obs)
            actor_out = model.actor(features)

            # Split outputs (same as model.get_action_and_value)
            cont_means = actor_out[:, :4]
            binary_logits = actor_out[:, 4:7]

            # Continuous loss (MSE on scaled actions)
            cont_loss = mse_loss(cont_means, cont_target)

            # Binary loss (BCE for jump and attack; skip crouch)
            jump_loss = bce_loss(binary_logits[:, 0], jump_target)
            attack_loss = bce_loss(binary_logits[:, 2], attack_target)
            binary_loss = jump_loss + attack_loss

            loss = cont_loss + binary_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_cont_loss += cont_loss.item()
            total_binary_loss += binary_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_cont = total_cont_loss / n_batches
        avg_binary = total_binary_loss / n_batches

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{args.epochs}: loss={avg_loss:.4f} "
                  f"(cont={avg_cont:.4f} binary={avg_binary:.4f})")

    # Save model
    output = args.output
    torch.save(model.state_dict(), output)
    print(f"\nSaved model to {output}")

    # Quick evaluation: sample some predictions
    print("\nSample predictions vs targets:")
    model.eval()
    with torch.no_grad():
        sample_idx = torch.randint(0, n, (5,))
        obs_sample = dataset["obs"][sample_idx]
        cont_target = dataset["continuous"][sample_idx].numpy()

        features = model.forward(obs_sample)
        actor_out = model.actor(features)
        cont_pred = actor_out[:, :4].numpy()
        binary_pred = torch.sigmoid(actor_out[:, 4:7]).numpy()

        for i in range(5):
            print(f"  [{i}] target: yaw={cont_target[i,0]:+.2f} pitch={cont_target[i,1]:+.2f} "
                  f"fwd={cont_target[i,2]:+.2f} side={cont_target[i,3]:+.2f} "
                  f"jump={dataset['jump'][sample_idx[i]]:.0f} attack={dataset['attack'][sample_idx[i]]:.0f}")
            print(f"       pred:   yaw={cont_pred[i,0]:+.2f} pitch={cont_pred[i,1]:+.2f} "
                  f"fwd={cont_pred[i,2]:+.2f} side={cont_pred[i,3]:+.2f} "
                  f"jump={binary_pred[i,0]:.2f} attack={binary_pred[i,2]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning from Q2AI demo data")
    parser.add_argument("--data", required=True, help="Path to demo_training_data.bin")
    parser.add_argument("--output", default="model_bc.pt", help="Output model path")
    parser.add_argument("--resume", default=None, help="Resume from existing model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
