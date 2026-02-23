"""
q2ai_env.py -- Gymnasium environment wrapping the Q2AI shared memory bridge.

Connects to a running q2ai headless client via POSIX shared memory and
semaphores. Each step() sends an action and receives the next observation.

Observation: 74-float vector (self state + inventory + nearest entities + game)
Action: dict with yaw_delta, pitch_delta, forwardmove, sidemove, jump, crouch, attack, weapon
"""

import ctypes
import math
import mmap

import gymnasium as gym
import numpy as np
import posix_ipc

from q2ai_structs import (
    AiAction,
    AiSharedState,
    AMMO_INDICES,
    AMMO_MAX,
    ENT_TYPE_ITEM,
    ENT_TYPE_PLAYER,
    ENT_TYPE_PROJECTILE,
    PM_DEAD,
    PM_GIB,
    PM_NORMAL,
    WEAPON_INDICES,
)

# Observation vector size
OBS_DIM = 111

# Observation vector layout:
#  0-2:   origin / 4096                          (self position)
#  3-5:   velocity / 400                         (self velocity)
#  6-7:   sin(yaw), cos(yaw)
#  8-9:   sin(pitch), cos(pitch)
#  10:    health / 100
#  11:    armor / 200
#  12:    ammo / 50
#  13-23: weapon one-hot (11 slots)
#  24:    is_alive
#  25-34: weapons owned (10 binary)
#  35-40: ammo counts normalized (6 floats)
#  --- Enemy player (explicit) ---
#  41:    enemy_visible (binary)
#  42-44: enemy relative position / 4096
#  45:    enemy distance / 4096
#  46-47: sin(angle_to_enemy), cos(angle_to_enemy)
#  48:    enemy_weapon / 11 (WEAP_* index normalized)
#  --- Nearest entities (8 slots x 5) ---
#  49-88: entities (8 slots x 5: rel_x, rel_y, rel_z, is_active, entity_type_norm)
#  --- Game ---
#  89:    frags / 20
#  90:    enemy_frags / 20  (TODO: needs scoreboard data)
#  91:    match_time_frac   (TODO: needs match timer)
#  92:    damage_dealt / 100 (from STAT_FLASHES, per frame)
#  --- Wall raycasts (18 rays) ---
#  93-100: 8 horizontal at eye height (fwd, fwd-R, R, back-R, back, back-L, L, fwd-L)
#  101:    floor distance
#  102:    ceiling distance
#  103-110: 8 horizontal at knee height (same directions, detects low obstacles)
#  All values are trace fraction [0.0=touching, 1.0=clear] over 1024 units


class Q2AIEnv(gym.Env):
    """Gymnasium environment for Q2AI headless client."""

    metadata = {"render_modes": []}

    def __init__(self, instance=0, obs_timeout=5.0):
        super().__init__()
        self.instance = instance
        self.obs_timeout = obs_timeout

        # Shared memory
        shm_name = f"/q2ai_state_{instance}"
        self._shm = posix_ipc.SharedMemory(shm_name)
        self._buf = mmap.mmap(self._shm.fd, ctypes.sizeof(AiSharedState))
        self._shm.close_fd()

        # Semaphores
        self._sem_obs = posix_ipc.Semaphore(f"/q2ai_obs_{instance}")
        self._sem_act = posix_ipc.Semaphore(f"/q2ai_act_{instance}")

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # Action space for documentation (model outputs raw values)
        self.action_space = gym.spaces.Dict({
            "yaw_delta": gym.spaces.Box(-10.0, 10.0, shape=(), dtype=np.float32),
            "pitch_delta": gym.spaces.Box(-5.0, 5.0, shape=(), dtype=np.float32),
            "forwardmove": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "sidemove": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "jump": gym.spaces.Discrete(2),
            "crouch": gym.spaces.Discrete(2),
            "attack": gym.spaces.Discrete(2),
            "weapon": gym.spaces.Discrete(11),
        })

        self._prev_sequence = 0
        self._episode_steps = 0

        # Drain stale semaphore counts
        self._drain_semaphore(self._sem_obs)
        self._drain_semaphore(self._sem_act)

    def _drain_semaphore(self, sem):
        """Drain any accumulated semaphore posts."""
        while True:
            try:
                sem.acquire(timeout=0)
            except posix_ipc.BusyError:
                break

    def _read_state(self):
        """Read AiSharedState from shared memory."""
        return AiSharedState.from_buffer_copy(self._buf)

    def _write_action(self, action_dict):
        """Write action into shared memory at the action field offset."""
        act = AiAction()
        act.yaw_delta = float(np.clip(action_dict["yaw_delta"], -10.0, 10.0))
        act.pitch_delta = float(np.clip(action_dict["pitch_delta"], -5.0, 5.0))
        act.forwardmove = float(np.clip(action_dict["forwardmove"], -1.0, 1.0))
        act.sidemove = float(np.clip(action_dict["sidemove"], -1.0, 1.0))
        act.jump = int(action_dict.get("jump", 0)) & 1
        act.crouch = int(action_dict.get("crouch", 0)) & 1
        act.attack = int(action_dict.get("attack", 0)) & 1
        act.weapon = int(action_dict.get("weapon", 0)) % 11

        # Write action bytes at the correct offset in shared memory
        offset = ctypes.sizeof(AiSharedState) - ctypes.sizeof(AiAction)
        self._buf[offset:offset + ctypes.sizeof(AiAction)] = bytes(act)

    def _extract_obs(self, state):
        """Convert raw AiSharedState into observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        o = state.obs

        # Self position and velocity (normalized)
        obs[0] = o.origin[0] / 4096.0
        obs[1] = o.origin[1] / 4096.0
        obs[2] = o.origin[2] / 4096.0
        obs[3] = o.velocity[0] / 400.0
        obs[4] = o.velocity[1] / 400.0
        obs[5] = o.velocity[2] / 400.0

        # View angles as sin/cos to avoid discontinuity
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

        # Weapon one-hot (11 slots: index 0=no weapon, 1-10=weapons)
        weapon_idx = o.current_weapon
        if 0 <= weapon_idx <= 10:
            obs[13 + weapon_idx] = 1.0

        # Alive flag
        obs[24] = 1.0 if (o.pm_type == PM_NORMAL and o.health > 0) else 0.0

        # Weapons owned (10 binary flags)
        for i, inv_idx in enumerate(WEAPON_INDICES):
            obs[25 + i] = 1.0 if o.inventory[inv_idx] > 0 else 0.0

        # Ammo counts (6 floats, normalized)
        for i, (inv_idx, max_val) in enumerate(zip(AMMO_INDICES, AMMO_MAX)):
            obs[35 + i] = o.inventory[inv_idx] / max_val

        # Enemy player (explicit tracking from C side, with last-known persistence)
        self_origin = np.array([o.origin[0], o.origin[1], o.origin[2]])
        enemy_pos = np.array([o.enemy_origin[0], o.enemy_origin[1], o.enemy_origin[2]])
        has_enemy_pos = enemy_pos[0] != 0.0 or enemy_pos[1] != 0.0 or enemy_pos[2] != 0.0

        if o.enemy_visible:
            obs[41] = 1.0  # visible now
        elif has_enemy_pos:
            # Last-known position: freshness decays over 10 seconds
            age_s = o.enemy_last_seen_age / 1000.0
            obs[41] = max(0.0, 1.0 - age_s / 10.0)

        if has_enemy_pos:
            rel_enemy = enemy_pos - self_origin
            obs[42] = rel_enemy[0] / 4096.0
            obs[43] = rel_enemy[1] / 4096.0
            obs[44] = rel_enemy[2] / 4096.0
            dist = np.linalg.norm(rel_enemy)
            obs[45] = min(dist / 4096.0, 1.0)
            # Angle to enemy (relative to our yaw)
            angle_to_enemy = math.atan2(rel_enemy[1], rel_enemy[0]) - yaw_rad
            obs[46] = math.sin(angle_to_enemy)
            obs[47] = math.cos(angle_to_enemy)

        # Enemy weapon (WEAP_* 1-11, normalized)
        obs[48] = o.enemy_weapon / 11.0

        # Entities: 8 nearest, 5 features each (rel_x, rel_y, rel_z, is_active, type)
        entities = []
        for j in range(o.num_entities):
            e = o.entities[j]
            if not e.active:
                continue
            e_origin = np.array([e.origin[0], e.origin[1], e.origin[2]])
            rel = e_origin - self_origin
            dist = np.linalg.norm(rel)
            etype = e.entity_type / 3.0  # normalize: 0=unknown, 0.33=player, 0.67=item, 1.0=projectile
            entities.append((dist, rel, etype))

        # Sort by distance, take 8 nearest
        entities.sort(key=lambda x: x[0])
        for i, (dist, rel, etype) in enumerate(entities[:8]):
            base = 49 + i * 5
            obs[base] = rel[0] / 4096.0
            obs[base + 1] = rel[1] / 4096.0
            obs[base + 2] = rel[2] / 4096.0
            obs[base + 3] = 1.0
            obs[base + 4] = etype

        # Game state
        obs[89] = o.frags / 20.0
        obs[90] = 0.0  # enemy frags (TODO: scoreboard)
        obs[91] = 0.0  # match time fraction (TODO)
        obs[92] = o.damage_dealt / 100.0  # damage dealt this frame

        # Wall raycasts (18 rays, already normalized 0.0-1.0 from C side)
        for i in range(18):
            obs[93 + i] = o.wall_distances[i]

        return obs

    def send_action(self, action):
        """Non-blocking: write action and signal C client."""
        self._write_action(action)
        self._sem_act.release()

    def recv_obs(self):
        """Blocking: wait for next observation from C client."""
        try:
            self._sem_obs.acquire(timeout=self.obs_timeout)
        except posix_ipc.BusyError:
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {"timeout": True}

        state = self._read_state()
        obs = self._extract_obs(state)
        reward = float(state.reward)
        terminated = bool(state.episode_done)
        truncated = False
        self._episode_steps += 1

        info = {
            "sequence": state.sequence,
            "cumulative_reward": float(state.cumulative_reward),
            "frags": state.obs.frags,
            "health": state.obs.health,
            "episode_steps": self._episode_steps,
        }

        return obs, reward, terminated, truncated, info

    def step(self, action):
        """Send action, wait for next observation."""
        self.send_action(action)
        return self.recv_obs()

    def reset(self, seed=None, options=None):
        """Wait for the agent to be alive and ready."""
        super().reset(seed=seed)
        self._episode_steps = 0

        # Signal reset request to C side
        reset_offset = 5  # offset of reset_request in AiSharedState (after sequence + episode_done)
        self._buf[reset_offset] = 1

        # Drain stale semaphore counts
        self._drain_semaphore(self._sem_obs)
        self._drain_semaphore(self._sem_act)

        # Wait for agent to be alive (PM_NORMAL with health > 0)
        for attempt in range(100):
            # Send no-op action to keep the loop going
            noop = {"yaw_delta": 0, "pitch_delta": 0, "forwardmove": 0,
                    "sidemove": 0, "jump": 0, "crouch": 0, "attack": 0, "weapon": 0}
            self._write_action(noop)
            self._sem_act.release()

            try:
                self._sem_obs.acquire(timeout=self.obs_timeout)
            except posix_ipc.BusyError:
                continue

            state = self._read_state()
            if (state.connected and
                    state.obs.pm_type == PM_NORMAL and
                    state.obs.health > 0 and
                    not state.episode_done):
                obs = self._extract_obs(state)
                info = {
                    "sequence": state.sequence,
                    "frags": state.obs.frags,
                    "health": state.obs.health,
                }
                return obs, info

        # Fallback: return whatever we have
        state = self._read_state()
        return self._extract_obs(state), {"sequence": state.sequence}

    def close(self):
        """Clean up shared memory (don't unlink — C side owns it)."""
        if hasattr(self, "_buf") and self._buf:
            self._buf.close()
        if hasattr(self, "_sem_obs") and self._sem_obs:
            self._sem_obs.close()
        if hasattr(self, "_sem_act") and self._sem_act:
            self._sem_act.close()
