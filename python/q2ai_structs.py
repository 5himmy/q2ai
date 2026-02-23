"""
q2ai_structs.py -- ctypes mirrors of ai_bridge.h C structs.

These must exactly match the C struct layout in q2pro/src/client/ai_bridge.h.
"""

import ctypes

# ----- Constants -----

Q2AI_MAX_ENTITIES = 64
MAX_ITEMS = 256

# PM types (pmtype_t)
PM_NORMAL = 0
PM_SPECTATOR = 1
PM_DEAD = 2
PM_GIB = 3
PM_FREEZE = 4

# PM flags
PMF_DUCKED = 1 << 0
PMF_JUMP_HELD = 1 << 1

# Inventory indices (from OpenTDM g_local.h enum, starts with ITEM_NULL=0)
# Armor
INV_BODY_ARMOR = 1
INV_COMBAT_ARMOR = 2
INV_JACKET_ARMOR = 3
INV_ARMOR_SHARD = 4
INV_POWER_SCREEN = 5
INV_POWER_SHIELD = 6

# Weapons (index = inventory slot)
INV_BLASTER = 7
INV_SHOTGUN = 8
INV_SUPER_SHOTGUN = 9
INV_MACHINEGUN = 10
INV_CHAINGUN = 11
INV_GRENADES = 12       # ammo + throwable weapon (same slot in Q2)
INV_GRENADE_LAUNCHER = 13
INV_ROCKET_LAUNCHER = 14
INV_HYPERBLASTER = 15
INV_RAILGUN = 16
INV_BFG10K = 17

# Ammo
INV_SHELLS = 18
INV_BULLETS = 19
INV_CELLS = 20
INV_ROCKETS = 21
INV_SLUGS = 22

# Powerups
INV_QUAD_DAMAGE = 23
INV_INVULNERABILITY = 24

# Grouped indices for observation extraction
WEAPON_INDICES = [
    INV_SHOTGUN, INV_SUPER_SHOTGUN, INV_MACHINEGUN, INV_CHAINGUN,
    INV_GRENADES, INV_GRENADE_LAUNCHER, INV_ROCKET_LAUNCHER,
    INV_HYPERBLASTER, INV_RAILGUN, INV_BFG10K,
]  # 10 weapons (skip blaster, always owned)

AMMO_INDICES = [INV_SHELLS, INV_BULLETS, INV_CELLS, INV_GRENADES, INV_ROCKETS, INV_SLUGS]
AMMO_MAX = [100.0, 200.0, 200.0, 50.0, 50.0, 50.0]

# Number of weapon types (for one-hot encoding, includes blaster)
NUM_WEAPONS = 11  # gunindex 0=none, 1-10=weapons... but gunindex is model index, not item index

# ----- ctypes Structures -----

# Entity type constants (match C ENT_TYPE_*)
ENT_TYPE_UNKNOWN = 0
ENT_TYPE_PLAYER = 1
ENT_TYPE_ITEM = 2
ENT_TYPE_PROJECTILE = 3


class AiEntityObs(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_float * 3),
        ("angles", ctypes.c_float * 3),
        ("modelindex", ctypes.c_int32),
        ("effects", ctypes.c_int32),
        ("solid", ctypes.c_int32),
        ("frame", ctypes.c_int32),
        ("skinnum", ctypes.c_int32),
        ("active", ctypes.c_uint8),
        ("entity_type", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 2),
    ]


class AiObservation(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_float * 3),
        ("velocity", ctypes.c_float * 3),
        ("viewangles", ctypes.c_float * 3),
        ("health", ctypes.c_int32),
        ("armor", ctypes.c_int32),
        ("ammo", ctypes.c_int32),
        ("current_weapon", ctypes.c_int32),
        ("frags", ctypes.c_int32),
        ("pm_type", ctypes.c_int32),
        ("pm_flags", ctypes.c_int32),
        ("inventory", ctypes.c_int32 * MAX_ITEMS),
        ("num_entities", ctypes.c_int32),
        ("entities", AiEntityObs * Q2AI_MAX_ENTITIES),
        ("enemy_origin", ctypes.c_float * 3),
        ("enemy_angles", ctypes.c_float * 3),
        ("enemy_visible", ctypes.c_int32),
        ("enemy_client_num", ctypes.c_int32),
        ("enemy_last_seen_age", ctypes.c_float),
        ("enemy_weapon", ctypes.c_int32),
        ("enemy_aim_yaw", ctypes.c_float),
        ("enemy_aim_pitch", ctypes.c_float),
        ("damage_dealt", ctypes.c_int32),
        ("wall_distances", ctypes.c_float * 18),
        ("mapname", ctypes.c_char * 64),
        ("serverframe", ctypes.c_int32),
        ("servertime", ctypes.c_int32),
        ("client_num", ctypes.c_int32),
    ]


class AiAction(ctypes.Structure):
    _fields_ = [
        ("yaw_delta", ctypes.c_float),
        ("pitch_delta", ctypes.c_float),
        ("forwardmove", ctypes.c_float),
        ("sidemove", ctypes.c_float),
        ("jump", ctypes.c_uint8),
        ("crouch", ctypes.c_uint8),
        ("attack", ctypes.c_uint8),
        ("weapon", ctypes.c_uint8),
    ]


class AiSharedState(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint32),
        ("episode_done", ctypes.c_uint8),
        ("reset_request", ctypes.c_uint8),
        ("connected", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
        ("reward", ctypes.c_float),
        ("cumulative_reward", ctypes.c_float),
        ("obs", AiObservation),
        ("action", AiAction),
    ]


# Verify struct sizes match C side
# AiEntityObs: 3*4 + 3*4 + 5*4 + 1 + 1 + 2 = 48 bytes (unchanged, entity_type uses former _pad[0])
assert ctypes.sizeof(AiEntityObs) == 48, f"AiEntityObs size mismatch: {ctypes.sizeof(AiEntityObs)}"
assert ctypes.sizeof(AiAction) == 20, f"AiAction size mismatch: {ctypes.sizeof(AiAction)}"
# AiObservation grew by: enemy_origin(12) + enemy_angles(12) + enemy_visible(4) + enemy_client_num(4) = 32 bytes
# AiSharedState = 4276 + 32 = 4308
# wall_distances added 18*4 = 72 bytes
assert ctypes.sizeof(AiSharedState) == 4400, f"AiSharedState size mismatch: {ctypes.sizeof(AiSharedState)}"
