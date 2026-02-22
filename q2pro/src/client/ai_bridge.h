/*
 * ai_bridge.h -- Q2AI headless client shared structures
 *
 * Defines observation, action, and shared state structures used for
 * communication between the C headless client and Python trainer
 * via POSIX shared memory.
 */

#pragma once

#include <stdint.h>

#define Q2AI_MAX_ENTITIES   64
#define Q2AI_SHM_NAME       "/q2ai_state"
#define Q2AI_SEM_OBS_NAME   "/q2ai_obs_ready"
#define Q2AI_SEM_ACT_NAME   "/q2ai_act_ready"

// Entity type classification
#define ENT_TYPE_UNKNOWN    0
#define ENT_TYPE_PLAYER     1
#define ENT_TYPE_ITEM       2
#define ENT_TYPE_PROJECTILE 3

// Per-entity observation
typedef struct {
    float   origin[3];
    float   angles[3];
    int32_t modelindex;
    int32_t effects;
    int32_t solid;
    int32_t frame;
    int32_t skinnum;
    uint8_t active;
    uint8_t entity_type;    // ENT_TYPE_*
    uint8_t _pad[2];
} ai_entity_obs_t;

// Full observation extracted each server frame
typedef struct {
    // Self state (from player_state_t)
    float       origin[3];
    float       velocity[3];
    float       viewangles[3];
    int32_t     health;
    int32_t     armor;
    int32_t     ammo;
    int32_t     current_weapon;
    int32_t     frags;
    int32_t     pm_type;
    int32_t     pm_flags;
    int32_t     inventory[256];     // MAX_ITEMS

    // Visible entities
    int32_t             num_entities;
    ai_entity_obs_t     entities[Q2AI_MAX_ENTITIES];

    // Enemy player tracking (explicit — for 1v1 duel)
    float       enemy_origin[3];    // absolute position
    float       enemy_angles[3];    // view angles
    int32_t     enemy_visible;      // 1 if enemy is in current frame
    int32_t     enemy_client_num;   // client number of enemy, -1 if none

    // Combat feedback
    int32_t     damage_dealt;       // from STAT_FLASHES, resets each frame

    // Spatial awareness: wall distances from raycasts
    // [0-7]   8 horizontal at eye height (relative to yaw: fwd, fwd-R, R, back-R, back, back-L, L, fwd-L)
    // [8]     floor distance
    // [9]     ceiling distance
    // [10-17] 8 horizontal at knee height (same directions, detects low obstacles)
    // Values are trace fraction [0.0-1.0] * 1024 units max range
    float       wall_distances[18];

    // Game info
    char        mapname[64];
    int32_t     serverframe;
    int32_t     servertime;
    int32_t     client_num;
} ai_observation_t;

// Action from Python trainer -> C client
typedef struct {
    float       yaw_delta;          // degrees
    float       pitch_delta;        // degrees
    float       forwardmove;        // -1.0 to 1.0
    float       sidemove;           // -1.0 to 1.0
    uint8_t     jump;               // 0/1
    uint8_t     crouch;             // 0/1
    uint8_t     attack;             // 0/1
    uint8_t     weapon;             // 0-10 (0 = no switch)
} ai_action_t;

// Shared memory layout between C client and Python
typedef struct {
    // Synchronization
    uint32_t    sequence;           // incremented each frame by C side
    uint8_t     episode_done;       // 1 = episode ended (death, match end)
    uint8_t     reset_request;      // Python sets to 1 to request reset
    uint8_t     connected;          // 1 = client is connected to server
    uint8_t     _pad;

    // Reward (computed in C)
    float       reward;
    float       cumulative_reward;

    // Data
    ai_observation_t    obs;
    ai_action_t         action;
} ai_shared_state_t;

// Global action state (read by input.c)
extern ai_action_t q2ai_shared_action;

// Initialize/shutdown the AI bridge
void AI_Bridge_Init(void);
void AI_Bridge_Shutdown(void);
void AI_Bridge_ExchangeFrame(void);

// Extract observations from game state
void AI_Observe_Update(ai_observation_t *obs);

// Apply AI action to client input
void AI_Input_Apply(const ai_action_t *action);

// Compute reward from frame deltas
float AI_Reward_Compute(void);

// Episode lifecycle management
void AI_Control_Init(void);
void AI_Control_Frame(void);

// Demo extraction (behavior cloning)
void AI_Demo_Init(void);
void AI_Demo_Frame(void);
void AI_Demo_Shutdown(void);
