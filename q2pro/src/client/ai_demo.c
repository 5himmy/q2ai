/*
 * ai_demo.c -- Extract training data from MVD demo playback
 *
 * During MVD playback (mvdplay), the client connects as a spectator
 * and chases a player. This module infers player actions from
 * frame-to-frame state deltas (view angles, velocity, weapon state)
 * and writes (observation, action) pairs to a binary file for
 * behavior cloning.
 *
 * Usage: ./q2ai +set q2ai_demo 1 +mvdplay demos/filename.mvd2
 */

#include "client.h"
#include "ai_bridge.h"

#include <stdio.h>
#include <math.h>

static cvar_t *q2ai_demo;
static bool demo_active;
static bool demo_was_active;   // true after first frame recorded
static FILE *demo_out;
static int demo_frames;
static int last_demo_frame;
static int stale_count;        // frames since last new server frame

// Previous frame state for computing deltas
static float prev_viewangles[3];
static float prev_origin[3];
static int prev_gunframe;
static int prev_pm_flags;

// Normalize angle difference to [-180, 180]
static float angle_delta(float cur, float prev)
{
    float d = cur - prev;
    while (d > 180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

// Get origin from player state
// During demo playback (both MVD and DM2), cl.predicted_origin is not valid
// because there are no usercmds to predict from. Always use pmove.origin.
static void get_player_origin(const player_state_t *ps, float *out)
{
    out[0] = SHORT2COORD(ps->pmove.origin[0]);
    out[1] = SHORT2COORD(ps->pmove.origin[1]);
    out[2] = SHORT2COORD(ps->pmove.origin[2]);
}


void AI_Demo_Init(void)
{
    q2ai_demo = Cvar_Get("q2ai_demo", "0", 0);
    demo_active = false;
    demo_out = NULL;
    demo_frames = 0;
    last_demo_frame = -1;
}

static void demo_start(void)
{
    demo_out = fopen("demo_training_data.bin", "wb");
    if (!demo_out) {
        Com_EPrintf("Q2AI Demo: Failed to open output file\n");
        return;
    }

    // Write header: obs_size, action_size
    int32_t obs_size = sizeof(ai_observation_t);
    int32_t act_size = sizeof(ai_action_t);
    fwrite(&obs_size, sizeof(int32_t), 1, demo_out);
    fwrite(&act_size, sizeof(int32_t), 1, demo_out);

    demo_active = true;
    demo_was_active = false;
    demo_frames = 0;
    last_demo_frame = -1;
    stale_count = 0;
    memset(prev_viewangles, 0, sizeof(prev_viewangles));
    memset(prev_origin, 0, sizeof(prev_origin));
    prev_gunframe = 0;
    prev_pm_flags = 0;

    Com_Printf("Q2AI Demo: Recording training data from demo\n");
}

void AI_Demo_Frame(void)
{
    if (!q2ai_demo || !q2ai_demo->integer)
        return;

    // Lazy init: start recording on first active frame
    if (!demo_active)
        demo_start();

    if (!demo_out)
        return;

    // Only process new server frames
    if (cl.frame.number == last_demo_frame) {
        // Track stale frames to detect demo end
        if (demo_was_active && ++stale_count > 100) {
            Com_Printf("Q2AI Demo: No new frames for %d ticks, demo ended.\n",
                        stale_count);
            AI_Demo_Shutdown();
            Cbuf_AddText(&cmd_buffer, "quit\n");
            return;
        }
        return;
    }
    stale_count = 0;
    last_demo_frame = cl.frame.number;

    const player_state_t *ps = &cl.frame.ps;

    // Skip if dead (PM_DEAD/PM_GIB) — but allow PM_FREEZE (MVD chase)
    if (ps->pmove.pm_type == PM_DEAD || ps->pmove.pm_type == PM_GIB)
        return;
    if (ps->stats[STAT_HEALTH] <= 0)
        return;

    // Current state from player_state_t
    float cur_angles[3];
    cur_angles[0] = ps->viewangles[0];  // pitch
    cur_angles[1] = ps->viewangles[1];  // yaw
    cur_angles[2] = ps->viewangles[2];  // roll

    float cur_origin[3];
    get_player_origin(ps, cur_origin);

    // Compute velocity from position deltas (MVD doesn't transmit velocity)
    // Q2 server runs at 10Hz, so velocity = delta_pos / 0.1 = delta_pos * 10
    float cur_velocity[3];
    cur_velocity[0] = (cur_origin[0] - prev_origin[0]) * 10.0f;
    cur_velocity[1] = (cur_origin[1] - prev_origin[1]) * 10.0f;
    cur_velocity[2] = (cur_origin[2] - prev_origin[2]) * 10.0f;

    if (demo_frames == 0) {
        // First frame — just store state, no delta yet
        VectorCopy(cur_angles, prev_viewangles);
        VectorCopy(cur_origin, prev_origin);
        prev_gunframe = ps->gunframe;
        prev_pm_flags = ps->pmove.pm_flags;
        demo_frames = -1;  // sentinel: first frame stored
        return;
    }
    if (demo_frames == -1)
        demo_frames = 0;  // reset sentinel after first real frame

    // Extract observation (with MVD-compatible origin/velocity)
    ai_observation_t obs;
    memset(&obs, 0, sizeof(obs));

    // Self state — use pmove values directly for MVD compatibility
    VectorCopy(cur_origin, obs.origin);
    VectorCopy(cur_velocity, obs.velocity);
    VectorCopy(cur_angles, obs.viewangles);

    obs.health = ps->stats[STAT_HEALTH];
    obs.armor = ps->stats[STAT_ARMOR];
    obs.ammo = ps->stats[STAT_AMMO];
    obs.frags = ps->stats[STAT_FRAGS];
    obs.pm_type = ps->pmove.pm_type;
    obs.pm_flags = ps->pmove.pm_flags;
    obs.current_weapon = ps->gunindex;

    // Inventory
    for (int i = 0; i < 256 && i < MAX_ITEMS; i++)
        obs.inventory[i] = cl.inventory[i];

    // Entities from current frame
    obs.enemy_visible = 0;
    obs.enemy_client_num = -1;
    int count = 0;
    for (int i = 0; i < cl.frame.numEntities && count < Q2AI_MAX_ENTITIES; i++) {
        int j = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
        const centity_state_t *es = &cl.entityStates[j];

        // Skip the followed player (self)
        if (es->number == cl.frame.clientNum + 1)
            continue;

        ai_entity_obs_t *eo = &obs.entities[count];
        VectorCopy(es->s.origin, eo->origin);
        VectorCopy(es->s.angles, eo->angles);
        eo->modelindex = es->s.modelindex;
        eo->effects = es->s.effects;
        eo->solid = es->s.solid;
        eo->frame = es->s.frame;
        eo->skinnum = es->s.skinnum;
        eo->active = 1;

        // Classify entity type
        if (es->number >= 1 && es->number <= cl.maxclients)
            eo->entity_type = ENT_TYPE_PLAYER;
        else if (es->s.effects & EF_ROTATE)
            eo->entity_type = ENT_TYPE_ITEM;
        else if (es->s.effects & (EF_BLASTER | EF_ROCKET | EF_GRENADE |
                                   EF_HYPERBLASTER | EF_BFG | EF_PLASMA))
            eo->entity_type = ENT_TYPE_PROJECTILE;
        else
            eo->entity_type = ENT_TYPE_UNKNOWN;

        // Track enemy player
        if (eo->entity_type == ENT_TYPE_PLAYER && !obs.enemy_visible) {
            VectorCopy(es->s.origin, obs.enemy_origin);
            VectorCopy(es->s.angles, obs.enemy_angles);
            obs.enemy_visible = 1;
            obs.enemy_client_num = es->number - 1;
        }

        count++;
    }
    obs.num_entities = count;

    // Game info
    Q_strlcpy(obs.mapname, cl.mapname, sizeof(obs.mapname));
    obs.serverframe = cl.frame.number;
    obs.servertime = cl.servertime;
    obs.client_num = cl.clientNum;

    // Infer action from state deltas
    ai_action_t action;
    memset(&action, 0, sizeof(action));

    // View angle deltas
    action.yaw_delta = angle_delta(cur_angles[1], prev_viewangles[1]);
    action.pitch_delta = angle_delta(cur_angles[0], prev_viewangles[0]);

    // Clamp to our action ranges
    action.yaw_delta = Q_clipf(action.yaw_delta, -10.0f, 10.0f);
    action.pitch_delta = Q_clipf(action.pitch_delta, -5.0f, 5.0f);

    // Infer movement from velocity relative to view direction
    float yaw_rad = cur_angles[1] * (M_PI / 180.0f);
    float cos_yaw = cosf(yaw_rad);
    float sin_yaw = sinf(yaw_rad);

    // Project velocity onto forward and right vectors
    float vx = cur_velocity[0];
    float vy = cur_velocity[1];
    float forward_vel = vx * cos_yaw + vy * sin_yaw;
    float side_vel = -vx * sin_yaw + vy * cos_yaw;

    // Normalize to [-1, 1] range (max speed ~400)
    action.forwardmove = Q_clipf(forward_vel / 300.0f, -1.0f, 1.0f);
    action.sidemove = Q_clipf(side_vel / 300.0f, -1.0f, 1.0f);

    // Jump: detect upward velocity when jump was just pressed
    if (cur_velocity[2] > 100.0f && !(prev_pm_flags & PMF_JUMP_HELD))
        action.jump = 1;

    // Attack: detect weapon firing (gunframe changes indicate firing animation)
    if (ps->gunframe != prev_gunframe && ps->gunframe > 0)
        action.attack = 1;

    // Write (obs, action) pair
    fwrite(&obs, sizeof(obs), 1, demo_out);
    fwrite(&action, sizeof(action), 1, demo_out);
    demo_frames++;
    demo_was_active = true;

    if (demo_frames % 100 == 0) {
        Com_Printf("Q2AI Demo: %d frames recorded\n", demo_frames);
        fflush(demo_out);
    }

    // Update previous state
    VectorCopy(cur_angles, prev_viewangles);
    VectorCopy(cur_origin, prev_origin);
    prev_gunframe = ps->gunframe;
    prev_pm_flags = ps->pmove.pm_flags;
}

void AI_Demo_Shutdown(void)
{
    if (demo_out) {
        fflush(demo_out);
        fclose(demo_out);
        demo_out = NULL;
        Com_Printf("Q2AI Demo: Saved %d frames to demo_training_data.bin\n",
                    demo_frames);
    }
}
