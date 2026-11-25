/*
 * ai_observe.c -- Extract game state into observation struct
 *
 * Reads from cl.frame.ps (player state), cl_entities[] (visible entities),
 * cl.inventory[], and wall raycasts (via CM_BoxTrace) each frame.
 */

#include "client.h"
#include "ai_bridge.h"

#include <math.h>

#define AI_RAY_LENGTH 1024.0f

// Persistent enemy tracking across frames (survives loss of visibility)
static vec3_t   s_last_enemy_origin;
static int      s_last_enemy_servertime;
static bool     s_enemy_ever_seen;

// Debug stats (printed every 5 seconds)
static unsigned s_debug_time;
static int      s_debug_frames;
static int      s_debug_enemy_visible;
static int      s_debug_entities_total;

void AI_OnEnemySound(const vec3_t origin)
{
    VectorCopy(origin, s_last_enemy_origin);
    s_last_enemy_servertime = cl.servertime;
    s_enemy_ever_seen = true;
}

static uint8_t AI_ClassifyEntity(int ent_number, const entity_state_t *s)
{
    int effects = s->effects;

    // Players: entity numbers 1..maxclients, but only live ones.
    // Dead/gibbed players are ignored so KaaN doesn't keep shooting the corpse:
    //   - player_die()/bot death set modelindex2 = 0 (weapon model removed)
    //   - ThrowClientHead() swaps model to a gib head and sets EF_GIB
    //   - live players always have modelindex 255 (skin model) + modelindex2 255
    if (ent_number >= 1 && ent_number <= cl.maxclients) {
        if ((effects & EF_GIB) || s->modelindex != 255 || s->modelindex2 == 0)
            return ENT_TYPE_UNKNOWN;
        return ENT_TYPE_PLAYER;
    }

    // Items: spinning pickups on the ground
    if (effects & EF_ROTATE)
        return ENT_TYPE_ITEM;

    // Projectiles: anything with a trail effect
    if (effects & (EF_BLASTER | EF_ROCKET | EF_GRENADE |
                   EF_HYPERBLASTER | EF_BFG | EF_PLASMA))
        return ENT_TYPE_PROJECTILE;

    return ENT_TYPE_UNKNOWN;
}

void AI_Observe_Update(ai_observation_t *obs)
{
    const player_state_t *ps = &cl.frame.ps;
    int i, j, count;

    memset(obs, 0, sizeof(*obs));

    // Self state from player_state_t
    VectorCopy(cl.predicted_origin, obs->origin);
    VectorCopy(cl.predicted_velocity, obs->velocity);
    VectorCopy(cl.predicted_angles, obs->viewangles);

    obs->health = ps->stats[STAT_HEALTH];
    obs->armor = ps->stats[STAT_ARMOR];
    obs->ammo = ps->stats[STAT_AMMO];
    obs->frags = ps->stats[STAT_FRAGS];
    obs->pm_type = ps->pmove.pm_type;
    obs->pm_flags = ps->pmove.pm_flags;

    // Current weapon from stats
    obs->current_weapon = ps->gunindex;

    // Inventory
    for (i = 0; i < 256 && i < MAX_ITEMS; i++)
        obs->inventory[i] = cl.inventory[i];

    // Enemy tracking init
    obs->enemy_visible = 0;
    obs->enemy_client_num = -1;

    // Visible entities from current frame
    count = 0;
    for (i = 0; i < cl.frame.numEntities && count < Q2AI_MAX_ENTITIES; i++) {
        j = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
        const centity_state_t *es = &cl.entityStates[j];

        // Skip self
        if (es->number == cl.frame.clientNum + 1)
            continue;

        ai_entity_obs_t *eo = &obs->entities[count];

        VectorCopy(es->s.origin, eo->origin);
        VectorCopy(es->s.angles, eo->angles);
        eo->modelindex = es->s.modelindex;
        eo->effects = es->s.effects;
        eo->solid = es->s.solid;
        eo->frame = es->s.frame;
        eo->skinnum = es->s.skinnum;
        eo->active = 1;
        eo->entity_type = AI_ClassifyEntity(es->number, &es->s);

        // Track enemy player (first non-self player entity)
        if (eo->entity_type == ENT_TYPE_PLAYER && !obs->enemy_visible) {
            VectorCopy(es->s.origin, obs->enemy_origin);
            VectorCopy(es->s.angles, obs->enemy_angles);
            obs->enemy_visible = 1;
            obs->enemy_client_num = es->number - 1;
            obs->enemy_weapon = (es->s.skinnum >> 8) & 0xff;  // WEAP_* index
        }

        count++;
    }

    obs->num_entities = count;

    // Debug: periodic enemy detection stats (every 5s)
    s_debug_frames++;
    s_debug_entities_total += cl.frame.numEntities;
    if (obs->enemy_visible)
        s_debug_enemy_visible++;
    if (cl.servertime - s_debug_time > 5000) {
        Com_Printf("Q2AI [OBS] %d frames: enemy_visible=%d/%d (%.0f%%), "
                   "avg_ents=%.1f, maxclients=%d, clientNum=%d, "
                   "enemy_pos=(%.0f %.0f %.0f)\n",
                   s_debug_frames, s_debug_enemy_visible, s_debug_frames,
                   s_debug_frames ? 100.0f * s_debug_enemy_visible / s_debug_frames : 0,
                   s_debug_frames ? (float)s_debug_entities_total / s_debug_frames : 0,
                   cl.maxclients, cl.frame.clientNum,
                   obs->enemy_visible ? obs->enemy_origin[0] : s_last_enemy_origin[0],
                   obs->enemy_visible ? obs->enemy_origin[1] : s_last_enemy_origin[1],
                   obs->enemy_visible ? obs->enemy_origin[2] : s_last_enemy_origin[2]);
        s_debug_time = cl.servertime;
        s_debug_frames = 0;
        s_debug_enemy_visible = 0;
        s_debug_entities_total = 0;
    }

    // Persist last-known enemy position
    if (obs->enemy_visible) {
        // Enemy is visible now — update statics
        VectorCopy(obs->enemy_origin, s_last_enemy_origin);
        s_last_enemy_servertime = cl.servertime;
        s_enemy_ever_seen = true;
        obs->enemy_last_seen_age = 0.0f;
    } else if (s_enemy_ever_seen) {
        // Enemy not visible but was seen/heard before — use last known position
        VectorCopy(s_last_enemy_origin, obs->enemy_origin);
        obs->enemy_last_seen_age = (float)(cl.servertime - s_last_enemy_servertime);
    }

    // Ideal aim deltas: exact yaw/pitch to face enemy (hint for NN)
    obs->enemy_aim_yaw = 0.0f;
    obs->enemy_aim_pitch = 0.0f;
    {
        // Use enemy_origin (either visible now or last-known)
        float ex = obs->enemy_origin[0];
        float ey = obs->enemy_origin[1];
        float ez = obs->enemy_origin[2];
        bool has_pos = (ex != 0.0f || ey != 0.0f || ez != 0.0f);

        if (has_pos && obs->pm_type == PM_NORMAL && obs->health > 0) {
            float dx = ex - cl.predicted_origin[0];
            float dy = ey - cl.predicted_origin[1];
            float dz = ez - cl.predicted_origin[2];

            // Desired angles to face enemy
            float desired_yaw = atan2f(dy, dx) * (180.0f / M_PI);
            float dist_horiz = sqrtf(dx * dx + dy * dy);
            float desired_pitch = -atan2f(dz, dist_horiz) * (180.0f / M_PI);

            // Current angles
            float cur_yaw = cl.predicted_angles[YAW];
            float cur_pitch = cl.predicted_angles[PITCH];

            // Yaw delta (handle wraparound)
            float yaw_diff = desired_yaw - cur_yaw;
            while (yaw_diff > 180.0f)  yaw_diff -= 360.0f;
            while (yaw_diff < -180.0f) yaw_diff += 360.0f;

            obs->enemy_aim_yaw = yaw_diff;
            obs->enemy_aim_pitch = desired_pitch - cur_pitch;
        }
    }

    // Combat feedback: damage dealt this frame (from STAT_FLASHES)
    obs->damage_dealt = ps->stats[STAT_FLASHES];

    // Wall raycasts: 8 eye-height + floor + ceiling + 8 knee-height = 18 rays
    // Gives the agent spatial awareness of walls, floors, and jumpable obstacles
    if (cl.bsp && cl.bsp->nodes) {
        float yaw_rad = cl.predicted_angles[1] * (M_PI / 180.0f);
        vec3_t start, end;
        trace_t trace;

        // Eye-height rays (standing viewheight ~22 units)
        VectorCopy(obs->origin, start);
        start[2] += 22.0f;

        // 8 horizontal rays at eye height, every 45° relative to player yaw
        for (int r = 0; r < 8; r++) {
            float angle = yaw_rad + r * (M_PI / 4.0f);
            end[0] = start[0] + cosf(angle) * AI_RAY_LENGTH;
            end[1] = start[1] + sinf(angle) * AI_RAY_LENGTH;
            end[2] = start[2];

            CM_BoxTrace(&trace, start, end, vec3_origin, vec3_origin,
                        cl.bsp->nodes, MASK_SOLID, cl.csr.extended);
            obs->wall_distances[r] = trace.fraction;
        }

        // Floor ray (straight down from eye height)
        VectorCopy(start, end);
        end[2] = start[2] - AI_RAY_LENGTH;
        CM_BoxTrace(&trace, start, end, vec3_origin, vec3_origin,
                    cl.bsp->nodes, MASK_SOLID, cl.csr.extended);
        obs->wall_distances[8] = trace.fraction;

        // Ceiling ray (straight up)
        VectorCopy(start, end);
        end[2] = start[2] + AI_RAY_LENGTH;
        CM_BoxTrace(&trace, start, end, vec3_origin, vec3_origin,
                    cl.bsp->nodes, MASK_SOLID, cl.csr.extended);
        obs->wall_distances[9] = trace.fraction;

        // Knee-height rays (~8 units above origin = shin/knee level)
        // If knee ray hits but eye ray doesn't → jumpable obstacle
        start[2] = obs->origin[2] + 8.0f;

        for (int r = 0; r < 8; r++) {
            float angle = yaw_rad + r * (M_PI / 4.0f);
            end[0] = start[0] + cosf(angle) * AI_RAY_LENGTH;
            end[1] = start[1] + sinf(angle) * AI_RAY_LENGTH;
            end[2] = start[2];

            CM_BoxTrace(&trace, start, end, vec3_origin, vec3_origin,
                        cl.bsp->nodes, MASK_SOLID, cl.csr.extended);
            obs->wall_distances[10 + r] = trace.fraction;
        }
    }

    // Game info
    Q_strlcpy(obs->mapname, cl.mapname, sizeof(obs->mapname));
    obs->serverframe = cl.frame.number;
    obs->servertime = cl.servertime;
    obs->client_num = cl.clientNum;
}
