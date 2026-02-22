/*
 * ai_reward.c -- Compute per-frame reward from game state deltas
 *
 * V2.0: Simplified rewards — core combat + critical anti-suicide signals.
 * Fewer signals = lower value loss = better learning.
 *
 *   +1.0  per frag
 *   -1.0  per death
 *   -damage/100 for damage taken
 *   +damage/100 for damage dealt (enemy in FOV)
 *   -0.15  self-splash penalty (shooting wall within 120u)
 *   -0.02  blaster penalty (has better weapon but won't switch)
 *   -0.01  stuck penalty (speed < 20)
 *   +0.1   health/armor pickup
 *   +0.15  weapon pickup
 */

#include "client.h"
#include "ai_bridge.h"

#include <math.h>

static int prev_frags;
static int prev_health;
static int prev_armor;
static int prev_pm_type;
static int prev_inventory[MAX_ITEMS];
static float prev_nearest_item_dist;
static float prev_enemy_dist;
static bool reward_initialized;

float AI_Reward_Compute(void)
{
    const player_state_t *ps = &cl.frame.ps;
    float reward = 0.0f;
    int i;

    if (!reward_initialized) {
        prev_frags = ps->stats[STAT_FRAGS];
        prev_health = ps->stats[STAT_HEALTH];
        prev_armor = ps->stats[STAT_ARMOR];
        prev_pm_type = ps->pmove.pm_type;
        prev_nearest_item_dist = 99999.0f;
        prev_enemy_dist = 99999.0f;
        memcpy(prev_inventory, cl.inventory, sizeof(prev_inventory));
        reward_initialized = true;
        return 0.0f;
    }

    int cur_frags = ps->stats[STAT_FRAGS];
    int cur_health = ps->stats[STAT_HEALTH];
    int cur_armor = ps->stats[STAT_ARMOR];
    int cur_pm_type = ps->pmove.pm_type;

    // === COMBAT REWARDS (primary) ===

    // Frag reward
    if (cur_frags > prev_frags) {
        reward += 1.0f * (cur_frags - prev_frags);
    }

    // Death penalty
    if ((cur_pm_type == PM_DEAD || cur_pm_type == PM_GIB) &&
        prev_pm_type != PM_DEAD && prev_pm_type != PM_GIB) {
        reward -= 1.0f;
    }

    // Damage taken penalty
    if (cur_health < prev_health && prev_health > 0 && cur_pm_type == PM_NORMAL) {
        int damage = prev_health - cur_health;
        reward -= damage / 100.0f;
    }

    // === ENEMY DETECTION (needed before damage reward) ===

    // Find enemy player
    bool enemy_visible = false;
    float enemy_dist = 99999.0f;
    float enemy_dx = 0, enemy_dy = 0, enemy_dz = 0;
    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        for (i = 0; i < cl.frame.numEntities; i++) {
            int k = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
            const centity_state_t *es = &cl.entityStates[k];
            if (es->number >= 1 && es->number <= cl.maxclients &&
                es->number != cl.frame.clientNum + 1) {
                enemy_visible = true;
                enemy_dx = es->s.origin[0] - cl.predicted_origin[0];
                enemy_dy = es->s.origin[1] - cl.predicted_origin[1];
                enemy_dz = es->s.origin[2] - cl.predicted_origin[2];
                enemy_dist = sqrtf(enemy_dx*enemy_dx + enemy_dy*enemy_dy + enemy_dz*enemy_dz);
                break;
            }
        }
    }

    // Compute aim dot product (how close crosshair is to enemy)
    // dot=1.0 = perfect aim, dot=0.707 = 45°, dot=0 = 90°, dot<0 = behind
    float aim_dot = -1.0f;
    if (enemy_visible && enemy_dist > 1.0f) {
        float yaw_r = ps->viewangles[1] * (M_PI / 180.0f);
        float pitch_r = ps->viewangles[0] * (M_PI / 180.0f);
        float vx2 = cosf(pitch_r) * cosf(yaw_r);
        float vy2 = cosf(pitch_r) * sinf(yaw_r);
        float vz2 = -sinf(pitch_r);
        float inv_d = 1.0f / enemy_dist;
        aim_dot = vx2 * (enemy_dx * inv_d) + vy2 * (enemy_dy * inv_d) + vz2 * (enemy_dz * inv_d);
    }
    bool enemy_in_fov = (aim_dot > 0.707f);  // cos(45°)

    // Damage dealt reward (from STAT_FLASHES, set by opentdm)
    int damage_dealt = ps->stats[STAT_FLASHES];
    if (damage_dealt > 0 && enemy_in_fov) {
        reward += damage_dealt / 100.0f;
    }

    // === MOVEMENT ===

    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        float vx = cl.predicted_velocity[0];
        float vy = cl.predicted_velocity[1];
        float speed = sqrtf(vx * vx + vy * vy);

        // Stuck penalty
        if (speed < 20.0f) {
            reward -= 0.01f;
        }
    }

    // === ITEM REWARDS (reduced) ===

    // Health/armor pickup bonus
    if (cur_health > prev_health && cur_health > 0 && prev_health > 0) {
        reward += 0.1f;
    }
    if (cur_armor > prev_armor) {
        reward += 0.1f;
    }

    // Weapon pickup bonus
    for (i = 0; i < MAX_ITEMS; i++) {
        if (cl.inventory[i] > prev_inventory[i]) {
            reward += 0.15f;
            break;
        }
    }

    // Self-splash penalty: shooting when too close to a wall causes self-damage
    // Cast a short forward ray to check wall proximity
    if (cur_pm_type == PM_NORMAL && cur_health > 0 && q2ai_shared_action.attack && cl.bsp && cl.bsp->nodes) {
        float yaw_r = ps->viewangles[1] * (M_PI / 180.0f);
        float pitch_r = ps->viewangles[0] * (M_PI / 180.0f);
        vec3_t ray_start, ray_end;
        trace_t tr;
        VectorCopy(cl.predicted_origin, ray_start);
        ray_start[2] += 22.0f;
        ray_end[0] = ray_start[0] + cosf(pitch_r) * cosf(yaw_r) * 120.0f;
        ray_end[1] = ray_start[1] + cosf(pitch_r) * sinf(yaw_r) * 120.0f;
        ray_end[2] = ray_start[2] - sinf(pitch_r) * 120.0f;
        CM_BoxTrace(&tr, ray_start, ray_end, vec3_origin, vec3_origin,
                    cl.bsp->nodes, MASK_SOLID, cl.csr.extended);
        if (tr.fraction < 1.0f) {
            reward -= 0.15f;  // wall within 120u (rocket splash radius) in aim direction
        }
    }

    // Blaster penalty: if holding blaster (item 7) but owns any better weapon
    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        int selected = ps->stats[STAT_SELECTED_ITEM];
        if (selected == 7) {
            if (cl.inventory[8] || cl.inventory[9] || cl.inventory[10] ||
                cl.inventory[11] || cl.inventory[13] || cl.inventory[14] ||
                cl.inventory[15] || cl.inventory[16]) {
                reward -= 0.02f;
            }
        }
    }

    prev_enemy_dist = enemy_visible ? enemy_dist : 99999.0f;

    // Update previous state
    prev_frags = cur_frags;
    prev_health = cur_health;
    prev_armor = cur_armor;
    prev_pm_type = cur_pm_type;
    memcpy(prev_inventory, cl.inventory, sizeof(prev_inventory));

    return reward;
}
