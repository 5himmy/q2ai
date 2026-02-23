/*
 * ai_reward.c -- Compute per-frame reward from game state deltas
 *
 * V3.0 Stage 1: Aim-only training against stationary dummy.
 * Smooth gradient rewards — the closer the crosshair, the better.
 *
 *   +aim_dot * 0.02   facing reward (continuous, any facing toward enemy)
 *   +aim_dot * 0.05   shooting reward (continuous, better aim = more reward)
 *   +damage/50        damage dealt
 *   +1.0              per frag
 *   -1.0              per death
 */

#include "client.h"
#include "ai_bridge.h"

#include <math.h>

static int prev_frags;
static int prev_health;
static int prev_pm_type;
static bool reward_initialized;

float AI_Reward_Compute(void)
{
    const player_state_t *ps = &cl.frame.ps;
    float reward = 0.0f;
    int i;

    if (!reward_initialized) {
        prev_frags = ps->stats[STAT_FRAGS];
        prev_health = ps->stats[STAT_HEALTH];
        prev_pm_type = ps->pmove.pm_type;
        reward_initialized = true;
        return 0.0f;
    }

    int cur_frags = ps->stats[STAT_FRAGS];
    int cur_health = ps->stats[STAT_HEALTH];
    int cur_pm_type = ps->pmove.pm_type;

    // === COMBAT REWARDS ===

    // Frag reward
    if (cur_frags > prev_frags) {
        reward += 1.0f * (cur_frags - prev_frags);
    }

    // Death penalty
    if ((cur_pm_type == PM_DEAD || cur_pm_type == PM_GIB) &&
        prev_pm_type != PM_DEAD && prev_pm_type != PM_GIB) {
        reward -= 1.0f;
    }

    // === ENEMY DETECTION ===

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

    // Aim dot product: 1.0 = perfect aim, 0 = 90°, -1.0 = behind
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

    // === AIM REWARDS (smooth gradient) ===

    if (enemy_visible && cur_pm_type == PM_NORMAL && cur_health > 0) {
        float facing = aim_dot > 0.0f ? aim_dot : 0.0f;  // 0 to 1

        // Facing reward: any turn toward enemy is good, closer = better
        reward += facing * 0.02f;

        // Shooting reward: shooting while facing enemy, scaled by aim quality
        if (q2ai_shared_action.attack) {
            reward += facing * 0.05f;
        }
    }

    // Damage dealt reward
    int damage_dealt = ps->stats[STAT_FLASHES];
    if (damage_dealt > 0) {
        reward += damage_dealt / 50.0f;
    }

    // Update previous state
    prev_frags = cur_frags;
    prev_health = cur_health;
    prev_pm_type = cur_pm_type;

    return reward;
}
