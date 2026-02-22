/*
 * ai_reward.c -- Compute per-frame reward from game state deltas
 *
 * V1.5: Precision-focused rewards. Aim quality scales all combat rewards.
 *
 *   +1.0  per frag
 *   -1.0  per death
 *   -damage/100 for damage taken
 *   +damage/100 for damage dealt (only when enemy in FOV)
 *   +0.15  hit confirmation bonus (only when enemy in FOV)
 *   +0.20  precision bonus (scaled by crosshair accuracy on hit)
 *   +0.01~0.05 graduated aim tracking (scales with dot product)
 *   +0.02~0.10 shooting at enemy in FOV (scales with aim precision)
 *   -0.10  crouch penalty per frame
 *   -0.01  stuck penalty (speed < 20)
 *   -0.15  circling penalty (< 128 units from checkpoint after 30 frames)
 *   -0.005 backward movement penalty (reduced from 0.01)
 *   -0.02  blaster penalty (has better weapon but won't switch)
 *   +0.1   health/armor pickup (reduced from 0.2)
 *   +0.15  weapon pickup (reduced from 0.3)
 *   +0.01  approaching enemy when visible (reduced from 0.03)
 *   item proximity shaping (reduced 3x)
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
static float checkpoint_pos[3];  // position snapshot every N frames
static int checkpoint_timer;     // frames since last checkpoint
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
        VectorCopy(cl.predicted_origin, checkpoint_pos);
        checkpoint_timer = 0;
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
    // Scaled by aim precision: tighter aim = more reward
    int damage_dealt = ps->stats[STAT_FLASHES];
    if (damage_dealt > 0 && enemy_in_fov) {
        reward += damage_dealt / 100.0f;           // base damage signal
        reward += 0.15f;                           // hit confirmation bonus
        // Precision bonus: scale 0→0.2 as dot goes from 0.707→1.0
        float precision = (aim_dot - 0.707f) / (1.0f - 0.707f);  // 0.0 to 1.0
        reward += 0.2f * precision;                // up to +0.2 for crosshair-on hits
    }

    // === MOVEMENT SHAPING (reduced) ===

    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        float vx = cl.predicted_velocity[0];
        float vy = cl.predicted_velocity[1];
        float speed = sqrtf(vx * vx + vy * vy);

        // Stuck penalty
        if (speed < 20.0f) {
            reward -= 0.01f;
        }

        // Backward movement penalty (reduced)
        float yaw_rad = ps->viewangles[1] * (M_PI / 180.0f);
        float cos_yaw = cosf(yaw_rad);
        float sin_yaw = sinf(yaw_rad);
        float forward_vel = vx * cos_yaw + vy * sin_yaw;

        if (forward_vel < -50.0f) {
            reward -= 0.005f;
        }

        // Crouch penalty (harsh — crouching is never good in duel)
        if (q2ai_shared_action.crouch) {
            reward -= 0.10f;
        }

        // Circling penalty: every 30 frames (~3s at 10Hz), check displacement
        checkpoint_timer++;
        if (checkpoint_timer >= 30) {
            float dx = cl.predicted_origin[0] - checkpoint_pos[0];
            float dy = cl.predicted_origin[1] - checkpoint_pos[1];
            float displacement = sqrtf(dx*dx + dy*dy);
            if (displacement < 128.0f) {
                reward -= 0.15f;
            }
            VectorCopy(cl.predicted_origin, checkpoint_pos);
            checkpoint_timer = 0;
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

    // === COMBAT ENGAGEMENT ===

    // Graduated aim tracking reward (uses aim_dot computed earlier)
    // 45° (~0.707): +0.01, 20° (~0.94): +0.03, 10° (~0.985): +0.05
    if (aim_dot > 0.707f) {
        float aim_quality = (aim_dot - 0.707f) / (1.0f - 0.707f);  // 0→1
        reward += 0.01f + 0.04f * aim_quality;  // +0.01 to +0.05
    }

    // Shooting reward — only reward aimed shots, no penalty for spam
    // Let the bot learn "shooting at enemy = good" without fearing the trigger
    if (cur_pm_type == PM_NORMAL && cur_health > 0 && q2ai_shared_action.attack && enemy_in_fov) {
        float aim_quality = (aim_dot - 0.707f) / (1.0f - 0.707f);
        reward += 0.02f + 0.08f * aim_quality;  // +0.02 to +0.10
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

    // Enemy approach reward (reduced from 0.03)
    if (enemy_visible && prev_enemy_dist < 90000.0f) {
        float approach = prev_enemy_dist - enemy_dist;
        if (approach > 0) {
            reward += 0.01f * fminf(approach / 100.0f, 1.0f);
        } else {
            reward -= 0.003f;  // mild retreat penalty
        }
    }
    prev_enemy_dist = enemy_visible ? enemy_dist : 99999.0f;

    // === ITEM PROXIMITY (reduced 3x) ===

    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        float nearest_item = 99999.0f;
        int j;
        for (i = 0; i < cl.frame.numEntities; i++) {
            j = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
            const centity_state_t *es = &cl.entityStates[j];
            if (es->number == cl.frame.clientNum + 1)
                continue;
            if (!(es->s.effects & EF_ROTATE))
                continue;
            float dx = es->s.origin[0] - cl.predicted_origin[0];
            float dy = es->s.origin[1] - cl.predicted_origin[1];
            float dz = es->s.origin[2] - cl.predicted_origin[2];
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < nearest_item)
                nearest_item = dist;
        }
        if (nearest_item < 90000.0f && prev_nearest_item_dist < 90000.0f) {
            float approach = prev_nearest_item_dist - nearest_item;
            reward += fminf(fmaxf(approach * 0.001f, -0.03f), 0.03f);
        }
        prev_nearest_item_dist = nearest_item;
    }

    // Update previous state
    prev_frags = cur_frags;
    prev_health = cur_health;
    prev_armor = cur_armor;
    prev_pm_type = cur_pm_type;
    memcpy(prev_inventory, cl.inventory, sizeof(prev_inventory));

    return reward;
}
