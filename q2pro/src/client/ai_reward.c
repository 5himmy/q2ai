/*
 * ai_reward.c -- Compute per-frame reward from game state deltas
 *
 * Design: Combat and items are the primary rewards (+1.0 frag, +0.2-0.3 pickup).
 * Per-frame shaping is minimal and directional — no reward for aimless circling.
 *
 *   +1.0  per frag
 *   -1.0  per death
 *   -damage/100 for damage taken
 *   -0.10  crouch penalty per frame
 *   -0.01  stuck penalty (speed < 20)
 *   -0.15  circling penalty (< 128 units from checkpoint after 30 frames)
 *   -0.01  backward movement penalty
 *   -0.15  blind-fire penalty
 *   +0.02  shooting at visible enemy
 *   -0.02  blaster penalty (has better weapon but won't switch)
 *   +0.2   health/armor pickup
 *   +0.3   weapon pickup
 *   +0.01  approaching enemy when visible
 *   item proximity shaping (approach items)
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

    // === MOVEMENT SHAPING (no free rewards for circling) ===

    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        float vx = cl.predicted_velocity[0];
        float vy = cl.predicted_velocity[1];
        float speed = sqrtf(vx * vx + vy * vy);

        // Stuck penalty
        if (speed < 20.0f) {
            reward -= 0.01f;
        }

        // Backward movement penalty: project velocity onto view direction
        // If moving backward (away from where you're looking), penalize
        float yaw_rad = ps->viewangles[1] * (M_PI / 180.0f);
        float cos_yaw = cosf(yaw_rad);
        float sin_yaw = sinf(yaw_rad);
        float forward_vel = vx * cos_yaw + vy * sin_yaw;

        if (forward_vel < -50.0f) {
            // Moving backward at significant speed
            reward -= 0.01f;
        }

        // Crouch penalty (harsh — crouching is never good in duel)
        if (q2ai_shared_action.crouch) {
            reward -= 0.10f;
        }

        // Circling penalty: every 30 frames (~3s at 10Hz), check displacement
        // from checkpoint. If less than 128 units away, penalize.
        checkpoint_timer++;
        if (checkpoint_timer >= 30) {
            float dx = cl.predicted_origin[0] - checkpoint_pos[0];
            float dy = cl.predicted_origin[1] - checkpoint_pos[1];
            float displacement = sqrtf(dx*dx + dy*dy);
            if (displacement < 128.0f) {
                // Circling in place — penalize
                reward -= 0.15f;
            }
            VectorCopy(cl.predicted_origin, checkpoint_pos);
            checkpoint_timer = 0;
        }
    }

    // === ITEM REWARDS ===

    // Health/armor pickup bonus
    if (cur_health > prev_health && cur_health > 0 && prev_health > 0) {
        reward += 0.2f;
    }
    if (cur_armor > prev_armor) {
        reward += 0.2f;
    }

    // Weapon pickup bonus
    for (i = 0; i < MAX_ITEMS; i++) {
        if (cl.inventory[i] > prev_inventory[i]) {
            reward += 0.3f;
            break;
        }
    }

    // === COMBAT ENGAGEMENT ===

    // Find enemy player
    bool enemy_visible = false;
    float enemy_dist = 99999.0f;
    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        for (i = 0; i < cl.frame.numEntities; i++) {
            int k = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
            const centity_state_t *es = &cl.entityStates[k];
            if (es->number >= 1 && es->number <= cl.maxclients &&
                es->number != cl.frame.clientNum + 1) {
                enemy_visible = true;
                float dx = es->s.origin[0] - cl.predicted_origin[0];
                float dy = es->s.origin[1] - cl.predicted_origin[1];
                float dz = es->s.origin[2] - cl.predicted_origin[2];
                enemy_dist = sqrtf(dx*dx + dy*dy + dz*dz);
                break;
            }
        }
    }

    // Shooting rewards/penalties
    if (cur_pm_type == PM_NORMAL && cur_health > 0 && q2ai_shared_action.attack) {
        if (!enemy_visible) {
            reward -= 0.15f;  // blind fire: very harsh penalty
        } else {
            reward += 0.02f;  // aimed fire: small reward
        }
    }

    // Blaster penalty: if holding blaster (item 7) but owns any better weapon, penalize
    // Item indices: 7=blaster, 8=SG, 9=SSG, 10=MG, 11=CG, 13=GL, 14=RL, 15=HB, 16=RG
    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        int selected = ps->stats[STAT_SELECTED_ITEM];
        if (selected == 7) {  // holding blaster
            // Check if we own anything better
            if (cl.inventory[8] || cl.inventory[9] || cl.inventory[10] ||
                cl.inventory[11] || cl.inventory[13] || cl.inventory[14] ||
                cl.inventory[15] || cl.inventory[16]) {
                reward -= 0.02f;  // switch to a real weapon!
            }
        }
    }

    // Enemy approach reward: when you see the enemy, go fight them
    if (enemy_visible && prev_enemy_dist < 90000.0f) {
        float approach = prev_enemy_dist - enemy_dist;
        // +0.01 per frame when approaching, -0.005 when retreating
        if (approach > 0) {
            reward += 0.01f * fminf(approach / 100.0f, 1.0f);
        } else {
            reward -= 0.005f;  // mild retreat penalty (sometimes retreat is smart)
        }
    }
    prev_enemy_dist = enemy_visible ? enemy_dist : 99999.0f;

    // === ITEM PROXIMITY (approach nearest visible item) ===

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
            reward += fminf(fmaxf(approach * 0.003f, -0.08f), 0.08f);
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
