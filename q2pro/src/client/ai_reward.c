/*
 * ai_reward.c -- Compute per-frame reward from game state deltas
 *
 * Tracks frame-over-frame changes in stats to generate reward signal.
 * Rewards:
 *   +1.0  per frag
 *   -1.0  per death
 *   +0.01 per frame alive (survival bonus)
 *   +0.005 movement reward (proportional to velocity — teaches "don't get stuck")
 *   -damage/100 for damage taken
 *   +0.05 per armor/health pickup
 *   +0.1  per item pickup
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
        memcpy(prev_inventory, cl.inventory, sizeof(prev_inventory));
        reward_initialized = true;
        return 0.0f;
    }

    int cur_frags = ps->stats[STAT_FRAGS];
    int cur_health = ps->stats[STAT_HEALTH];
    int cur_armor = ps->stats[STAT_ARMOR];
    int cur_pm_type = ps->pmove.pm_type;

    // Frag reward
    if (cur_frags > prev_frags) {
        reward += 1.0f * (cur_frags - prev_frags);
    }

    // Death penalty
    if ((cur_pm_type == PM_DEAD || cur_pm_type == PM_GIB) &&
        prev_pm_type != PM_DEAD && prev_pm_type != PM_GIB) {
        reward -= 1.0f;
    }

    // Damage taken penalty (proportional to health lost)
    if (cur_health < prev_health && prev_health > 0 && cur_pm_type == PM_NORMAL) {
        int damage = prev_health - cur_health;
        reward -= damage / 100.0f;
    }

    // Alive rewards
    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        // Survival bonus
        reward += 0.01f;

        // Movement reward: reward having velocity, penalize being stuck
        float vx = cl.predicted_velocity[0];
        float vy = cl.predicted_velocity[1];
        float speed = sqrtf(vx * vx + vy * vy);  // horizontal speed only
        if (speed > 20.0f) {
            // Moving: reward proportional to speed, up to +0.01/frame
            reward += 0.01f * fminf(speed / 400.0f, 1.0f);
        } else {
            // Stuck: penalty to encourage turning away from walls
            reward -= 0.02f;
        }
    }

    // Health/armor pickup bonus
    if (cur_health > prev_health && cur_health > 0 && prev_health > 0) {
        reward += 0.2f;
    }
    if (cur_armor > prev_armor) {
        reward += 0.2f;
    }

    // Weapon pickup bonus (any new weapon in inventory)
    for (i = 0; i < MAX_ITEMS; i++) {
        if (cl.inventory[i] > prev_inventory[i]) {
            reward += 0.3f;
            break;
        }
    }

    // Penalty for shooting without a visible enemy (wastes ammo, self-damage risk)
    if (cur_pm_type == PM_NORMAL && cur_health > 0 && q2ai_shared_action.attack) {
        bool enemy_visible = false;
        for (i = 0; i < cl.frame.numEntities; i++) {
            int k = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
            const centity_state_t *es = &cl.entityStates[k];
            if (es->number >= 1 && es->number <= cl.maxclients &&
                es->number != cl.frame.clientNum + 1) {
                enemy_visible = true;
                break;
            }
        }
        if (!enemy_visible) {
            reward -= 0.02f;  // penalty for blind firing
        }
    }

    // Item proximity reward: bonus for getting closer to nearest visible item
    if (cur_pm_type == PM_NORMAL && cur_health > 0) {
        float nearest_item = 99999.0f;
        int j;
        for (i = 0; i < cl.frame.numEntities; i++) {
            j = (cl.frame.firstEntity + i) & PARSE_ENTITIES_MASK;
            const centity_state_t *es = &cl.entityStates[j];
            if (es->number == cl.frame.clientNum + 1)
                continue;
            // Items have EF_ROTATE effect (spinning pickups)
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
            // +0.003 per unit approached, capped at ±0.08
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
