/*
 * ai_reward.c -- Compute per-frame reward from game state deltas
 *
 * V3.0 Stage 1: Hit-only rewards + auxiliary aim supervision.
 * Aim direction learned via MSE loss in Python (train.py).
 * RL reward only for actual hits — forces NN to tighten aim std.
 *
 *   +damage/50        damage dealt (actual hit confirmation)
 *   +1.0              per frag
 *   -1.0              per death
 *   Aim direction learned via auxiliary supervision loss in Python.
 *   RL reward only for actual hits — forces NN to tighten aim std.
 */

#include "client.h"
#include "ai_bridge.h"

static int prev_frags;
static int prev_health;
static int prev_pm_type;
static bool reward_initialized;

float AI_Reward_Compute(void)
{
    const player_state_t *ps = &cl.frame.ps;
    float reward = 0.0f;

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

    // Damage dealt reward — only actual hits count
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
