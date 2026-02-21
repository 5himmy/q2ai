/*
 * ai_observe.c -- Extract game state into observation struct
 *
 * Reads from cl.frame.ps (player state), cl_entities[] (visible entities),
 * and cl.inventory[] each frame.
 */

#include "client.h"
#include "ai_bridge.h"

static uint8_t AI_ClassifyEntity(int ent_number, int effects)
{
    // Players: entity numbers 1..maxclients
    if (ent_number >= 1 && ent_number <= cl.maxclients)
        return ENT_TYPE_PLAYER;

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
        eo->entity_type = AI_ClassifyEntity(es->number, es->s.effects);

        // Track enemy player (first non-self player entity)
        if (eo->entity_type == ENT_TYPE_PLAYER && !obs->enemy_visible) {
            VectorCopy(es->s.origin, obs->enemy_origin);
            VectorCopy(es->s.angles, obs->enemy_angles);
            obs->enemy_visible = 1;
            obs->enemy_client_num = es->number - 1;
        }

        count++;
    }

    obs->num_entities = count;

    // Combat feedback: damage dealt this frame (from STAT_FLASHES)
    obs->damage_dealt = ps->stats[STAT_FLASHES];

    // Game info
    Q_strlcpy(obs->mapname, cl.mapname, sizeof(obs->mapname));
    obs->serverframe = cl.frame.number;
    obs->servertime = cl.servertime;
    obs->client_num = cl.clientNum;
}
