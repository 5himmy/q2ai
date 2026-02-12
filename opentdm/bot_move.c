/*
 * bot_move.c -- M_CheckGround and M_CatagorizePosition stubs for bot AI
 * Adapted from zigock-q2/src/g_monster.c for OpenTDM
 */
#include "bot.h"
#include "m_player.h"

#define STEPSIZE    18

/*
=============
M_CheckBottom

Returns false if any part of the bottom of the entity is off an edge that
is not a staircase.
=============
*/
qboolean M_CheckBottom(edict_t *ent)
{
    vec3_t  mins, maxs, start, stop;
    trace_t trace;
    int     x, y;
    float   mid, bottom;

    VectorAdd(ent->s.origin, ent->mins, mins);
    VectorAdd(ent->s.origin, ent->maxs, maxs);

    // if all of the points under the corners are solid world, don't bother
    // with the tougher checks
    // the corners must be within 16 of the midpoint
    start[2] = mins[2] - 1;
    for (x = 0; x <= 1; x++)
        for (y = 0; y <= 1; y++) {
            start[0] = x ? maxs[0] : mins[0];
            start[1] = y ? maxs[1] : mins[1];
            if (gi.pointcontents(start) != CONTENTS_SOLID) {
                goto realcheck;
            }
        }

    return true;        // we got out easy

realcheck:
    //
    // check it for real...
    //
    start[2] = mins[2];

    // the midpoint must be within 16 of the bottom
    start[0] = stop[0] = (mins[0] + maxs[0]) * 0.5;
    start[1] = stop[1] = (mins[1] + maxs[1]) * 0.5;
    stop[2] = start[2] - 2 * STEPSIZE;
    trace = gi.trace(start, vec3_origin, vec3_origin, stop, ent, MASK_PLAYERSOLID);

    if (trace.fraction == 1.0) {
        return false;
    }
    mid = bottom = trace.endpos[2];

    // the corners must be within 16 of the midpoint
    for (x = 0; x <= 1; x++)
        for (y = 0; y <= 1; y++) {
            start[0] = stop[0] = x ? maxs[0] : mins[0];
            start[1] = stop[1] = y ? maxs[1] : mins[1];

            trace = gi.trace(start, vec3_origin, vec3_origin, stop, ent, MASK_PLAYERSOLID);

            if (trace.fraction != 1.0 && trace.endpos[2] > bottom) {
                bottom = trace.endpos[2];
            }
            if (trace.fraction == 1.0 || mid - trace.endpos[2] > STEPSIZE) {
                return false;
            }
        }

    return true;
}

void M_CheckGround(edict_t *ent)
{
    vec3_t      point, stp, v1, v2;
    trace_t     trace, tracep;

    if (ent->flags & (FL_SWIM | FL_FLY)) {
        return;
    }

    if (ent->client) {
        ent->client->zc.ground_slope = 1.0;
    }

    if (ent->velocity[2] > 100) {
        ent->groundentity = NULL;
        return;
    }

    // if the hull point one-quarter unit down is solid the entity is on ground
    point[0] = ent->s.origin[0];
    point[1] = ent->s.origin[1];
    point[2] = ent->s.origin[2] - GROUND_TEST_EPSILON;

    trace = gi.trace(ent->s.origin, ent->mins, ent->maxs, point, ent, MASK_BOTGROUND);

    // check steepness
    if (trace.fraction == 1.0
        && (!trace.startsolid && !trace.allsolid)) {
        ent->groundentity = NULL;
        return;
    }

    if (trace.allsolid) {
        VectorSet(v1, -16, -16, -24);
        VectorSet(v2, 16, 16, 4);

        VectorCopy(ent->s.origin, stp);
        stp[2] += 24;
        tracep = gi.trace(stp, v1, v2, point, ent, MASK_BOTGROUND);
        if (tracep.ent && !tracep.allsolid) {
            if (tracep.ent->classname[0] == 'f') {
                VectorCopy(tracep.endpos, ent->s.origin);
                ent->groundentity = tracep.ent;
                ent->groundentity_linkcount = tracep.ent->linkcount;
                gi.linkentity(ent);
                return;
            }
        }
    }

    if (!trace.allsolid) {
        if (ent->client) {
            ent->client->zc.ground_contents = trace.contents;
            ent->client->zc.ground_slope = trace.plane.normal[2];
        }
        VectorCopy(trace.endpos, ent->s.origin);
        ent->groundentity = trace.ent;
        ent->groundentity_linkcount = trace.ent->linkcount;
    }
    gi.linkentity(ent);
}


void M_CatagorizePosition(edict_t *ent)
{
    vec3_t      point;
    int         cont;

    // get waterlevel
    point[0] = ent->s.origin[0];
    point[1] = ent->s.origin[1];
    point[2] = ent->s.origin[2] + ent->mins[2] + 1;
    cont = gi.pointcontents(point);

    if (!(cont & MASK_WATER)) {
        ent->waterlevel = 0;
        ent->watertype = 0;
        return;
    }

    ent->watertype = cont;
    ent->waterlevel = 1;
    point[2] += 26;
    cont = gi.pointcontents(point);
    if (!(cont & MASK_WATER)) {
        return;
    }

    ent->waterlevel = 2;
    point[2] += 22;
    cont = gi.pointcontents(point);
    if (cont & MASK_WATER) {
        ent->waterlevel = 3;
    }
}
