/*
 * bot.c -- Bot trace utilities and angle calculations
 * Adapted for OpenTDM from 3ZB2 (zigock-q2)
 */
#include "bot.h"
#include "m_player.h"

void droptofloor(edict_t *ent);

void SpawnItem2(edict_t *ent, gitem_t *item)
{
    ent->item = item;
    ent->nextthink = level.framenum;
    ent->think = droptofloor;
    ent->s.effects = 0;
    ent->s.renderfx = RF_GLOW;
    droptofloor(ent);
    ent->s.modelindex = 0;
    ent->nextthink = level.framenum + SECS_TO_FRAMES(10);
    ent->think = G_FreeEdict;
}

void SpawnItem3(edict_t *ent, gitem_t *item)
{
    ent->item = item;
    ent->nextthink = level.framenum + 2;
    ent->think = droptofloor;
    ent->s.effects = 0;
    ent->s.renderfx = 0;
    ent->s.modelindex = 0;
}

//=====================================

qboolean TraceX(edict_t *ent, vec3_t p2)
{
    trace_t     rs_trace;
    vec3_t      v1, v2;
    int         contents;

    contents = CONTENTS_SOLID | CONTENTS_WINDOW;

    if (!(ent->svflags & SVF_MONSTER)) {
        if (ent->client->zc.waterstate) {
            VectorCopy(ent->mins, v1);
            VectorCopy(ent->maxs, v2);
        }
        else if (!(ent->client->ps.pmove.pm_flags & PMF_DUCKED)) {
            VectorSet(v1, -16, -16, -4);
            VectorSet(v2, 16, 16, 32);
        }
        else {
            VectorSet(v1, -4, -4, -4);
            VectorSet(v2, 4, 4, 4);
        }
    }
    else {
        VectorSet(v1, 0, 0, 0);
        VectorSet(v2, 0, 0, 0);
        contents |= CONTENTS_LAVA | CONTENTS_SLIME;
    }

    rs_trace = gi.trace(ent->s.origin, v1, v2, p2, ent, contents);
    if (rs_trace.fraction == 1.0 && !rs_trace.allsolid && !rs_trace.startsolid) {
        return true;
    }

    if (ent->client->zc.route_trace && rs_trace.ent && (ent->svflags & SVF_MONSTER)) {
        if (!Q_stricmp(rs_trace.ent->classname, "func_door")) {
            if (rs_trace.ent->moveinfo.state == PSTATE_UP) {
                return true;
            }
            else {
                return false;
            }
        }
    }

    return false;
}

//=====================================

qboolean Bot_trace(edict_t *ent, edict_t *other)
{
    trace_t     rs_trace;
    vec3_t  ttx;
    vec3_t  tty;

    VectorCopy(ent->s.origin, ttx);
    VectorCopy(other->s.origin, tty);
    if (ent->maxs[2] >= 32) {
        if (tty[2] > ttx[2]) {
            tty[2] += 16;
        }
        ttx[2] += 30;
    }
    else {
        ttx[2] -= 12;
    }

    rs_trace = gi.trace(ttx, NULL, NULL, tty, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);
    if (rs_trace.fraction == 1.0 && !rs_trace.allsolid && !rs_trace.startsolid) {
        return true;
    }
    if (ent->maxs[2] < 32) {
        return false;
    }

    if (other->classname[6] == 'F'
        || other->classname[0] == 'w')
    {}
    else if (other->classname[0] == 'i') {
        if (other->classname[5] == 'q'
            || other->classname[5] == 'f'
            || other->classname[5] == 't'
            || other->classname[5] == 'i'
            || other->classname[5] == 'h'
            || other->classname[5] == 'a') {}
        else {
            return false;
        }
    }
    else {
        return false;
    }

    if (rs_trace.ent != NULL) {
        if (rs_trace.ent->classname[0] == 'f'
            && rs_trace.ent->classname[5] == 'd'
            && rs_trace.ent->targetname == NULL) {
            return true;
        }
    }

    if (ent->s.origin[2] < other->s.origin[2]
        || ent->s.origin[2] - 24 > other->s.origin[2]) {
        return false;
    }

    ttx[2] -= 36;
    rs_trace = gi.trace(ttx, NULL, NULL, other->s.origin, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);
    if (rs_trace.fraction == 1.0 && !rs_trace.allsolid && !rs_trace.startsolid) {
        return true;
    }
    return false;
}


qboolean Bot_traceX(edict_t *ent, edict_t *other)
{
    trace_t     rs_trace;
    vec3_t  ttx, tty;
    VectorCopy(ent->s.origin, ttx);
    VectorCopy(other->s.origin, tty);
    ttx[2] += 16;
    tty[2] += 16;

    rs_trace = gi.trace(ttx, NULL, NULL, tty, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);
    if (rs_trace.fraction == 1.0) {
        return true;
    }
    return false;
}

qboolean Bot_traceY(edict_t *ent, edict_t *other)
{
    trace_t     rs_trace;
    vec3_t  ttx, tty;
    VectorCopy(ent->s.origin, ttx);
    VectorCopy(other->s.origin, tty);
    if (ent->maxs[2] >= 32) {
        ttx[2] += 24;
    }
    else {
        ttx[2] -= 12;
    }

    tty[2] += 16;

    rs_trace = gi.trace(ttx, NULL, NULL, tty, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);
    if (rs_trace.fraction == 1.0) {
        return true;
    }
    return false;
}

qboolean Bot_trace2(edict_t *ent, vec3_t ttz)
{
    trace_t     rs_trace;
    vec3_t  ttx;
    VectorCopy(ent->s.origin, ttx);
    if (ent->maxs[2] >= 32) {
        ttx[2] += 24;
    }
    else {
        ttx[2] -= 12;
    }

    rs_trace = gi.trace(ttx, NULL, NULL, ttz, ent, CONTENTS_SOLID | CONTENTS_LAVA | CONTENTS_SLIME);
    if (rs_trace.fraction != 1.0) {
        return false;
    }
    return true;
}

qboolean Bot_traceS(edict_t *ent, edict_t *other)
{
    trace_t     rs_trace;
    vec3_t  start, end;
    int     mycont;

    VectorCopy(ent->s.origin, start);
    VectorCopy(other->s.origin, end);

    start[2] += ent->viewheight - 8;
    end[2] += other->viewheight - 8;

    if (Bot[ent->client->zc.botindex].param[BOP_NOSTHRWATER]) {
        goto WATERMODE;
    }

    rs_trace = gi.trace(start, NULL, NULL, end, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);

    if (rs_trace.fraction != 1.0) {
        return false;
    }
    return true;

WATERMODE:
    mycont = gi.pointcontents(start);

    if ((mycont & CONTENTS_WATER) && !other->waterlevel) {
        rs_trace = gi.trace(end, NULL, NULL, start, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME | CONTENTS_WATER);
        if (rs_trace.surface) {
            if (rs_trace.surface->flags & SURF_WARP) {
                return false;
            }
        }
        rs_trace = gi.trace(start, NULL, NULL, end, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);
        if (rs_trace.fraction != 1.0) {
            return false;
        }
        return true;
    }
    else if ((mycont & CONTENTS_WATER) && other->waterlevel) {
        VectorCopy(other->s.origin, end);
        end[2] -= 16;
        rs_trace = gi.trace(start, NULL, NULL, end, ent, CONTENTS_SOLID | CONTENTS_WINDOW);
        if (rs_trace.fraction != 1.0) {
            return false;
        }
        return true;
    }

    if (other->waterlevel) {
        VectorCopy(other->s.origin, end);
        end[2] += 32;
        rs_trace = gi.trace(start, NULL, NULL, end, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_WATER);
        if (rs_trace.surface) {
            if (rs_trace.surface->flags & SURF_WARP) {
                return false;
            }
        }
    }

    rs_trace = gi.trace(start, NULL, NULL, end, ent, CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_LAVA | CONTENTS_SLIME);
    if (rs_trace.fraction != 1.0) {
        return false;
    }
    return true;
}


float Get_yaw(vec3_t vec)
{
    vec3_t      out;
    double      yaw;

    VectorCopy(vec, out);
    out[2] = 0;
    VectorNormalize2(out, out);

    yaw = acos((double) out[0]);
    yaw = yaw / M_PI * 180;

    if (asin((double) out[1]) < 0) {
        yaw *= -1;
    }

    return (float)yaw;
}

float Get_pitch(vec3_t vec)
{
    vec3_t      out;
    float       pitch;

    VectorNormalize2(vec, out);

    pitch = acos((double) out[2]);
    pitch = ((float)pitch) / M_PI * 180;

    pitch -= 90;
    if (pitch < -180) {
        pitch += 360;
    }

    return pitch;
}

float Get_vec_yaw(vec3_t vec, float yaw)
{
    float       vecsyaw;

    vecsyaw = Get_yaw(vec);

    if (vecsyaw > yaw) {
        vecsyaw -= yaw;
    }
    else {
        vecsyaw = yaw - vecsyaw;
    }

    if (vecsyaw > 180) {
        vecsyaw = 360 - vecsyaw;
    }

    return vecsyaw;
}

float Get_vec_yaw2(vec3_t vec, float yaw)
{
    float       vecsyaw;

    vecsyaw = Get_yaw(vec);

    vecsyaw -= yaw;
    if (vecsyaw > 180) {
        vecsyaw -= 360;
    }
    else if (vecsyaw < -180) {
        vecsyaw += 360;
    }

    return vecsyaw;
}
