/*
 * botstr.h -- Zigock bot client state structure
 * Adapted for OpenTDM from 3ZB2 (zigock-q2)
 */
#ifndef BOTSTRUCT
#define BOTSTRUCT

// Zigock client info
#define ALEAT_MAX   10

typedef struct zgcl_s {
    int         zclass;         // class no.
    int         botindex;       // botlist's index NO.

    // true client zoom flags
    int         aiming;         // 0-not 1-aiming 2-firing zoomingflag
    float       distance;       // zoom FOV value
    float       olddistance;   // previous zoom FOV value
    qboolean    autozoom;       // autozoom
    qboolean    lockon;         // lockon flag false-not true-locking

    // bot state
    int         zcstate;        // status
    int         zccmbstt;       // combat status

    // duck
    float       n_duckedtime;   // non ducked time

    // targets
    edict_t     *first_target;  // enemy
    float       targetlock;     // target locking time
    short       firstinterval;  // enemy search count
    edict_t     *second_target; // items
    short       secondinterval; // item pickup call count

    // waiting
    vec3_t      movtarget_pt;   // moving target waiting point
    edict_t     *waitin_obj;    // for waiting sequence complete

    // basic moving
    float       moveyaw;        // true moving yaw

    // combat
    int         total_bomb;     // total put bomb
    float       gren_time;      // grenade time

    // contents
    int         ground_contents;
    float       ground_slope;

    // count (inc only)
    int         tmpcount;

    // moving hist
    float       nextcheck;      // checking time
    vec3_t      pold_origin;    // old origin
    vec3_t      pold_angles;    // old angles
    float       push_time;      // level.time when trigger_push affected bot
    vec3_t      push_norm;      // velocity normal imparted by last trigger_push

    // target object shot
    qboolean    objshot;

    edict_t     *sighten;       // sighting enemy to me info from entity sight
    edict_t     *locked;        // locking enemy to me info from lockon missile

    // waterstate
    int         waterstate;

    // route
    qboolean    route_trace;
    int         routeindex;     // routing index
    float       rt_locktime;
    float       rt_releasetime;
    qboolean    havetarget;     // target on/off
    int         targetindex;

    // battle
    edict_t     *last_target;   // old enemy
    vec3_t      last_pos;       // old origin
    int         battlemode;     // mode
    int         battlecount;    // temporary count
    int         battlesubcnt;   // subcount
    int         battleduckcnt;  // duck
    float       fbattlecount;   // float temporary count
    vec3_t      vtemp;          // temporary vec
    int         foundedenemy;   // foundedenemy
    char        secwep_selected; // secondweapon selected

    vec3_t      aimedpos;       // shot point
    qboolean    trapped;        // trapflag

    // team
    short       tmplstate;      // teamplay state
    short       ctfstate;       // ctf state (kept for struct compat)
    edict_t     *followmate;    // follow
    float       matelock;       // team mate locking time

    // stuck detection (10-second global check)
    vec3_t      stuck_origin;       // position at last stuck check
    int         stuck_check_frame;  // next frame to check
    int         stuck_count;        // consecutive stuck checks failed
} zgcl_t;

#endif
