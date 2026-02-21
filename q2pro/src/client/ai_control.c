/*
 * ai_control.c -- Episode lifecycle management for AI training
 *
 * Handles auto-connect, team assignment, ready-up, and respawn
 * so the AI can run unsupervised episode loops.
 */

#include "client.h"
#include "ai_bridge.h"

static cvar_t *q2ai_autoconnect;
static cvar_t *q2ai_server;
static cvar_t *q2ai_autojoin;
static cvar_t *q2ai_team;          // "a", "b", or "auto"
static cvar_t *q2ai_name;
static cvar_t *q2ai_demo;
static bool control_initialized;

// Throttle command sending to avoid spamming
static unsigned join_time;
static unsigned ready_time;
static unsigned respawn_time;
static bool joined;
static bool readied;
static int join_attempts;

#define CONTROL_DELAY   3000    // ms between control commands

void AI_Control_Init(void)
{
    q2ai_autoconnect = Cvar_Get("q2ai_autoconnect", "1", 0);
    q2ai_server = Cvar_Get("q2ai_server", "127.0.0.1:27910", 0);
    q2ai_autojoin = Cvar_Get("q2ai_autojoin", "1", 0);
    q2ai_team = Cvar_Get("q2ai_team", "auto", 0);
    q2ai_name = Cvar_Get("q2ai_name", "KaaN[V1.0]", 0);
    q2ai_demo = Cvar_Get("q2ai_demo", "0", 0);
    control_initialized = true;
    joined = false;
    readied = false;
    join_time = 0;
    ready_time = 0;
    respawn_time = 0;
    join_attempts = 0;

    // Initialize the shared memory bridge
    AI_Bridge_Init();

    Com_Printf("Q2AI: Control initialized (server=%s, team=%s)\n",
               q2ai_server->string, q2ai_team->string);
}

void AI_Control_Frame(void)
{
    unsigned now;

    if (!control_initialized)
        return;

    // In demo mode, skip all auto-connect/join/ready
    if (q2ai_demo && q2ai_demo->integer)
        return;

    now = cls.realtime;

    // Auto-connect if disconnected
    if (q2ai_autoconnect->integer && cls.state == ca_disconnected) {
        joined = false;
        join_attempts = 0;
        Cbuf_AddText(&cmd_buffer, va("connect %s\n", q2ai_server->string));
        return;
    }

    if (!q2ai_autojoin->integer || cls.state != ca_active)
        return;

    const player_state_t *ps = &cl.frame.ps;
    int pm_type = ps->pmove.pm_type;

    // If we have health > 0, we're playing
    if (pm_type == PM_NORMAL && ps->stats[STAT_HEALTH] > 0) {
        if (!joined) {
            joined = true;
            Cbuf_AddText(&cmd_buffer, va("set name \"%s\"\n", q2ai_name->string));
            Com_Printf("Q2AI: Spawned into game (health=%d, name=%s)\n",
                       ps->stats[STAT_HEALTH], q2ai_name->string);
        }
        // Auto-ready: keep sending until match starts (every 2s)
        if (!readied && now - ready_time > 2000) {
            Cbuf_AddText(&cmd_buffer, "cmd ready\n");
            ready_time = now;
            Com_Printf("Q2AI: Sent ready command\n");
        }
        return;
    }

    // In spectator mode — try to join
    if (!joined && (pm_type == PM_SPECTATOR || pm_type == PM_FREEZE)) {
        if (now - join_time > CONTROL_DELAY) {
            const char *team = q2ai_team->string;

            if (!strcmp(team, "auto")) {
                // Alternate between teams on retries
                team = (join_attempts % 2 == 0) ? "a" : "b";
            }

            Cbuf_AddText(&cmd_buffer, va("cmd team %s\n", team));
            join_time = now;
            join_attempts++;
            Com_Printf("Q2AI: Joining team %s (attempt %d)\n",
                       team, join_attempts);
        }
        return;
    }

    // Dead — auto-respawn by pressing attack
    if (pm_type == PM_DEAD || pm_type == PM_GIB) {
        if (now - respawn_time > 1000) {
            Cbuf_AddText(&cmd_buffer, "cmd putaway\n");
            respawn_time = now;
        }
    }
}
