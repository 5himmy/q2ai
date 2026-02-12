/*
 * bot_func.c -- Bot initialization, spawning, thinking
 * Adapted for OpenTDM from 3ZB2 (zigock-q2)
 */
#include "bot.h"
#include "g_tdm_core.h"
#include "m_player.h"

//----------------------------------------------------------------
// Load Chain Data - port of CTFSetupNavSpawn() from zigock-q2
// Loads pre-computed waypoint routes from .chn files for bot navigation
//----------------------------------------------------------------
void Bot_LoadChainData(void)
{
    FILE        *fpout;
    char        name[256];
    char        code[8];
    int         i, j;
    vec3_t      v;
    edict_t     *other;
    unsigned int size;

    CurrentIndex = 0;
    memset(Route, 0, sizeof(Route));
    memset(code, 0, sizeof(code));

    // Build path: ./<gamedir>/chdtm/<mapname>.chn
    snprintf(name, sizeof(name), "./%s/chdtm/%s.chn", game.gamedir, level.mapname);

    fpout = fopen(name, "rb");
    if (fpout == NULL) {
        gi.dprintf("Chaining: file %s.chn not found.\n", level.mapname);
        return;
    }

    // Verify file signature (8-byte magic: "3ZBRGDTM" for deathmatch)
    if (fread(code, sizeof(char), 8, fpout) != 8) {
        gi.dprintf("Chaining: %s.chn read error.\n", level.mapname);
        fclose(fpout);
        return;
    }
    if (strncmp(code, "3ZBRGDTM", 8)) {
        CurrentIndex = 0;
        gi.dprintf("Chaining: %s.chn is not a valid chain file.\n", level.mapname);
        fclose(fpout);
        return;
    }

    // Read route count and route data
    if (fread(&CurrentIndex, sizeof(int), 1, fpout) != 1) {
        CurrentIndex = 0;
        gi.dprintf("Chaining: %s.chn read error (index).\n", level.mapname);
        fclose(fpout);
        return;
    }
    if (CurrentIndex <= 0 || CurrentIndex > MAXNODES) {
        gi.dprintf("Chaining: %s.chn invalid node count %i.\n", level.mapname, CurrentIndex);
        CurrentIndex = 0;
        fclose(fpout);
        return;
    }

    // Chain files were written with 32-bit route_t (32 bytes per record).
    // On 64-bit, route_t is 40 bytes (8-byte pointer + padding), so we
    // must read into a packed 32-byte struct and convert.
    {
        typedef struct {
            vec3_t  Pt;                             // 12 bytes
            union {
                vec3_t          Tcourner;            // 12 bytes
                unsigned short  linkpod[MAXLINKPOD]; // 12 bytes
            };
            uint32_t ent_pad;                        // 4 bytes (32-bit pointer)
            short   index;                           // 2 bytes
            short   state;                           // 2 bytes
        } route_file_t;                              // = 32 bytes total

        route_file_t *filebuf;

        size = (unsigned int)CurrentIndex * (unsigned int)sizeof(route_file_t);
        filebuf = malloc(size);
        if (!filebuf) {
            CurrentIndex = 0;
            gi.dprintf("Chaining: out of memory for %i routes.\n", CurrentIndex);
            fclose(fpout);
            return;
        }
        if (fread(filebuf, size, 1, fpout) != 1) {
            CurrentIndex = 0;
            gi.dprintf("Chaining: %s.chn read error (routes).\n", level.mapname);
            free(filebuf);
            fclose(fpout);
            return;
        }

        for (i = 0; i < CurrentIndex; i++) {
            VectorCopy(filebuf[i].Pt, Route[i].Pt);
            VectorCopy(filebuf[i].Tcourner, Route[i].Tcourner);
            Route[i].ent   = NULL;
            Route[i].index = filebuf[i].index;
            Route[i].state = filebuf[i].state;
        }

        free(filebuf);
    }

    // Link routes to in-game entities (platforms, doors, items, etc.)
    for (i = 0; i < CurrentIndex; i++) {
        Route[i].ent = NULL;  // clear stale pointers from file data

        if ((Route[i].state > GRS_TELEPORT && Route[i].state <= GRS_PUSHBUTTON)
            || Route[i].state == GRS_REDFLAG || Route[i].state == GRS_BLUEFLAG) {

            other = &g_edicts[game.maxclients + 1];
            for (j = game.maxclients + 1; j < globals.num_edicts; j++, other++) {
                if (!other->inuse)
                    continue;

                if (Route[i].state == GRS_ONPLAT
                    || Route[i].state == GRS_ONTRAIN
                    || Route[i].state == GRS_PUSHBUTTON
                    || Route[i].state == GRS_ONDOOR) {
                    VectorAdd(other->s.origin, other->mins, v);
                    if (VectorCompare(Route[i].Pt, v)) {
                        if (Route[i].state == GRS_ONPLAT
                            && !Q_stricmp(other->classname, "func_plat")) {
                            Route[i].ent = other;
                            break;
                        } else if (Route[i].state == GRS_ONTRAIN
                                   && !Q_stricmp(other->classname, "func_train")) {
                            Route[i].ent = other;
                            break;
                        } else if (Route[i].state == GRS_PUSHBUTTON
                                   && !Q_stricmp(other->classname, "func_button")) {
                            Route[i].ent = other;
                            break;
                        } else if (Route[i].state == GRS_ONDOOR
                                   && !Q_stricmp(other->classname, "func_door")) {
                            Route[i].ent = other;
                            break;
                        }
                    }
                } else if (Route[i].state == GRS_ITEMS
                           || Route[i].state == GRS_REDFLAG
                           || Route[i].state == GRS_BLUEFLAG) {
                    if (VectorCompare(Route[i].Pt, other->s.origin)) {
                        Route[i].ent = other;
                        break;
                    }
                }
            }

            // If entity not found, demote to normal route node
            if (j >= globals.num_edicts) {
                Route[i].state = GRS_NORMAL;
            }
        } else {
            Route[i].ent = NULL;
        }
    }

    gi.dprintf("Chaining: Total %i chaining pods assigned.\n", CurrentIndex);
    fclose(fpout);
}

//----------------------------------------------------------------
// Load Bot Info - simplified for opentdm (no config file)
//----------------------------------------------------------------
void Bot_InitDefaults(void)
{
    int i, j;

    SpawnWaitingBots = 0;
    ListedBotCount = 0;
    ListedBots = 0;

    memset(ClientMessage, 0, sizeof(ClientMessage));

    j = 1;
    for (i = 0; i < MAXBOTS; i++) {
        // netname
        Com_sprintf(Bot[i].netname, sizeof(Bot[i].netname), "Bot[%d]", i);
        // model
        strcpy(Bot[i].model, "male");
        // skin
        strcpy(Bot[i].skin, "grunt");

        // params - medium skill defaults
        Bot[i].param[BOP_WALK] = 0;
        Bot[i].param[BOP_AIM] = 5;
        Bot[i].param[BOP_PICKUP] = 5;
        Bot[i].param[BOP_OFFENCE] = 5;
        Bot[i].param[BOP_COMBATSKILL] = 5;
        Bot[i].param[BOP_ROCJ] = 0;
        Bot[i].param[BOP_REACTION] = 3;
        Bot[i].param[BOP_VRANGE] = 90;
        Bot[i].param[BOP_HRANGE] = 180;
        Bot[i].param[BOP_PRIWEP] = WEAP_ROCKETLAUNCHER;
        Bot[i].param[BOP_SECWEP] = WEAP_RAILGUN;
        Bot[i].param[BOP_DODGE] = 5;
        Bot[i].param[BOP_ESTIMATE] = 3;
        Bot[i].param[BOP_NOISECHK] = 3;
        Bot[i].param[BOP_NOSTHRWATER] = 0;
        Bot[i].param[BOP_TEAMWORK] = 5;

        // spawn flag
        Bot[i].spflg = 0;
        // team
        Bot[i].team = j;
        if (++j > 2) {
            j = 1;
        }
    }

    // Pre-cache item pointers
    Fdi_BLASTER = (gitem_t *)FindItem("Blaster");
    Fdi_SHOTGUN = (gitem_t *)FindItem("Shotgun");
    Fdi_SUPERSHOTGUN = (gitem_t *)FindItem("Super Shotgun");
    Fdi_MACHINEGUN = (gitem_t *)FindItem("Machinegun");
    Fdi_CHAINGUN = (gitem_t *)FindItem("Chaingun");
    Fdi_GRENADES = (gitem_t *)FindItem("Grenades");
    Fdi_GRENADELAUNCHER = (gitem_t *)FindItem("Grenade Launcher");
    Fdi_ROCKETLAUNCHER = (gitem_t *)FindItem("Rocket Launcher");
    Fdi_HYPERBLASTER = (gitem_t *)FindItem("HyperBlaster");
    Fdi_RAILGUN = (gitem_t *)FindItem("Railgun");
    Fdi_BFG = (gitem_t *)FindItem("BFG10K");

    Fdi_SHELLS = (gitem_t *)FindItem("Shells");
    Fdi_BULLETS = (gitem_t *)FindItem("Bullets");
    Fdi_CELLS = (gitem_t *)FindItem("Cells");
    Fdi_ROCKETS = (gitem_t *)FindItem("Rockets");
    Fdi_SLUGS = (gitem_t *)FindItem("Slugs");

    skullindex = gi.modelindex("models/objects/gibs/skull/tris.md2");
    headindex = gi.modelindex("models/objects/gibs/head2/tris.md2");

    gi.dprintf("Bot defaults initialized.\n");
}

//----------------------------------------------------------------
// Get Number of Client (including bots)
//----------------------------------------------------------------
int Get_NumOfPlayer(void)
{
    int i, j;
    edict_t *ent;

    j = 0;
    for (i = 0; i < game.maxclients; i++) {
        ent = g_edicts + 1 + i;
        if (ent->inuse) {
            j++;
        }
    }
    return j;
}

//----------------------------------------------------------------
// Get New Client - find a free client slot for the bot
//----------------------------------------------------------------
edict_t *Get_NewClient(void)
{
    int         i;
    edict_t     *e;

    // Search from the end to avoid conflicting with real players
    for (i = game.maxclients; i >= 1; i--) {
        e = &g_edicts[i];
        if (!e->inuse) {
            // Check freed_framenum (opentdm uses unsigned framenum)
            if (e->freed_framenum < 2 || level.framenum - e->freed_framenum > SECS_TO_FRAMES(0.5)) {
                G_InitEdict(e);
                return e;
            }
        }
    }

    return NULL;
}

//----------------------------------------------------------------
// Bot Think
//----------------------------------------------------------------
void Bot_Think(edict_t *self)
{
    if (self->linkcount != self->monsterinfo.linkcount) {
        M_CheckGround(self);
    }

    if (self->deadflag) {
        if (self->s.modelindex == skullindex || self->s.modelindex == headindex) {
            self->s.frame = 0;
        }
        else if (self->s.frame < FRAME_crdeath1 && self->s.frame != 0) {
            self->s.frame = FRAME_death308;
        }
        self->s.modelindex2 = 0;    // remove linked weapon model
        self->s.modelindex3 = 0;

        self->client->zc.route_trace = false;
        if (self->client->respawn_framenum <= level.framenum) {
            if (self->svflags & SVF_MONSTER) {
                self->client->respawn_framenum = level.framenum;
                CopyToBodyQue(self);
                PutBotInServer(self);
            }
        }
    }
    else {
        Bots_Move_NORM(self);
        if (!self->inuse) {
            return;    // removed bot self
        }

        ClientBeginServerFrame(self);
    }

    if (self->linkcount != self->monsterinfo.linkcount) {
        M_CheckGround(self);
    }
    M_CatagorizePosition(self);
    BotEndServerFrame(self);

    self->nextthink = level.framenum + 1;
    return;
}

//----------------------------------------------------------------
// Initialize Bot
//----------------------------------------------------------------
void InitializeBot(edict_t *ent, int botindex)
{
    gclient_t   *client;
    char        pinfo[200];
    int         index;

    index = ent - g_edicts - 1;
    ent->client = &game.clients[index];

    client = ent->client;

    memset(&client->zc, 0, sizeof(zgcl_t));
    memset(&client->pers, 0, sizeof(client->pers));
    memset(&client->resp, 0, sizeof(client->resp));

    // set botindex NO.
    client->zc.botindex = botindex;

    client->resp.enterframe = level.framenum;

    // set netname model skin
    Com_sprintf(pinfo, sizeof(pinfo),
        "\\rate\\25000\\msg\\1\\fov\\90\\skin\\%s/%s\\name\\%s\\hand\\0",
        Bot[botindex].model, Bot[botindex].skin, Bot[botindex].netname);

    ent->svflags = SVF_MONSTER;  // CAUTION: ClientUserinfoChanged tests this flag
    ClientUserinfoChanged(ent, pinfo);

    ent->health                 = 100;
    ent->max_health             = 100;

    // opentdm stores ammo caps in gclient_t directly
    client->max_bullets    = 200;
    client->max_shells     = 100;
    client->max_rockets    = 50;
    client->max_grenades   = 50;
    client->max_cells      = 200;
    client->max_slugs      = 50;

    // Mark as not a real connected client (bots have no network connection)
    client->pers.connected = false;

    // Assign team based on Bot[].team (alternates A/B in Bot_InitDefaults)
    if (Bot[botindex].team == 1) {
        client->pers.team = TEAM_A;
    } else {
        client->pers.team = TEAM_B;
    }

    gi.dprintf("%s connected\n", ent->client->pers.netname);
    gi.bprintf(PRINT_HIGH, "%s entered the game\n", client->pers.netname);
}


static qboolean moveEntUntilNotStartsolid(edict_t *ent, int contentmask, float step_x, float step_y, float step_z, int max_steps)
{
    trace_t trace;

    while (max_steps-- > 0) {
        trace = gi.trace(ent->s.origin, ent->mins, ent->maxs, ent->s.origin, ent, contentmask);
        if (!trace.startsolid) {
            return true;
        }

        ent->s.origin[0] += step_x;
        ent->s.origin[1] += step_y;
        ent->s.origin[2] += step_z;
    }

    return false;
}

//----------------------------------------------------------------
// Put Bot In Server
//----------------------------------------------------------------
void PutBotInServer(edict_t *ent)
{
    int         i, j;
    gitem_t     *item;
    gclient_t   *client;
    vec3_t  spawn_origin, spawn_angles;
    zgcl_t      *zc;

    zc = &ent->client->zc;

    // current weapon
    client = ent->client;
    item = Fdi_BLASTER;
    client->selected_item = ITEM_INDEX(item);
    client->inventory[client->selected_item] = 1;
    client->weapon = item;
    client->silencer_shots = 0;
    client->weaponstate = WEAPON_READY;
    client->newweapon = NULL;

    // clear powerups (opentdm uses framenum)
    client->quad_framenum = 0;
    client->invincible_framenum = 0;
    client->breather_framenum = 0;
    client->enviro_framenum = 0;
    client->grenade_state = GRENADE_NONE;
    client->grenade_framenum = 0;

    j = zc->botindex;
    i = zc->routeindex;
    memset(&client->zc, 0, sizeof(zgcl_t));
    zc->botindex = j;
    zc->routeindex = i;

    // clear entity values
    ent->classname = "player";
    ent->movetype = MOVETYPE_STEP;
    ent->solid = SOLID_BBOX;
    ent->model = "players/male/tris.md2";
    VectorSet(ent->mins, -16, -16, -24);
    VectorSet(ent->maxs, 16, 16, 32);

    ent->health = 100;
    ent->max_health = 100;
    ent->gib_health = -40;
    ent->viewheight = 22;

    ent->gravity = 1.0;
    ent->mass = 200;
    ent->target_ent = NULL;
    ent->s.frame = 0;

    // clear entity state values
    ent->s.modelindex = 255;        // will use the skin specified model
    ent->s.skinnum = ent - g_edicts - 1;
    // ShowGun equivalent - set weapon model (255 = skin-specified model)
    ent->s.modelindex2 = 255;

    ent->s.sound = 0;

    ent->monsterinfo.scale = MODEL_SCALE;

    ent->pain = player_pain;
    ent->die = player_die;
    ent->touch = NULL;

    ent->pain_debounce_framenum = level.framenum;
    ent->targetname = NULL;

    ent->prethink = NULL;
    ent->think = Bot_Think;
    ent->nextthink = level.framenum + 1;
    ent->svflags = 0;  // temporary, for SelectSpawnPoint
    ent->s.renderfx = 0;
    ent->s.effects = 0;

    SelectSpawnPoint(ent, spawn_origin, spawn_angles);
    VectorCopy(spawn_origin, ent->s.origin);
    VectorCopy(spawn_angles, ent->s.angles);

    ent->s.origin[2] += 1;  // make sure off ground
    VectorCopy(ent->s.origin, ent->s.old_origin);
    VectorCopy(ent->s.origin, ent->old_origin);
    ent->groundentity = NULL;

    ent->svflags = SVF_MONSTER;

    {
        qboolean success = moveEntUntilNotStartsolid(ent, MASK_BOTSOLIDX, 0, 0, 1, 16);
        if (!success) {
            vec3_t forward;

            VectorCopy(ent->s.old_origin, ent->s.origin);
            AngleVectors(ent->s.angles, forward, NULL, NULL);
            success = moveEntUntilNotStartsolid(ent, MASK_BOTSOLIDX, forward[0], forward[1], 1, 16);

            if (!success) {
                VectorCopy(ent->s.old_origin, ent->s.origin);
                gi.dprintf("PutBotInServer: attempt to fix bot spawn startsolid failed (%f,%f,%f)\n",
                            ent->s.old_origin[0], ent->s.old_origin[1], ent->s.old_origin[2]);
            }
        }

        if (success) {
            VectorCopy(ent->s.origin, ent->s.old_origin);
            VectorCopy(ent->s.origin, ent->old_origin);
        }
    }

    VectorSet(ent->velocity, 0, 0, 0);
    VectorSet(ent->client->oldvelocity, 0, 0, 0);
    ent->client->ps.pmove.pm_flags &= ~PMF_DUCKED;

    Set_BotAnim(ent, ANIM_BASIC, FRAME_run1, FRAME_run6);
    client->anim_run = true;

    ent->client->quad_framenum = level.framenum;
    ent->client->invincible_framenum = level.framenum;
    ent->client->enviro_framenum = level.framenum;
    ent->client->breather_framenum = level.framenum;
    ent->client->weaponstate = WEAPON_READY;
    ent->takedamage = DAMAGE_AIM;
    ent->air_finished_framenum = level.framenum + SECS_TO_FRAMES(12);
    ent->clipmask = MASK_PLAYERSOLID;
    ent->flags &= ~FL_NO_KNOCKBACK;

    ent->client->anim_priority = ANIM_BASIC;
    ent->s.frame = FRAME_run1 - 1;
    ent->client->anim_end = FRAME_run6;
    ent->deadflag = DEAD_NO;
    ent->svflags &= ~SVF_DEADMONSTER;

    zc->waitin_obj = NULL;
    zc->first_target = NULL;
    zc->second_target = NULL;
    zc->zcstate = STS_IDLE;

    if (ent->client->resp.enterframe == level.framenum) {
        gi.WriteByte(SVC_MUZZLEFLASH);
        gi.WriteShort(ent - g_edicts);
        gi.WriteByte(MZ_LOGIN);
        gi.multicast(ent->s.origin, MULTICAST_PVS);
    }
    else {
        gi.WriteByte(SVC_MUZZLEFLASH);
        gi.WriteShort(ent - g_edicts);
        gi.WriteByte(MZ_RESPAWN);
        gi.multicast(ent->s.origin, MULTICAST_PVS);
    }

    if (!KillBox(ent)) {
        // couldn't spawn in?
    }

    gi.linkentity(ent);
    SetVWepInfo(ent);
    G_TouchTriggers(ent);
}

//----------------------------------------------------------------
// Spawn Bot
//----------------------------------------------------------------
qboolean SpawnBot(int i)
{
    edict_t     *bot;

    if (Get_NumOfPlayer() >= game.maxclients) {
        gi.cprintf(NULL, PRINT_HIGH, "Can't add bot (maxclients reached)\n");
        return false;
    }

    bot = Get_NewClient();
    if (bot == NULL) {
        gi.cprintf(NULL, PRINT_HIGH, "Can't add bot (no free edicts)\n");
        return false;
    }

    InitializeBot(bot, i);
    PutBotInServer(bot);

    // Give proper weapons/items for current match state
    TDM_SetInitialItems(bot);

    // Register with TDM team system (inline JoinedTeam without respawn/menu/hud)
    if (!teaminfo[bot->client->pers.team].captain) {
        TDM_SetCaptain(bot->client->pers.team, bot);
    }
    TDM_UpdateTeamNames();

    // Set player skin configstring
    gi.configstring(CS_PLAYERSKINS + (bot - g_edicts) - 1,
            va("%s\\%s", bot->client->pers.netname,
                    teaminfo[bot->client->pers.team].skin));

    // Set team skins for all players
    TDM_SetAllTeamSkins(bot);
    TDM_SetTeamSkins(bot, NULL);

    // Set spectator overlay string
    if (g_gamemode->value != GAMEMODE_1V1) {
        gi.configstring(CS_TDM_SPECTATOR_STRINGS + (bot - g_edicts) - 1,
                va("%s (%s)", bot->client->pers.netname,
                        teaminfo[bot->client->pers.team].name));
    } else {
        gi.configstring(CS_TDM_SPECTATOR_STRINGS + (bot - g_edicts) - 1,
                bot->client->pers.netname);
    }

    TDM_TeamsChanged();

    return true;
}

//----------------------------------------------------------------
// Spawn Call - spawn waiting bots
//----------------------------------------------------------------
void Bot_SpawnCall(void)
{
    int i;

    for (i = 0; i < MAXBOTS; i++) {
        if (Bot[i].spflg == BOT_SPRESERVED) {
            if (SpawnBot(i)) {
                Bot[i].spflg = BOT_SPAWNED;
            }
            else {
                Bot[i].spflg = BOT_SPAWNNOT;
                targetindex = 0;
            }
            SpawnWaitingBots--;
            break;
        }
    }
}

//----------------------------------------------------------------
// SpawnBotReserving - reserve a bot for spawning
//----------------------------------------------------------------
void SpawnBotReserving(void)
{
    int i;

    for (i = 0; i < MAXBOTS; i++) {
        if (Bot[i].spflg == BOT_SPAWNNOT) {
            Bot[i].spflg = BOT_SPRESERVED;
            SpawnWaitingBots++;
            return;
        }
    }
    gi.cprintf(NULL, PRINT_HIGH, "Now max of bots(%i) already spawned.\n", MAXBOTS);
}

//----------------------------------------------------------------
// Bot_SpawnOne - spawn a single bot (for sv addbot)
//----------------------------------------------------------------
void Bot_SpawnOne(void)
{
    SpawnBotReserving();
    if (SpawnWaitingBots > 0) {
        Bot_SpawnCall();
    }
}

//----------------------------------------------------------------
// Remove Bot
//----------------------------------------------------------------
void RemoveBot(void)
{
    int         i;
    int         botindex;
    edict_t     *e;
    gclient_t   *client;

    for (i = MAXBOTS - 1; i >= 0; i--) {
        if (Bot[i].spflg == BOT_SPAWNED || Bot[i].spflg == BOT_NEXTLEVEL) {
            break;
        }
    }

    if (i < 0) {
        gi.cprintf(NULL, PRINT_HIGH, "No Bots in server.\n");
        return;
    }

    botindex = i;

    for (i = game.maxclients; i >= 1; i--) {
        e = &g_edicts[i];
        if (!e->inuse) {
            continue;
        }
        client = &game.clients[i - 1];
        if (client == NULL) {
            continue;
        }
        if (!client->pers.connected && (e->svflags & SVF_MONSTER)) {
            if (client->zc.botindex == botindex) {
                if (Bot[botindex].spflg != BOT_NEXTLEVEL) {
                    Bot[botindex].spflg = BOT_SPAWNNOT;
                }
                else {
                    Bot[botindex].spflg = BOT_SPRESERVED;
                }

                // Unregister from TDM team system
                TDM_LeftTeam(e, false);
                TDM_TeamsChanged();

                gi.bprintf(PRINT_HIGH, "%s disconnected\n", e->client->pers.netname);

                // send effect
                gi.WriteByte(SVC_MUZZLEFLASH);
                gi.WriteShort(e - g_edicts);
                gi.WriteByte(MZ_LOGOUT);
                gi.multicast(e->s.origin, MULTICAST_PVS);

                e->s.modelindex = 0;
                e->solid = SOLID_NOT;

                gi.linkentity(e);

                e->inuse = false;
                G_FreeEdict(e);
                return;
            }
        }
    }
    gi.cprintf(NULL, PRINT_HIGH, "Can't find bot to remove.\n");
}

//----------------------------------------------------------------
// Bot_RemoveOne - remove a single bot (for sv removebot)
//----------------------------------------------------------------
void Bot_RemoveOne(void)
{
    RemoveBot();
}

//----------------------------------------------------------------
// Level Change Removing
//----------------------------------------------------------------
void Bot_LevelChange(void)
{
    int i, k;

    k = 0;

    for (i = 0; i < MAXBOTS; i++) {
        if (Bot[i].spflg) {
            if (Bot[i].spflg == BOT_SPAWNED) {
                k++;
                Bot[i].spflg = BOT_NEXTLEVEL;
            }
        }
    }
    for (i = 0; i < k; i++) {
        RemoveBot();
    }

    SpawnWaitingBots = k;
}


//----------------------------------------------------------------
// BotEndServerFrame - simplified end-of-frame for bots
//----------------------------------------------------------------
void BotEndServerFrame(edict_t *ent)
{
    vec3_t  v;

    if (!ent->client) {
        return;
    }

    // set model angles from AI-determined angles
    ent->s.angles[ROLL] = 0;

    // calculate speed for animation
    VectorSubtract(ent->s.origin, ent->s.old_origin, v);
    v[2] = 0;
    float xyspeed = VectorLength(v) * 10;

    // falling damage
    if (ent->groundentity && ent->client->oldvelocity[2] < -300) {
        float delta = (ent->client->oldvelocity[2] + 300) * 0.05f;
        if (delta < -1) {
            int damage = (int)(-delta);
            if (damage > 0) {
                // minimal falling damage for bots
                if (damage > 10) {
                    T_Damage(ent, &g_edicts[0], &g_edicts[0], vec3_origin, ent->s.origin, vec3_origin, damage, 0, DAMAGE_NO_ARMOR, MOD_FALLING);
                }
            }
        }
    }

    // set effects
    ent->s.effects = 0;
    ent->s.renderfx = 0;

    if (ent->health <= 0) {
        ent->client->anim_priority = ANIM_DEATH;
        if (ent->s.modelindex != skullindex && ent->s.modelindex != headindex) {
            if (ent->s.frame < ent->client->anim_end) {
                ent->s.frame++;
            }
        }
        else {
            ent->s.frame = 0;
        }
    }
    else {
        // advance animation frames
        if (ent->s.frame < ent->client->anim_end) {
            ent->s.frame++;
        }
        else {
            // loop run animation
            if (ent->client->anim_priority == ANIM_BASIC) {
                if (xyspeed > 5) {
                    ent->s.frame = FRAME_run1;
                    ent->client->anim_end = FRAME_run6;
                }
                else {
                    ent->s.frame = FRAME_stand01;
                    ent->client->anim_end = FRAME_stand40;
                }
            }
        }
    }

    // powerup effects
    if (ent->client->quad_framenum > level.framenum) {
        ent->s.effects |= EF_QUAD;
    }
    if (ent->client->invincible_framenum > level.framenum) {
        ent->s.effects |= EF_PENT;
    }

    VectorCopy(ent->velocity, ent->client->oldvelocity);

    // clear weapon kicks
    VectorClear(ent->client->kick_origin);
    VectorClear(ent->client->kick_angles);
}
