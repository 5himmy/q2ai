/*
 * bot.h -- Bot function declarations, constants, route structures
 * Adapted for OpenTDM from 3ZB2 (zigock-q2)
 */
#ifndef BOTHEAD
#define BOTHEAD
#include "g_local.h"

// general func
void player_die(edict_t *self, edict_t *inflictor, edict_t *attacker, int damage, vec3_t point);

// bot spawn & remove
qboolean    SpawnBot(int i);
void        Bot_LevelChange(void);
void        Bot_SpawnCall(void);
void        RemoveBot(void);
void        SpawnBotReserving(void);
void        Bot_SpawnOne(void);
void        Bot_RemoveOne(void);

// weapon
void Weapon_Blaster(edict_t *ent);
void Weapon_Shotgun(edict_t *ent);
void Weapon_SuperShotgun(edict_t *ent);
void Weapon_Machinegun(edict_t *ent);
void Weapon_Chaingun(edict_t *ent);
void Weapon_HyperBlaster(edict_t *ent);
void Weapon_RocketLauncher(edict_t *ent);
void Weapon_Grenade(edict_t *ent);
void Weapon_GrenadeLauncher(edict_t *ent);
void Weapon_Railgun(edict_t *ent);
void Weapon_BFG(edict_t *ent);

// wideuse
qboolean Bot_trace(edict_t *ent, edict_t *other);
qboolean Bot_trace2(edict_t *ent, vec3_t ttz);
float Get_yaw(vec3_t vec);
float Get_pitch(vec3_t vec);
float Get_vec_yaw(vec3_t vec, float yaw);
void SpawnItem3(edict_t *ent, gitem_t *item);
int Bot_moveT(edict_t *ent, float ryaw, vec3_t pos, float dist, float *bottom);
void Set_BotAnim(edict_t *ent, int anim, int frame, int end);
int Get_KindWeapon(const gitem_t *it);
qboolean TargetJump(edict_t *ent, vec3_t tpos);
qboolean Bot_traceS(edict_t *ent, edict_t *other);
qboolean Bot_Fall(edict_t *ent, vec3_t pos, float dist);

void SelectSpawnPoint(edict_t *ent, vec3_t origin, vec3_t angles);
void ClientUserinfoChanged(edict_t *ent, char *userinfo);
void CopyToBodyQue(edict_t *ent);

// route util
qboolean TraceX(edict_t *ent, vec3_t p2);
void Get_RouteOrigin(int index, vec3_t pos);

// Bot Func
void BotEndServerFrame(edict_t *ent);
void SpawnItem2(edict_t *ent, gitem_t *item);
void Get_WaterState(edict_t *ent);
void Bot_Think(edict_t *self);
void PutBotInServer(edict_t *ent);
void Bot_LoadChainData(void);

// Combat AI
void Combat_Level0(edict_t *ent, int foundedenemy, int enewep, float aim, float distance, int skill);
void Combat_LevelX(edict_t *ent, int foundedenemy, int enewep, float aim, float distance, int skill);
void UsePrimaryWeapon(edict_t *ent);

// Explosion Index
void UpdateExplIndex(edict_t *ent);

// Bot AI routine
void Bots_Move_NORM(edict_t *ent);      // normal AI

// bot_move.c stubs
void M_CheckGround(edict_t *ent);
void M_CatagorizePosition(edict_t *ent);

//----------------------------------------------------------------

// moving speed
#define MOVE_SPD_WALK   20
#define MOVE_SPD_RUN    32
#define MOVE_SPD_DUCK   10
#define MOVE_SPD_WATER  16
#define MOVE_SPD_JUMP   32
#define VEL_BOT_JUMP    340
#define VEL_BOT_WJUMP   341
#define VEL_BOT_LADRUP  200
#define VEL_BOT_WLADRUP 200

// classes
#define CLS_NONE    0
#define CLS_ALPHA   1
#define CLS_BETA    2
#define CLS_GAMMA   3
#define CLS_DELTA   4
#define CLS_EPSILON 5
#define CLS_ZETA    6
#define CLS_ETA     7

// function's state P
#define PSTATE_TOP          0
#define PSTATE_BOTTOM       1
#define PSTATE_UP           2
#define PSTATE_DOWN         3

#define PDOOR_TOGGLE        32

// height
#define TOP_LIMIT           52
#define TOP_LIMIT_WATER     100
#define BOTTOM_LIMIT        -52
#define BOTTOM_LIMIT_WATER  -8190
#define BOTTOM_LIMITM       -300

// waterstate
#define WAS_NONE            0
#define WAS_FLOAT           1
#define WAS_IN              2

// route
// chaining pod state
#define GRS_NORMAL      0
#define GRS_ONROTATE    1
#define GRS_TELEPORT    2
#define GRS_ITEMS       3
#define GRS_ONPLAT      4
#define GRS_ONTRAIN     5
#define GRS_ONDOOR      6
#define GRS_PUSHBUTTON  7

#define GRS_GRAPSHOT    20
#define GRS_GRAPHOOK    21
#define GRS_GRAPRELEASE 22

#define GRS_REDFLAG     -10
#define GRS_BLUEFLAG    -11

#define POD_LOCKFRAME   15
#define POD_RELEFRAME   20

#define MAX_SEARCH      12
#define MAX_DOORSEARCH  10

// trace param
#define TRP_NOT         0
#define TRP_NORMAL      1
#define TRP_ANGLEKEEP   2
#define TRP_MOVEKEEP    3
#define TRP_ALLKEEP     4

// bot spawning status
#define BOT_SPAWNNOT    0
#define BOT_SPRESERVED  1
#define BOT_SPAWNED     2
#define BOT_NEXTLEVEL   3

// combat
#define AIMING_POSGAP       5
#define AIMING_ANGLEGAP_S   0.75
#define AIMING_ANGLEGAP_M   0.35

// team play state
#define TMS_NONE        0
#define TMS_LEADER      1
#define TMS_FOLLOWER    2

// ctf state (kept for compat, not used in opentdm)
#define CTFS_NONE       0
#define CTFS_CARRIER    1
#define CTFS_ROAMER     2
#define CTFS_OFFENCER   3
#define CTFS_DEFENDER   4
#define CTFS_SUPPORTER  5

// fire modes
#define FIRE_SLIDEMODE      0x00000001
#define FIRE_PRESTAYFIRE    0x00000002
#define FIRE_STAYFIRE       0x00000004
#define FIRE_CHIKEN         0x00000008
#define FIRE_RUSH           0x00000010
#define FIRE_JUMPNRUSH      0x00000020
#define FIRE_ESTIMATE       0x00000040
#define FIRE_SCATTER        0x00000080
#define FIRE_RUNNIN         0x00000100
#define FIRE_JUMPROC        0x00000200

#define FIRE_REFUGE         0x00001000
#define FIRE_EXPAVOID       0x00002000

#define FIRE_QUADUSE        0x00004000
#define FIRE_AVOIDINV       0x00008000

#define FIRE_BFG            0x00010000

#define FIRE_SHIFT_R        0x00020000
#define FIRE_SHIFT_L        0x00040000

#define FIRE_SHIFT          (FIRE_SHIFT_R | FIRE_SHIFT_L)

#define FIRE_REFLECT        0x00080000

#define FIRE_IGNORE         0x10000000

// means of death
#define MOD_SNIPERAIL           50
#define MOD_LOCMISSILE          51
#define MOD_BFG100K             52
#define MOD_AIRSTRIKE           70

//----------------------------------------------------------------
// general status list
#define STS_IDLE        0x00000000
#define STS_THINK       0x00000001
#define STS_LADDERUP    0x00000002
#define STS_ROCJ        0x00000004
#define STS_TURBOJ      0x00000008
#define STS_WATERJ      0x00000010

#define STS_SJMASK      (STS_ROCJ | STS_TURBOJ | STS_WATERJ)
#define STS_SJMASKEXW   (STS_ROCJ | STS_TURBOJ)

#define STS_TALKING     0x00000200
#define STS_ESC_WXPL    0x00000400

#define STS_MOVE_WPOINT 0x00000800

// wait
#define STS_W_DONT      0x00001000
#define STS_W_DOOROPEN  0x00002000
#define STS_W_ONPLAT    0x00004000
#define STS_W_ONDOORUP  0x00008000
#define STS_W_ONDOORDWN 0x00010000
#define STS_W_ONTRAIN   0x00020000
#define STS_W_COMETRAIN 0x00040000
#define STS_W_COMEPLAT  0x00080000

#define STS_WAITS       (STS_W_DONT | STS_W_DOOROPEN | STS_W_COMEPLAT | STS_W_ONPLAT | STS_W_ONDOORUP | STS_W_ONDOORDWN | STS_W_ONTRAIN)
#define STS_WAITSMASK   (STS_W_DOOROPEN | STS_W_ONPLAT | STS_W_ONDOORUP | STS_W_COMEPLAT | STS_W_ONDOORDWN | STS_W_ONTRAIN)
#define STS_WAITSMASK2  (STS_W_ONDOORDWN | STS_W_ONDOORUP | STS_W_ONPLAT | STS_W_ONTRAIN)
#define STS_WAITSMASKCOM (STS_W_DOOROPEN | STS_W_ONPLAT | STS_W_ONDOORUP | STS_W_ONDOORDWN | STS_W_ONTRAIN)

//----------------------------------------------------------------
// combat status
#define CTS_ENEM_NSEE   0x00000001
#define CTS_AGRBATTLE   0x00000002
#define CTS_ESCBATTLE   0x00000004
#define CTS_HIPBATTLE   0x00000008

#define CTS_PREAIMING   0x00000010
#define CTS_AIMING      0x00000020
#define CTS_GRENADE     0x00000040
#define CTS_JUMPSHOT    0x00000080

#define CTS_COMBS (CTS_AGRBATTLE | CTS_ESCBATTLE | CTS_HIPBATTLE | CTS_ENEM_NSEE)

//----------------------------------------------------------------
// route struct
#define MAXNODES            10000
#define MAXLINKPOD          6

typedef struct {
    vec3_t  Pt;
    union {
        vec3_t          Tcourner;
        unsigned short  linkpod[MAXLINKPOD];
    };
    edict_t *ent;
    short   index;
    short   state;
} route_t;

//----------------------------------------------------------------
// bot info struct
#define MAXBOTS     128
#define MAXBOP      16

// bot params
#define BOP_WALK        0
#define BOP_AIM         1
#define BOP_PICKUP      2
#define BOP_OFFENCE     3
#define BOP_COMBATSKILL 4
#define BOP_ROCJ        5
#define BOP_REACTION    6
#define BOP_VRANGE      7
#define BOP_HRANGE      8
#define BOP_PRIWEP      9
#define BOP_SECWEP      10
#define BOP_DODGE       11
#define BOP_ESTIMATE    12
#define BOP_NOISECHK    13
#define BOP_NOSTHRWATER 14
#define BOP_TEAMWORK    15

typedef struct {
    char    netname[21];
    char    model[21];
    char    skin[21];
    int     spflg;
    int     team;
    int     arena;
    unsigned char   param[MAXBOP];
} botinfo_t;

//----------------------------------------------------------------
// message section name
#define MESS_DEATHMATCH     "[MessDeathMatch]"

//----------------------------------------------------------------
// bot list section name
#define BOTLIST_SECTION_DM  "[BotList]"

//----------------------------------------------------------------
#define MAX_BOTSKILL        10

#define FALLCHK_LOOPMAX 30

// laser Index
#define MAX_LASERINDEX      30
extern edict_t     *LaserIndex[MAX_LASERINDEX];

// Explosion Index
#define MAX_EXPLINDEX       12
extern edict_t     *ExplIndex[MAX_EXPLINDEX];

// trigger_push idle times
#define BOT_PUSH_IDLE_TIME_AIR      2.5
#define BOT_PUSH_IDLE_TIME_GROUND   0.5

// ground test
#define GROUND_TEST_EPSILON     0.25

// MASK defines for bot traces
#ifndef MASK_BOTSOLID
#define MASK_BOTSOLID           (CONTENTS_SOLID|CONTENTS_WINDOW|CONTENTS_MONSTER)
#endif
#ifndef MASK_BOTSOLIDX
#define MASK_BOTSOLIDX          (CONTENTS_SOLID|CONTENTS_WINDOW|CONTENTS_PLAYERCLIP|CONTENTS_MONSTER)
#endif
#ifndef MASK_BOTGROUND
#define MASK_BOTGROUND          MASK_BOTSOLIDX
#endif

// model scale
#ifndef MODEL_SCALE
#define MODEL_SCALE     1.000000
#endif

// Xatrix/ROGUE weapon defines (not available in opentdm, stub out)
#ifndef WEAP_PHALANX
#define WEAP_PHALANX    12
#endif
#ifndef WEAP_BOOMER
#define WEAP_BOOMER     13
#endif
#ifndef WEAP_TRAP
#define WEAP_TRAP       17
#endif
#ifndef WEAP_GRAPPLE
#define WEAP_GRAPPLE    20
#endif

extern  int         targetindex;
extern  int         ListedBotCount;
extern  int         SpawnWaitingBots;
extern  char        ClientMessage[MAX_STRING_CHARS];
extern  botinfo_t   Bot[MAXBOTS];
extern  route_t     Route[MAXNODES];
extern  int         CurrentIndex;
extern  float       JumpMax;
extern  int         botskill;
extern  int         trace_priority;
extern  int         FFlg[MAX_BOTSKILL];
extern  int         ListedBots;

// for avoid abnormal frame error
extern  int         skullindex;
extern  int         headindex;

// pre-searched items (cast to non-const for zigock compat)
extern  gitem_t *Fdi_BLASTER;
extern  gitem_t *Fdi_SHOTGUN;
extern  gitem_t *Fdi_SUPERSHOTGUN;
extern  gitem_t *Fdi_MACHINEGUN;
extern  gitem_t *Fdi_CHAINGUN;
extern  gitem_t *Fdi_GRENADES;
extern  gitem_t *Fdi_GRENADELAUNCHER;
extern  gitem_t *Fdi_ROCKETLAUNCHER;
extern  gitem_t *Fdi_HYPERBLASTER;
extern  gitem_t *Fdi_RAILGUN;
extern  gitem_t *Fdi_BFG;

// Xatrix/ROGUE/CTF weapons (NULL in opentdm, kept for code compat)
extern  gitem_t *Fdi_PHALANX;
extern  gitem_t *Fdi_BOOMER;
extern  gitem_t *Fdi_TRAP;
extern  gitem_t *Fdi_MAGSLUGS;
extern  gitem_t *Fdi_GRAPPLE;

extern  gitem_t *Fdi_SHELLS;
extern  gitem_t *Fdi_BULLETS;
extern  gitem_t *Fdi_CELLS;
extern  gitem_t *Fdi_ROCKETS;
extern  gitem_t *Fdi_SLUGS;

extern  gitem_t *zflag_item;
extern  edict_t *zflag_ent;
extern  int      zigflag_spawn;

#endif
