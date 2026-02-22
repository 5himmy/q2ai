/*
 * ai_stubs.c -- No-op stubs for renderer, sound, screen, and video
 *
 * Used by the Q2AI headless client to satisfy linker dependencies
 * without pulling in SDL, OpenGL, or audio libraries.
 */

#include "client.h"
#include "ai_bridge.h"

// ============================================================
// Renderer stubs (refresh/refresh.h)
// ============================================================

refcfg_t r_config;

bool R_Init(bool total)
{
    return true;
}

void R_Shutdown(bool total)
{
}

void R_BeginRegistration(const char *map)
{
}

qhandle_t R_RegisterModel(const char *name)
{
    return 1;  // Non-zero so callers treat it as valid
}

qhandle_t R_RegisterImage(const char *name, imagetype_t type, imageflags_t flags)
{
    return 1;  // Non-zero — Con_RegisterMedia calls Com_Error(ERR_FATAL) on 0
}

void R_SetSky(const char *name, float rotate, bool autorotate, const vec3_t axis)
{
}

void R_EndRegistration(void)
{
}

void R_RenderFrame(const refdef_t *fd)
{
}

void R_LightPoint(const vec3_t origin, vec3_t light)
{
    VectorClear(light);
}

void R_ClearColor(void)
{
}

void R_SetAlpha(float alpha)
{
}

void R_SetColor(uint32_t color)
{
}

void R_SetClipRect(const clipRect_t *clip)
{
}

float R_ClampScale(cvar_t *var)
{
    return 1.0f;
}

void R_SetScale(float scale)
{
}

void R_DrawChar(int x, int y, int flags, int ch, qhandle_t font)
{
}

int R_DrawString(int x, int y, int flags, size_t maxChars,
                 const char *string, qhandle_t font)
{
    return x;
}

bool R_GetPicSize(int *w, int *h, qhandle_t pic)
{
    if (w) *w = 0;
    if (h) *h = 0;
    return false;
}

void R_DrawPic(int x, int y, qhandle_t pic)
{
}

void R_DrawStretchPic(int x, int y, int w, int h, qhandle_t pic)
{
}

void R_DrawKeepAspectPic(int x, int y, int w, int h, qhandle_t pic)
{
}

void R_DrawStretchRaw(int x, int y, int w, int h)
{
}

void R_UpdateRawPic(int pic_w, int pic_h, const uint32_t *pic)
{
}

void R_TileClear(int x, int y, int w, int h, qhandle_t pic)
{
}

void R_DrawFill8(int x, int y, int w, int h, int c)
{
}

void R_DrawFill32(int x, int y, int w, int h, uint32_t color)
{
}

void R_BeginFrame(void)
{
}

void R_EndFrame(void)
{
}

void R_ModeChanged(int width, int height, int flags)
{
}

bool R_VideoSync(void)
{
    return false;
}

r_opengl_config_t R_GetGLConfig(void)
{
    r_opengl_config_t cfg = {0};
    return cfg;
}

// ============================================================
// Sound stubs (client/sound/sound.h)
// ============================================================

vec3_t listener_origin;
vec3_t listener_forward;
vec3_t listener_right;
vec3_t listener_up;
int listener_entnum;

void S_Init(void)
{
}

void S_Shutdown(void)
{
}

void S_StartSound(const vec3_t origin, int entnum, int entchannel,
                  qhandle_t sfx, float fvol, float attenuation, float timeofs)
{
    // Track sounds from enemy players for AI observation
    if (entnum > 0 && entnum != cl.clientNum + 1 && entnum <= cl.maxclients) {
        if (origin)
            AI_OnEnemySound(origin);
        else
            AI_OnEnemySound(cl_entities[entnum].current.origin);
    }
}

void S_ParseStartSound(void)
{
}

void S_StartLocalSound(const char *s)
{
}

void S_StartLocalSoundOnce(const char *s)
{
}

void S_FreeAllSounds(void)
{
}

void S_StopAllSounds(void)
{
}

void S_Update(void)
{
}

void S_Activate(void)
{
}

void S_BeginRegistration(void)
{
}

qhandle_t S_RegisterSound(const char *sample)
{
    return 0;
}

void S_EndRegistration(void)
{
}

void S_RawSamples(int samples, int rate, int width, int channels, const byte *data)
{
}

int S_GetSampleRate(void)
{
    return 22050;
}

// ============================================================
// Screen stubs
// Note: SCR_UpdateScreen and SCR_DebugGraph are in screen.c
// ============================================================

// ============================================================
// Video driver stub — no window, no input devices
// ============================================================

const vid_driver_t *vid = NULL;

cvar_t *vid_geometry;
cvar_t *vid_modelist;
cvar_t *vid_fullscreen;
cvar_t *_vid_fullscreen;

bool VID_GetFullscreen(vrect_t *rc, int *freq_p, int *depth_p)
{
    return false;
}

bool VID_GetGeometry(vrect_t *rc)
{
    if (rc) {
        rc->x = 0;
        rc->y = 0;
        rc->width = 640;
        rc->height = 480;
    }
    return true;
}

void VID_SetGeometry(const vrect_t *rc)
{
}

void VID_ToggleFullscreen(void)
{
}

// ============================================================
// Cinematics stubs
// ============================================================

void SCR_InitCinematics(void)
{
}

void SCR_RunCinematic(void)
{
}

void SCR_StopCinematic(void)
{
}

void SCR_FinishCinematic(void)
{
}

void SCR_DrawCinematic(void)
{
}

void SCR_ReloadCinematic(void)
{
}

void SCR_PlayCinematic(const char *name)
{
}

int SCR_CheckForCinematic(const char *name)
{
    return 0;
}

void SCR_Cinematic_g(genctx_t *ctx)
{
}

// ============================================================
// Refresh management stubs (from src/client/refresh.c)
// ============================================================

void CL_InitRefresh(void)
{
    cls.ref_initialized = true;

    // Initialize screen cvars and console media (normally done in refresh.c)
    // Must happen before CL_RestartFilesystem calls SCR_RegisterMedia
    V_Init();
    SCR_Init();
    SCR_RegisterMedia();
    Con_RegisterMedia();
}

void CL_ShutdownRefresh(void)
{
    cls.ref_initialized = false;
}

void CL_RunRefresh(void)
{
}

// ============================================================
// View stubs — minimal implementations
// ============================================================

// Renderer particle arrays referenced by effects.c
int         r_numparticles;
particle_t  r_particles[MAX_PARTICLES];

void V_Init(void)
{
}

void V_Shutdown(void)
{
}

void V_RenderView(void)
{
}

void V_AddEntity(const entity_t *ent)
{
}

void V_AddParticle(const particle_t *p)
{
}

void V_AddLight(const vec3_t org, float intensity, float r, float g, float b)
{
}

void V_AddLightStyle(int style, float value)
{
}

// gun_model and gun_frame — referenced by entities.c
int gun_frame;
qhandle_t gun_model;

float V_CalcFov(float fov_x, float width, float height)
{
    // Standard Quake 2 fov calculation
    if (fov_x < 1 || fov_x > 179)
        fov_x = 90;

    float x = width / tanf(fov_x / 360.0f * M_PIf);
    float a = atanf(height / x);
    return a * 360.0f / M_PIf;
}

void CL_UpdateBlendSetting(void)
{
}

// CL_CalcViewValues is defined in entities.c, not needed here
