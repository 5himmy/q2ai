/*
 * ai_bridge.c -- POSIX shared memory bridge between C client and Python trainer
 *
 * Each frame:
 *   1. Extract observations from game state
 *   2. Compute reward from frame deltas
 *   3. Write obs + reward to shared memory
 *   4. Signal Python (sem_post obs_ready)
 *   5. Wait for Python's action (sem_wait act_ready, with timeout)
 *   6. Read action from shared memory
 */

#include "client.h"
#include "ai_bridge.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <errno.h>

// Global action state, read by input.c
ai_action_t q2ai_shared_action;

static ai_shared_state_t *shm_state;
static int shm_fd = -1;
static sem_t *sem_obs_ready;
static sem_t *sem_act_ready;
static bool bridge_active;
static int last_server_frame = -1;
static bool awaiting_respawn;      // after reset request, wait for alive before marking new deaths

// Instance ID for multiple parallel clients
static cvar_t *q2ai_instance;

static char shm_name[64];
static char sem_obs_name[64];
static char sem_act_name[64];

void AI_Bridge_Init(void)
{
    int instance;

    // Skip bridge init in demo mode (no Python trainer needed)
    cvar_t *demo = Cvar_Get("q2ai_demo", "0", 0);
    if (demo && demo->integer)
        return;

    q2ai_instance = Cvar_Get("q2ai_instance", "0", 0);
    instance = q2ai_instance->integer;

    // Build unique names for this instance
    Q_snprintf(shm_name, sizeof(shm_name), "/q2ai_state_%d", instance);
    Q_snprintf(sem_obs_name, sizeof(sem_obs_name), "/q2ai_obs_%d", instance);
    Q_snprintf(sem_act_name, sizeof(sem_act_name), "/q2ai_act_%d", instance);

    // Create shared memory
    shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        Com_EPrintf("Q2AI: shm_open(%s) failed: %s\n", shm_name, strerror(errno));
        return;
    }

    if (ftruncate(shm_fd, sizeof(ai_shared_state_t)) < 0) {
        Com_EPrintf("Q2AI: ftruncate failed: %s\n", strerror(errno));
        close(shm_fd);
        shm_fd = -1;
        return;
    }

    shm_state = mmap(NULL, sizeof(ai_shared_state_t),
                      PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_state == MAP_FAILED) {
        Com_EPrintf("Q2AI: mmap failed: %s\n", strerror(errno));
        close(shm_fd);
        shm_fd = -1;
        shm_state = NULL;
        return;
    }

    memset(shm_state, 0, sizeof(ai_shared_state_t));

    // Create semaphores
    sem_obs_ready = sem_open(sem_obs_name, O_CREAT, 0666, 0);
    sem_act_ready = sem_open(sem_act_name, O_CREAT, 0666, 0);

    if (sem_obs_ready == SEM_FAILED || sem_act_ready == SEM_FAILED) {
        Com_EPrintf("Q2AI: sem_open failed: %s\n", strerror(errno));
        AI_Bridge_Shutdown();
        return;
    }

    bridge_active = true;
    Com_Printf("Q2AI: Bridge initialized (instance %d, shm=%s)\n",
               instance, shm_name);
}

void AI_Bridge_Shutdown(void)
{
    if (shm_state && shm_state != MAP_FAILED) {
        munmap(shm_state, sizeof(ai_shared_state_t));
        shm_state = NULL;
    }

    if (shm_fd >= 0) {
        close(shm_fd);
        shm_unlink(shm_name);
        shm_fd = -1;
    }

    if (sem_obs_ready && sem_obs_ready != SEM_FAILED) {
        sem_close(sem_obs_ready);
        sem_unlink(sem_obs_name);
        sem_obs_ready = NULL;
    }

    if (sem_act_ready && sem_act_ready != SEM_FAILED) {
        sem_close(sem_act_ready);
        sem_unlink(sem_act_name);
        sem_act_ready = NULL;
    }

    bridge_active = false;
    Com_Printf("Q2AI: Bridge shutdown\n");
}

void AI_Bridge_ExchangeFrame(void)
{
    if (!bridge_active || !shm_state)
        return;

    // Only exchange on new server frames (10Hz), not every client frame (~1000Hz)
    if (cl.frame.number == last_server_frame)
        return;
    last_server_frame = cl.frame.number;

    // 1. Extract observations
    AI_Observe_Update(&shm_state->obs);

    // 2. Compute reward
    shm_state->reward = AI_Reward_Compute();
    shm_state->cumulative_reward += shm_state->reward;

    // 3. Update connection/episode state
    shm_state->connected = (cls.state == ca_active) ? 1 : 0;

    int pm = cl.frame.ps.pmove.pm_type;

    // After reset request, wait until alive before tracking new deaths
    if (awaiting_respawn) {
        if (pm == PM_NORMAL && cl.frame.ps.stats[STAT_HEALTH] > 0) {
            awaiting_respawn = false;
            shm_state->episode_done = 0;
        }
        // Force attack button to trigger respawn immediately
        q2ai_shared_action.attack = 1;
    } else {
        // On death: auto-respawn but do NOT end episode.
        // Continuous training: reward signal (-1.0 death) teaches KaaN
        // without the overhead of episode resets.
        if (pm == PM_DEAD || pm == PM_GIB) {
            q2ai_shared_action.attack = 1;
        }
    }

    // 4. Increment sequence and signal Python
    shm_state->sequence++;
    sem_post(sem_obs_ready);

    // 5. Check for Python's action (non-blocking)
    if (sem_trywait(sem_act_ready) == 0) {
        q2ai_shared_action = shm_state->action;
    }

    // 6. Handle reset request from Python
    if (shm_state->reset_request) {
        shm_state->reset_request = 0;
        shm_state->episode_done = 0;
        shm_state->cumulative_reward = 0;
        awaiting_respawn = true;
    }
}
