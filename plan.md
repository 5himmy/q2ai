# Q2AI — Project Roadmap

## 1. Project Overview

**Goal:** Build a learning AI (KaaN) that plays Quake 2 competitively using reinforcement learning, training against scripted bots and eventually surpassing them.

**Stack:**
- **Server:** q2pro + opentdm (competitive duel/TDM) + zigock/3ZB2 scripted bots as opponents
- **Client:** Headless q2pro client (no rendering/audio) — connects like a real player
- **IPC:** POSIX shared memory + semaphores between C client and Python trainer
- **Training:** Python (PyTorch, Gymnasium, PPO) with behavior cloning warm start
- **Game mode:** Start with 1v1 duel, expand to team modes later

## 2. Architecture

```
q2pro server (opentdm + scripted bots)
     | Q2 network protocol (localhost)
q2ai headless client (stripped q2pro)
     | POSIX shared memory + semaphores
Python trainer (PyTorch, Gymnasium env, PPO)
```

**Multi-instance parallel training:**
```
Server:27910 + KaaN[instance=0]  ─┐
Server:27911 + KaaN[instance=1]  ─┤── Python trainer (--num-envs N)
Server:27912 + KaaN[instance=2]  ─┤
Server:27913 + KaaN[instance=3]  ─┘
```

**Why headless client instead of server-side?** The scripted bots bypass player physics (directly set `ent->s.origin` instead of generating `usercmd_t` through `Pmove`). By connecting as a real client, the AI uses proper player physics — strafe jumping, rocket jumping, realistic movement — same as a human player.

## 3. Current Status

### Phase 0 — Bot Integration: COMPLETE

- 3ZB2 bot code ported to opentdm (~9,646 lines)
- `g_bot_skill` cvar (1-10), `g_bot_config` cvar + `botconfig.cfg`
- 43+ waypoint chain files, 10 skill levels, 16 parameters per bot
- Auto-ready for bots in warmup (opentdm `g_main.c` calls `TDM_Ready_f()`)

### Phase 1 — Minimal Headless Client: COMPLETE

- Stripped q2pro client of rendering/audio/UI
- `ai_stubs.c`: No-op stubs for all `R_*`, `S_*`, `SCR_*`, `UI_*` functions
- `#ifdef Q2AI_HEADLESS` guards in `main.c`, `input.c`, `precache.c`
- Meson build target `q2ai` with `-DQ2AI_HEADLESS=1 -DUSE_CLIENT=1`, no SDL/OpenGL/sound deps
- Binary: ~3MB, connects to server and stays connected

### Phase 2 — Observation Extraction: COMPLETE

- `ai_observe.c`: Extracts game state into `ai_observation_t` struct each frame
- Self state: position, velocity, angles, health, armor, ammo, weapon, inventory, pm_flags, frags
- Entity state: visible entities with positions, angles, modelindex, effects, entity type classification
- Entity types: `ENT_TYPE_PLAYER`, `ENT_TYPE_ITEM`, `ENT_TYPE_PROJECTILE`
- Enemy tracking: nearest enemy player position, angles, visibility flag
- Game state: map name, server frame/time

### Phase 3 — AI Input Replacement: COMPLETE

- `ai_input.c`: `AI_GenerateCmd()` replaces keyboard/mouse input
- Reads `ai_action_t` from shared memory, sets viewangles, movement, buttons, weapon impulse
- Player moves correctly when spectated from regular q2pro client

### Phase 4 — Shared Memory Bridge: COMPLETE

- `ai_bridge.c`: POSIX shared memory (`/q2ai_state_N`) + semaphores (`/q2ai_obs_N`, `/q2ai_act_N`)
- Non-blocking `sem_trywait` for action reads (avoids frame loop stalls)
- Per-frame: write obs → signal Python → check for action → read action
- Multiple instance support via `q2ai_instance` cvar
- Python side: `q2ai_env.py` (Gymnasium), `q2ai_structs.py` (ctypes mirrors)

### Phase 5 — Reward System + Episode Management: COMPLETE

- `ai_reward.c`: V2.0 simplified — core combat + critical anti-suicide only
  - +1.0 per frag, -1.0 per death
  - -damage/100 for damage taken
  - +damage/100 for damage dealt (FOV-gated: only when enemy within 45°)
  - -0.15 self-splash penalty (shooting wall within 120u)
  - -0.02 blaster penalty (has better weapon but won't switch)
  - -0.01 stuck penalty (speed < 20)
  - +0.1 per health/armor pickup, +0.15 per weapon pickup
  - Stripped from 15 signals to 9 (value loss 47→0.4)
- `ai_control.c`: Auto-connect, auto-join team, auto-respawn, auto-ready, name override
- Full episode loop: connect → join team → ready → play → die → respawn → repeat
- Cvars: `q2ai_team`, `q2ai_name`, `q2ai_autoconnect`, `q2ai_server`, `q2ai_instance`

### Phase 6 — Training Pipeline: IN PROGRESS

#### Completed:
- [x] `python/model.py` — ActorCritic with hybrid action space
  - 4 continuous + 3 binary + 11 weapon logits
  - State-independent learned log_std for continuous actions
  - Orthogonal initialization, bias tuning for safe initial behavior
  - `act_batch()` — batched N-env inference in one GPU forward pass
- [x] `python/train.py` — PPO training loop with multi-instance support
  - `--num-envs N` for N parallel environments (tested up to 300)
  - CUDA/GPU support (RTX 3070)
  - GAE advantage estimation, mini-batch updates, gradient clipping
  - CSV logging, periodic checkpoints
  - ThreadPoolExecutor for parallel semaphore waits (I/O-bound recv_obs)
  - Semaphore drain + no-op resync between rollout iterations
  - Batched GPU inference via `act_batch()` (1 forward pass per step, not N)
  - Pipelined env I/O: send all actions (non-blocking) → recv all obs (parallel)
- [x] `python/q2ai_env.py` — Gymnasium environment (110-dim observation vector)
  - Normalized observations: position/4096, velocity/400, sin/cos angles, etc.
  - 8 nearest entities with relative position and type
  - 18 wall raycasts for spatial awareness (eye + knee height + floor/ceiling)
  - Inventory: 10 weapon ownership flags + 6 normalized ammo counts
  - Split step() into send_action() (non-blocking) + recv_obs() (blocking wait)
- [x] `python/q2ai_structs.py` — ctypes mirrors of C structs
- [x] Behavior cloning from demo playback (`ai_demo.c` + `python/imitate.py`)
  - Supports both MVD (.mvd2) and DM2 (.dm2) demo formats
  - Infers actions from state deltas (viewangle changes, velocity, gunframe)
  - Auto-quit when demo playback ends (stale frame detection)
  - Batch extraction script for all q2dm1 demos
- [x] Multi-instance training script (`q2ai_train.sh start N`)
  - Starts N server+KaaN pairs on ports 27910..27910+N
  - Each KaaN uses unique `q2ai_instance` for separate shared memory
  - Single Python trainer connects to all N environments
- [x] `q2ai_train.sh` — Control script (start/stop/restart/status/log)
- [x] Demo extraction: 53 q2dm1 demos, 17,539 frames combined
- [x] BC model retrained on full dataset (loss 0.098)
- [x] Scaled to 300 parallel instances (300 servers + 300 KaaN clients)
  - ~1,300 steps/sec throughput, ~30s per iteration (38,400 steps/iter)
  - CPU: ~90% on 12-core, GPU: bursts during PPO updates, RAM: 12/16GB
- [x] Continuous episodes — no episode resets on death
  - Auto-respawn in C (`ai_bridge.c`), death penalty (-1.0) still teaches
  - Eliminates Python-side reset overhead entirely (0 episodes in logs)
- [x] CPU optimization: `Sys_Sleep(2)` in headless main loop, `cl_maxfps 30`
- [x] Reward shaping V1.4 — damage dealt via STAT_FLASHES, hit confirmation, tighter aim, reduced shaping
- [x] Reward shaping V1.5 — FOV-gated damage, graduated aim precision, no blind-fire penalty, attack bias 0.5
- [x] Wall raycasts — 18 rays (8 eye-height + floor + ceiling + 8 knee-height) via CM_BoxTrace
- [x] Inventory fix — server-side svc_inventory push on spawn, fixed off-by-one index mapping
- [x] `python/monitor.py` — HTML evolution dashboard (auto-starts with training)
  - Kanban-style milestone cards, training charts, server rcon queries
  - Generates `evolve.html`, saves history to `evolve_data.json`
- [x] Entropy fix: log_std clamped [-2.0, 0.5], entropy_coef tuned through 0.003→0.01→0.005
  - Prevented entropy runaway (was 16.4 and climbing, now stable ~8.3-8.6)
- [x] Reward scaling: 5x multiplier so value function learns faster
- [x] Observation normalization: running mean/std (Welford's algorithm)
  - Adaptive normalization on top of fixed scaling, clips to [-5, 5]
- [x] Batch size: 128 → 512 (reduces gradient updates from 1,200 to 300 per iteration)
  - Faster iterations (26s vs 33s), more stable gradients

#### Completed (V2.0–V2.6):
- [x] V2.0 LSTM Memory — separate actor/critic LSTMs (256 hidden), sequence-based PPO
  - Truncated BPTT with 32-step chunks, hidden state carried between chunks
  - `act_batch()` carries per-env LSTM hidden states between frames
  - Hidden states reset on death
- [x] V2.5 Grouped Encoder — Fourier position encoding, entity attention, grouped encoders
  - Self/Inventory/Enemy/Entity groups with separate encoders
  - 2-layer fusion (292→256→256), ~1.2M params
- [x] V2.6 Wall Raycasts + Inventory — spatial awareness and correct inventory
  - 18-ray CM_BoxTrace (8 eye + floor + ceiling + 8 knee), OBS_DIM 92→110
  - Server-side svc_inventory push on spawn, fixed off-by-one index mapping
  - V1.5 precision rewards: FOV-gated damage, graduated aim, attack bias 0.5
  - Bot spawn fix: startsolid nudge range 16→48 for hop map
  - Training map: hop (smaller than 2box6, more frequent encounters)
- [x] V2.7 Reward Simplification + Curriculum Learning
  - Stripped rewards from 15 signals to 9 (value loss 47→0.4)
  - Removed: aim tracking, precision bonus, shooting reward, approach, item proximity, backward, crouch, circling
  - Kept: frag, death, damage dealt/taken, self-splash, blaster, stuck, pickups
  - Curriculum: bot skill 3→1, graduate as K/D improves
  - Hyperparams: lr 3e-5→1e-4, epochs 2→4, value_coef 0.5→1.0, entropy_coef→0.005
  - Rollout 128→256 (more combat events per rollout for sparse rewards)
  - Removed hardcoded bias overrides on resume (let the network learn)
  - CSV now logs step_reward, frags, deaths per iteration

#### Completed (V2.8):
- [x] V2.8 Adaptive Hyperparameter Tuning — `AdaptiveTuner` class in `train.py`
  - Entropy coef: targets 7.0, decays ×0.995 when above, grows ×1.002 when below, clamped [0.0005, 0.02]
  - LR decay: linear from initial LR to 0 over total iterations (standard PPO practice)
  - Reward scale: adjusts based on value_loss magnitude (shrink if >10, grow if <0.1), clamped [1.0, 20.0]
  - Bot skill curriculum: rolling K/D over last 10 iterations, graduate at K/D>0.5, demote at K/D<0.05
  - Sends `g_bot_skill N` via rcon to all servers on skill change
  - All tuner values logged to CSV and console (`[TUNER]` lines)

#### In Progress:
- [ ] V2.8 training: 250 instances on hop, adaptive tuning active

#### Next:
- [ ] Multi-client per server (4 KaaN per server, cut 250 servers to 63)
- [ ] Add back shaping rewards one at a time once basic combat learned

### Phase 7 — 1v1 Mastery

- [ ] Item timing and map control
- [ ] Weapon selection strategy
- [ ] Movement optimization (strafe jumping, rocket jumping)
- [ ] Opponent modeling
- [x] Curriculum learning: automatic bot skill via AdaptiveTuner (V2.8)

### Phase 8 — Human Demo Learning

- [x] Parse q2pro MVD/DM2 demos to extract (state, action) pairs
- [x] Pre-train model via behavior cloning before RL fine-tuning
- [ ] Offline RL on demo data (DAgger or similar)
- [ ] Use demo data for reward shaping

### Phase 9 — Expansion

- [ ] Team modes (2v2, 4v4 TDM)
- [ ] Multiple maps
- [ ] Play against humans on live servers

## 4. Action Space

AI generates per-frame actions converted to real player inputs (`usercmd_t`):

| Field | Range | Description |
|-------|-------|-------------|
| `yaw_delta` | ±10.0 degrees | Look left/right change |
| `pitch_delta` | ±5.0 degrees | Look up/down change |
| `forwardmove` | -1.0 to 1.0 | Forward/backward (scaled to -400..400) |
| `sidemove` | -1.0 to 1.0 | Strafe left/right (scaled to -400..400) |
| `jump` | 0/1 | Jump (upmove=400) |
| `crouch` | 0/1 | Crouch (upmove=-400) |
| `attack` | 0/1 | Fire weapon |
| `weapon` | 0-10 | Weapon selection (0=no switch) |

## 5. Key Files

### C Headless Client (`q2pro/src/client/`)

| File | Purpose |
|------|---------|
| `ai_bridge.h` | Shared structs: `ai_observation_t`, `ai_action_t`, `ai_shared_state_t` |
| `ai_bridge.c` | POSIX shared memory IPC, frame exchange |
| `ai_observe.c` | Game state → observation struct |
| `ai_input.c` | AI actions → `usercmd_t` |
| `ai_reward.c` | Reward computation from frame deltas |
| `ai_control.c` | Episode lifecycle: connect, join, respawn, name, auto-ready |
| `ai_demo.c` | Demo playback → training data extraction (MVD + DM2) |
| `ai_stubs.c` | No-op stubs for renderer, audio, UI |

### Python (`python/`)

| File | Purpose |
|------|---------|
| `q2ai_env.py` | Gymnasium environment (89-dim obs, hybrid actions) |
| `q2ai_structs.py` | ctypes mirrors of `ai_bridge.h` |
| `model.py` | ActorCritic network (PPO + behavior cloning, LSTM planned) |
| `train.py` | PPO training loop (multi-instance, GPU, adaptive tuning) |
| `imitate.py` | Behavior cloning from demo data |
| `monitor.py` | HTML evolution dashboard (auto-starts with training) |
| `test_shm_read.py` | Shared memory test script |

### Build

```bash
# Build headless client
meson compile -C q2pro/builddir_q2ai q2ai

# Build opentdm (with bot auto-ready fix)
cd opentdm && make

# Copy binary to runtime directory
cp q2pro/builddir_q2ai/q2ai /home/oky/q2ai/q2ai
```

### Training (Multi-Instance)

The training script manages everything: servers, KaaN clients, and the Python trainer.

```bash
cd /home/oky/q2ai

# Start training with 16 parallel instances (default)
./q2ai_train.sh start

# Start with custom number of instances (tested up to 300)
./q2ai_train.sh start 300

# Check progress
./q2ai_train.sh log

# Check running processes
./q2ai_train.sh status

# Stop everything
./q2ai_train.sh stop

# Restart (stop + start)
./q2ai_train.sh restart 100
```

Each instance runs:
- A q2proded server on port 27910+i with a scripted bot
- A KaaN headless client with `q2ai_instance=i` connecting to that server
- Shared memory `/q2ai_state_i` + semaphores for IPC

The Python trainer connects to all N instances via `--num-envs N` and collects rollouts using threaded parallel I/O. Recommended instance counts by hardware:
- **12-core/16GB/RTX 3070:** 200-300 instances (~90% CPU, ~12GB RAM)
- Per-instance overhead: ~40MB RAM (server+client), ~1.4% CPU

The monitor auto-starts with training and generates an HTML dashboard:
- **Dashboard:** `evolve.html` (open in browser, auto-refreshes every 10 min)
- **Data:** `evolve_data.json` (persistent history across restarts)
- **Log:** `/tmp/monitor.log`

Run monitor standalone:
```bash
# Manual check (single run)
cd python && .venv/bin/python monitor.py --once

# Custom interval (5 minutes)
cd python && .venv/bin/python monitor.py --interval 300
```

### Behavior Cloning (Demo Extraction)

```bash
# Extract single demo
./q2ai_demo +set q2ai_demo 1 +demo filename

# Batch extract all q2dm1 demos
bash /tmp/extract_all_demos.sh

# Train BC model from extracted data
cd python
.venv/bin/python imitate.py --data /home/oky/q2ai/demo_extracts/all_demos_combined.bin --epochs 200

# Run PPO with BC warm start
.venv/bin/python train.py --resume model_bc.pt --num-envs 8
```

### Bot Configuration (`opentdm/`)

| File | Purpose |
|------|---------|
| `botconfig.cfg` | Per-bot skill/model/skin definitions |
| `bot_za.c`, `bot_fire.c`, `bot_func.c` | Scripted bot AI |
| `chdtm/*.chn` | 43+ waypoint chain files |
| `g_local.h` | `zgcl_t`, `ENT_IS_BOT()` macro |
| `g_main.c` | Bot auto-ready in warmup |

## 6. Technical Notes

- **Timing:** Server 10Hz, client `cl_maxfps 30` (capped for CPU savings), headless loop `Sys_Sleep(2)` between frames. Python inference <1ms on GPU with batched forward pass.
- **BSP maps:** Required in client's game directory for prediction/collision. Server auto-downloads them.
- **Scaling:** Tested up to 300 parallel instances on 12-core/16GB/RTX 3070. Each KaaN client uses ~1.4% CPU. Bottleneck is CPU (88-92% at 300), GPU only bursts during PPO updates.
- **Continuous episodes:** No episode resets on death — C client auto-respawns, Python sees continuous observation stream. Death penalty (-1.0) still trains combat avoidance. Eliminates all Python-side reset overhead.
- **Threading:** `ThreadPoolExecutor` for parallel `recv_obs()` calls — semaphore waits are I/O-bound (GIL released). Sequential recv at 300 envs would take 30s; parallel takes ~100ms.
- **Semaphore resync:** Between PPO update iterations, drain all stale semaphore posts (accumulated while Python was doing gradient updates) and send one no-op round to get fresh observations.
- **Server setup:** OpenTDM 1v1 mode, `q2ai_team b` since team A has a scripted bot.
- **Demo extraction:** Both MVD and DM2 formats supported. Velocity computed from position deltas (10Hz). Auto-quit on demo end.
- **GPU:** CUDA with batched inference — `act_batch()` stacks N observations into one tensor for a single GPU forward pass instead of N separate calls. Model weights stay on GPU, only final actions transferred to CPU.
- **Adaptive tuning:** `AdaptiveTuner` in `train.py` adjusts 4 hyperparameters each iteration: entropy coef (target-based), LR (linear decay), reward scale (value-loss feedback), bot skill (K/D curriculum via rcon). Logged as `[TUNER]` lines and 4 extra CSV columns.
