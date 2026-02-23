# KaaN — Evolution Milestones

## Game Rules: Quake 2 1v1 Duel (OpenTDM)

**Objective:** Score more frags than your opponent within the time limit.

**Spawning:** Players spawn with only a blaster (weakest weapon), 100 health, no armor. Everything else must be picked up from the map.

**Weapons** (strongest to weakest for duel):
- **Rocket Launcher (RL)** — splash damage, self-damage risk, mid-range king
- **Railgun (RG)** — instant hitscan, 100 damage, long reload, long range
- **Super Shotgun (SSG)** — devastating close range, slow reload
- **Chaingun (CG)** — sustained damage, good mid-range
- **Hyperblaster (HB)** — rapid fire energy weapon
- **Grenade Launcher (GL)** — bouncing explosives, area denial
- **Machine Gun (MG)** — starter upgrade, decent accuracy
- **Blaster** — infinite ammo, weak, spawn weapon

**Items:**
- **Mega Health** — +100 health (can exceed 100, decays back down), most important item on q2dm1
- **Yellow Armor** — +50 armor (max 100)
- **Health packs** — +25 health (medium) or +2 (stimpacks)
- **Ammo boxes** — weapon-specific ammunition

**Item Respawn:** Items respawn 20-30 seconds after pickup. Controlling when items respawn (timing) is the core strategic skill.

**Training Map: hop** — Tiny box map, guaranteed line of sight, frequent encounters.

**Current Mode: Instagib** (`g_gamemode 1`) — railgun only, one-shot kill, hitscan, infinite ammo (`dmflags 8192`). Simplifies training to pure aiming.

**Target Map: q2dm1 "The Edge"** — Classic duel map. Two main levels connected by ramps, lifts, and a teleporter. No red armor on this map. Key positions: RL room, mega health area, yellow armor spots. (Used after KaaN learns combat basics.)

**Match Format:**
- 3-minute timelimit (short rounds for fast iteration)
- No overtime (g_tie_mode 0)
- 1-second countdown
- Auto-ready via bots
- Instagib: railgun only, one-shot kill

**Movement Physics:**
- Base run speed: 300 units/second
- Strafe jumping: accelerate beyond 300 by jumping + strafing at angles (Quake physics exploit)
- Rocket jumping: use rocket splash damage to launch yourself to higher positions
- Crouch: slower movement, rarely useful in duel

---

## Reward Rules V3.0 (what KaaN learns from)

Hit-only RL rewards + auxiliary aim supervision. The NN learns aim direction from a dense MSE supervision loss, not from reward shaping. RL reward only fires for actual combat outcomes.

| Signal | Reward | Purpose |
|--------|--------|---------|
| Kill enemy | +1.0 | Primary objective |
| Die | -1.0 | Primary penalty |
| Deal damage | +damage/50 | Dense hit confirmation (via STAT_FLASHES) |

**Auxiliary aim supervision (not reward):**
- MSE loss between NN yaw/pitch output means and ideal aim deltas from C-side observation
- Only active when enemy is visible (`aim_mask`)
- Provides dense per-frame gradient for aim direction (vs sparse frag reward)
- `aim_loss` dropped from 1.4 → 0.01 within ~20 iterations

**V3.0 changes:** Removed ALL continuous facing/aim rewards. Previous versions tried `aim_dot * 0.02`, then `aim_dot² * 0.15` — frags stayed flat at 3-7 (random luck). Auxiliary supervision loss solved it instantly: frags jumped to 280+ within 20 iterations.

**Design philosophy:** Don't try to teach aim through reward shaping — the signal is too weak and sparse for PPO. Instead, use direct supervision (MSE loss) for aim direction, and only reward actual combat outcomes (hits, frags, deaths). The NN learns "where to aim" from supervision and "when to shoot" from RL.

---

## Training Infrastructure

- **KaaN V3.0** — 250 parallel instances on 12-core/16GB/RTX 3070
- **Map:** hop (small box map, guaranteed line of sight, frequent encounters)
- **Game mode:** Instagib (`g_gamemode 1`) — railgun only, one-shot kill, hitscan, infinite ammo (`dmflags 8192`)
- **Architecture:** V2.6 Grouped Encoder + Entity Attention + Fourier Position + LSTM (1.2M params)
- **Separate Actor/Critic LSTMs** — no shared trunk (V1.2 fix: value gradients were destroying policy)
- **Continuous episodes** — no resets on death. Auto-respawn in C, continuous observation stream.
- **~1,300 steps/sec** throughput, ~26s per iteration (64,000 steps/iter at 250 instances)
- **Behavior cloning warm start** from 53 human demos on q2dm1 (model_bc.pt)
- **Reward scaling** — adaptive via AdaptiveTuner (started 5x, auto-adjusts)
- **Observation normalization** — running mean/std (Welford's algorithm), clips to [-5, 5]
- **PPO hyperparameters:** lr=1e-4 (linear decay), epochs=4, value_coef=1.0, target_kl=0.05, entropy_coef=0.005
- **Rollout:** 256 steps/env, seq_len=32 (LSTM chunks), batch_chunks=16
- **Auxiliary aim supervision:** MSE loss on yaw/pitch means vs ideal aim deltas (aim_coef=1.0)
- **Per-dimension std clamping:** yaw/pitch [-4.0, -2.5] (max ±0.8°), movement [-2.0, 0.5]
- **Input smoothing:** EMA low-pass filter (0.6/0.4) on yaw/pitch deltas in C client
- **Curriculum learning:** bot skill auto-graduates via AdaptiveTuner (K/D thresholds)
- **Damage dealt tracking** — opentdm exposes per-frame damage via STAT_FLASHES (self-damage filtered)
- **Dashboard sync:** evolve.html auto-synced to web server every 30 min via scp
- Batched GPU inference, threaded parallel I/O, semaphore resync between PPO updates
- Real frag/death tracking from game state (not episode counters)
- CSV logs: step_reward, frags, deaths, aim_loss per iteration

---

## Stage 1: Aim Training (current stage)

KaaN trains on hop — a tiny box map with guaranteed line of sight. Instagib mode (railgun only, one-shot kill). Bot stands still (`g_bot_nomove 1`, `g_bot_noshoot 1`). KaaN also stationary (`q2ai_nomove 1`) — pure mouse aiming training.

**V3.0 approach:** Auxiliary aim supervision loss teaches aim direction (MSE between NN output means and ideal aim deltas from C observation). RL reward only for actual hits. Per-dimension std clamping keeps yaw/pitch tight (max ±0.8° scatter). EMA low-pass filter smooths sampling noise.

**Results (V3.0, hop, 250 instances, 35 iterations):**
- aim_loss: 1.4 → 0.01 (NN learned aim direction)
- Frags: 3 (random) → 280-395 per iteration
- Deaths: 0 (bot can't shoot)
- step_reward: 0.03-0.05 (consistently positive)
- Entropy: 3.88-3.94 (stable)
- Bot skill: auto-graduated 1 → 3 → 4 (meaningless since bot is stationary)

**What worked:**
- Auxiliary aim supervision (MSE loss) — solved aim in ~20 iterations where reward shaping failed
- Hit-only rewards — removes noise from continuous facing rewards
- Per-dimension std clamping — prevents sampling jitter from ruining precise aim

**What didn't work (tried and abandoned):**
- Facing reward `aim_dot * 0.02` → frags stayed flat at 3-7 (too weak for PPO)
- Sharper `aim_dot² * 0.15` → still flat (gradient too sparse)
- Uniform std clamping → movement and aim need different exploration ranges

**Known issues:**
- EMA low-pass filter causes slow zigzag circling around target (momentum overshoot)
- Bot skill graduation is meaningless with stationary bots

**Next:** Enable bot shooting (Stage 2) or bot movement (Stage 3) once aim is stable.

**Architecture:** V2.6 Grouped Encoder + Entity Attention + Fourier Position + LSTM (1.2M params). Separate actor/critic LSTMs with truncated BPTT (seq_len=32).

---

## Stage 2: Survivor

KaaN learns that being alive is good and dying is bad. He starts avoiding damage — backing away from rockets, not running off ledges. He discovers that health and armor pickups make him harder to kill.

**Signs of progress:**
- Stays alive for 20+ seconds between deaths
- Picks up health/armor when nearby (not by accident)
- Doesn't suicide with own rockets
- Moves away from danger (negative health delta = retreat)
- Frags are still mostly lucky

**Metrics:** deaths per minute < 3, pickup rate > 1 per life, step_reward consistently > 0

---

## Stage 3: Map Awareness

KaaN starts building a spatial model of q2dm1. He learns which corridors connect to which rooms, where the walls are, and stops getting stuck in corners. He knows there are items at specific locations and starts patrolling toward them.

**Signs of progress:**
- Rarely gets stuck in corners (stuck penalty near zero)
- Navigates between rooms (uses doors, ramps, lifts)
- Moves toward item spawn locations even when items aren't visible
- Uses jump pads and teleporters correctly
- Average speed stays high (> 200 units/s)

**Metrics:** stuck frames < 5%, map coverage (visits multiple rooms per life)

---

## Stage 4: Item Hunter

KaaN understands the item game. He knows where weapons, armor, and health spawn on q2dm1. He prioritizes picking up the rocket launcher, super shotgun, and mega health. He starts timing major items (mega health respawns every 20s).

**Signs of progress:**
- Picks up weapons within first 10 seconds of spawning
- Routes toward mega health and yellow armor
- Carries 2-3 weapons most of the time
- Switches weapons based on situation (not random)
- Ammo count stays above zero

**Metrics:** weapons owned > 2 average, armor > 30 average, weapon switches are contextual

---

## Stage 5: Combat Basics

KaaN learns to fight. He aims at the enemy, leads targets with projectiles, and chooses weapons based on distance. Close range = super shotgun. Mid range = rockets. Long range = railgun. He learns that shooting *at* the enemy is better than shooting *near* them.

**Signs of progress:**
- Tracks enemy with viewangles (aim follows enemy position)
- Hits enemy more than misses (accuracy > 20% with RL)
- Switches to appropriate weapon for range
- Stops shooting walls and floors
- Gets 1-2 frags per life

**Metrics:** frag/death ratio > 0.5, weapon accuracy > 15%, frags per episode > 1

---

## Stage 6: Tactical Movement

KaaN learns that movement *during* combat matters. He strafes to dodge incoming fire, uses cover, and controls engagement distance. He starts bunny-hopping to maintain speed between fights.

**Signs of progress:**
- Strafes during combat (sidemove active when enemy visible)
- Uses walls/corners as cover
- Maintains high speed while fighting
- Doesn't run in straight lines toward enemy
- Dodges rockets and grenades

**Metrics:** survival rate in combat > 40%, speed during combat > 250 units/s

---

## Stage 7: Strafe Jumping

KaaN discovers Quake physics. By jumping and strafing at specific angles, he can accelerate beyond normal running speed. He starts strafe-jumping through corridors, reaching items faster and making himself harder to hit.

**Signs of progress:**
- Accelerates above 400 units/s through strafe-jumping
- Chains multiple strafe jumps together
- Uses speed to control map faster than opponent
- Reaches item spawns before enemy

**Metrics:** average speed > 350 units/s, top speed > 600 units/s

---

## Stage 8: Map Control

KaaN plays strategically. He doesn't just collect items — he denies them to the opponent. He controls red armor timing, holds advantageous positions, and forces fights when he has superior stack (health + armor + weapons).

**Signs of progress:**
- Times mega health and yellow armor spawns
- Holds position near important items
- Engages when stacked, retreats when weak
- Denies enemy access to weapons
- Wins most 1v1 encounters

**Metrics:** frag/death ratio > 1.5, mega health control > 60%, wins matches

---

## Stage 9: Opponent Modeling

KaaN reads the enemy. He predicts where the opponent will go based on their current trajectory, health status, and available items. He sets traps, pre-aims at corners, and punishes predictable movement patterns.

**Signs of progress:**
- Pre-aims at corners where enemy will appear
- Predicts enemy routing (cuts them off at items)
- Adapts strategy based on opponent behavior
- Uses sound cues (opponent footsteps, weapon pickups)
- Punishes same escape routes

**Metrics:** frag/death ratio > 2.0, prediction accuracy visible in aim patterns

---

## Stage 10: Mastery

KaaN plays like a top competitive player. He combines all skills — perfect movement, item timing, weapon selection, aim, positioning, and opponent reading — into fluid gameplay. He beats skill-10 bots consistently and can compete with human players.

**Signs of progress:**
- Rocket jumps to reach positions faster
- Controls entire map cycle (all major items)
- Adapts to different opponent styles
- Wins against multiple bot skill levels
- Gameplay looks intentional and strategic to human observers

**Metrics:** wins > 80% against skill-10 bots, competitive with experienced human players

---

## The Dream: Stage 11

KaaN discovers techniques no human has used. Through millions of training episodes, he finds novel movement tricks, unexpected angles, and strategies that surprise even veteran Quake 2 players. He doesn't just learn to play — he innovates.
