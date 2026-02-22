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

**Training Map: hop** — Small map with frequent encounters. KaaN spawns with all weapons + 100 armor. Scripted bot (skill 1, curriculum learning) also on the map.

**Target Map: q2dm1 "The Edge"** — Classic duel map. Two main levels connected by ramps, lifts, and a teleporter. No red armor on this map. Key positions: RL room, mega health area, yellow armor spots. (Used after KaaN learns combat basics on 2box6.)

**Match Format:**
- 3-minute timelimit (short rounds for fast iteration)
- No overtime (g_tie_mode 0)
- 1-second countdown
- Auto-ready via bots
- All weapons given on spawn (2box6 has no pickups)

**Movement Physics:**
- Base run speed: 300 units/second
- Strafe jumping: accelerate beyond 300 by jumping + strafing at angles (Quake physics exploit)
- Rocket jumping: use rocket splash damage to launch yourself to higher positions
- Crouch: slower movement, rarely useful in duel

---

## Reward Rules V2.0 (what KaaN learns from)

Simplified to core combat + critical anti-suicide signals. Fewer signals = lower value loss = better learning.

| Signal | Reward | Purpose |
|--------|--------|---------|
| Kill enemy | +1.0 | Primary objective |
| Die | -1.0 | Primary penalty |
| Take damage | -damage/100 | Learn to dodge |
| Deal damage | +damage/100 | Dense combat signal (via STAT_FLASHES, FOV-gated) |
| Self-splash penalty | -0.15 | Don't shoot rockets into walls at close range |
| Using blaster with better weapon | -0.02 | Switch to a real weapon! |
| Get stuck (speed < 20) | -0.01 | Don't stand still |
| Pick up health/armor | +0.1 | Learn to collect resources |
| Pick up weapon | +0.15 | Learn to arm yourself |

**V2.0 changes:** Stripped from 15 signals to 9. Removed aim tracking, precision bonus, shooting reward, approach reward, item proximity, backward penalty, crouch penalty, circling penalty. Value loss dropped from ~47 to ~0.4 immediately — the critic can now actually learn returns.

**Design philosophy:** Let the core objective (frags, deaths, damage) drive learning. Only add anti-suicide signals for critical bad behaviors (rocket self-splash, blaster hoarding). Aim tracking and movement shaping can be added back once KaaN learns basic combat.

---

## Training Infrastructure

- **KaaN V2.7** — 250 parallel instances on 12-core/16GB/RTX 3070
- **Map:** hop (small map, frequent encounters)
- **Architecture:** V2.6 Grouped Encoder + Entity Attention + Fourier Position + LSTM (1.2M params)
- **Separate Actor/Critic LSTMs** — no shared trunk (V1.2 fix: value gradients were destroying policy)
- **Continuous episodes** — no resets on death. Auto-respawn in C, continuous observation stream.
- **~1,300 steps/sec** throughput, ~50s per iteration (64,000 steps/iter at 250 instances)
- **Behavior cloning warm start** from 53 human demos on q2dm1 (model_bc.pt)
- **Reward scaling** — 5x multiplier (REWARD_SCALE = 5.0)
- **Observation normalization** — running mean/std (Welford's algorithm), clips to [-5, 5]
- **PPO hyperparameters:** lr=1e-4, epochs=4, value_coef=1.0, target_kl=0.05, entropy_coef=0.005
- **Rollout:** 256 steps/env, seq_len=32 (LSTM chunks), batch_chunks=16
- **KL stability:** avg_kl=0.005-0.007, target_kl early stopping per mini-batch
- **Curriculum learning:** bot skill 1 (graduating to higher skills as K/D improves)
- **Damage dealt tracking** — opentdm exposes per-frame damage via STAT_FLASHES (self-damage filtered)
- Batched GPU inference, threaded parallel I/O, semaphore resync between PPO updates
- Real frag/death tracking from game state (not episode counters)
- CSV logs: step_reward, frags, deaths per iteration (not just episode-level)

---

## Stage 1: Newborn (current stage)

KaaN trains on hop — a small map with frequent encounters. He faces a skill-1 scripted bot (curriculum learning — starting easy). V2.7 uses simplified rewards (9 signals, down from 15) so the value function can actually learn.

**Current observations (V2.7, hop, 250 instances, bot skill 1):**
- step_reward: -0.006 to -0.004 (improving with simplified rewards)
- 120-170 deaths per iteration (250 instances)
- 9-12 frags per iteration (consistent combat engagement)
- Value loss: 0.3-0.5 (was 47 with old rewards — massive improvement)
- KL: 0.006-0.007 (stable)
- Entropy: 8.3-8.6
- Deaths trending down: 235 → 126 over 30 iterations

**Signs of progress:**
- Gets frags consistently (9-12 per iteration)
- Deaths decreasing over training
- Value function actually learning (loss 0.3 vs 47 before)
- Doesn't self-kill with rockets as often (self-splash penalty working)
- Switches away from blaster

**Metrics:** step_reward > 0.01, frags > 20 per iteration, frag/death ratio > 0.1

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
