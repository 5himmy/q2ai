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

**Map: q2dm1 "The Edge"** — Classic duel map. Two main levels connected by ramps, lifts, and a teleporter. No red armor on this map. Key positions: RL room, mega health area, yellow armor spots.

**Match Format:**
- 10-minute timelimit
- 1 minute overtime if tied
- Both players must `/ready` to start
- Warmup phase before match (all weapons available, no score)

**Movement Physics:**
- Base run speed: 300 units/second
- Strafe jumping: accelerate beyond 300 by jumping + strafing at angles (Quake physics exploit)
- Rocket jumping: use rocket splash damage to launch yourself to higher positions
- Crouch: slower movement, rarely useful in duel

---

## Reward Rules (what KaaN learns from)

These are the signals that shape KaaN's behavior during PPO training:

| Signal | Reward | Purpose |
|--------|--------|---------|
| Kill enemy | **+1.0** | Primary objective |
| Die | **-1.0** | Primary penalty |
| Take damage | **-damage/100** | Learn to dodge |
| Get stuck (speed < 20) | -0.01 | Don't stand still |
| Circling (< 128 units in 3s) | **-0.15** | Must explore, not circle in place |
| Move backward | -0.01 | No backward circling |
| Crouch (per frame) | **-0.10** | Crouching is never good in duel |
| Shoot at visible enemy | +0.02 | Reward aimed fire |
| Shoot blindly (no enemy) | **-0.15** | Very harsh penalty for spray-and-pray |
| Using blaster with better weapon | **-0.02** | Switch to a real weapon! |
| Approach visible enemy | **+0.01** | Go fight, don't run away |
| Retreat from enemy | -0.005 | Mild penalty (sometimes retreat is smart) |
| Pick up health/armor | +0.2 | Learn to collect resources |
| Pick up weapon | +0.3 | Learn to arm yourself |
| Approach nearest item | +0.003/unit | Subtle pull toward items |

**Design philosophy:** No free rewards for existing. No survival bonus, no undirected movement reward. The only way to earn positive reward is fighting (frags, aimed shots, approaching enemy) or picking up items. Circling, crouching, blind-firing, and ignoring collected weapons are heavily punished.

---

## Stage 1: Newborn (current stage)

KaaN spawns into q2dm1 with David's muscle memory from behavior cloning. He moves forward, shoots when he sees someone, and sometimes jumps. But he has no understanding of *why* — he's mimicking without comprehension.

**Signs of progress:**
- Doesn't stand still (moves forward most of the time)
- Stops shooting when no enemy is visible
- Survives more than 5 seconds per life
- Occasionally picks up items by accident

**Metrics:** step_reward > 0, episode length > 50 frames, blind-fire rate < 30%

---

## Stage 2: Survivor

KaaN learns that being alive is good and dying is bad. He starts avoiding damage — backing away from rockets, not running off ledges. He discovers that health and armor pickups make him harder to kill.

**Signs of progress:**
- Average episode length > 200 frames (~20 seconds)
- Picks up health/armor when nearby (not by accident)
- Doesn't suicide with own rockets
- Moves away from danger (negative health delta = retreat)
- Frags are still mostly lucky

**Metrics:** deaths per minute < 3, pickup rate > 1 per life

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
