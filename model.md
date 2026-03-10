# Simulate Trajectories

## Given

1. Tables $TAB = [T_1, \ldots, T_n]$.  
   Each $T_i$ has corner vertices in fixed counter-clockwise order, and **offset vertices** $[\hat\nu_0, \hat\nu_1, \ldots]$ — one per corner, pushed $d_{off} = 0.1$ m radially outward from the centroid.

2. Exhibits $EXH = [E_1, \ldots, E_m]$.  
   Each $E_j$ has coordinates $(x_{E_j}, y_{E_j})$, a parent table $T_{E_j}$, and a **normal angle** $\theta_{n,j}$ (outward normal of the table edge it faces).

3. For each exhibit $E_j$: a **neighbourhood set** $N_j \subseteq EXH$ within a fixed radius.

4. A **start set** $\mathcal{E}_{start} \subseteq EXH$ — must be non-empty; `MultiSimulate` raises `ValueError` immediately if empty.

5. A **start area** $[x_{min}^{start}, x_{max}^{start}] \times [y_{min}^{start}, y_{max}^{start}]$.

6. A **museum boundary** $\mathcal{B} = [x_{min}, x_{max}] \times [y_{min}, y_{max}]$.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $D(p, q)$ | Euclidean distance between $p$ and $q$ |
| $sdist(\nu, L)$ | Signed distance from $\nu$ to directed line $L$; positive = left (CCW side) |
| $distToTable(S, T)$ | Min distance from $S$ to boundary of $T$; $0$ if inside |
| $intersects(L, T)$ | True iff segment $L$ properly crosses any edge of $T$ |
| $vertices(T)$ | CCW-ordered corners of $T$ |
| $offset\_vertices(T)$ | Offset corners $\hat\nu_i = \nu_i + d_{off} \cdot \tfrac{\nu_i - \bar\nu}{|\nu_i - \bar\nu|}$ |
| $normal\_angle(E)$ | Outward normal of exhibit edge: $\text{atan2}(-e_x, e_y)$, $\vec{e} = \nu_{i+1} - \nu_i$ |
| $phase$ | Behavioural state: `WALK`, `APPROACH`, or `STOP` |
| $sr$ | Sampling rate (12 Hz) |
| $r_{stop}$ | Wander radius in `STOP` phase |
| $r_{coll}$ | Collision detection radius |
| $\alpha_{tol}$ | Angular tolerance for converging-pair detection |
| $\tau_E$ | Absolute dwell deadline [s] for the current exhibit |
| $\varepsilon$ | Small tolerance ($10^{-9}$) |

---

## Agent Types

$$\text{move\_type} \sim \begin{cases} \text{SLOW} & p = 0.40 \\ \text{FAST} & p = 0.60 \end{cases}$$

Affects `WALK` speed only.

---

## Distributions

**Walk speed:**

$$v \mid \texttt{WALK} = \begin{cases} \text{Beta}(2.0,\,10.1) \to [0.30,\,1.20]\,\text{m/s} & \text{SLOW} \\ \text{Beta}(2.0,\,5.0) \to [0.80,\,2.60]\,\text{m/s} & \text{FAST} \end{cases}$$

**Speed by phase (same for all agents):**

$$v \mid phase = \begin{cases} \text{Uniform}(0.3,\,0.7)\,\text{m/s} & \texttt{APPROACH} \\ \text{Uniform}(0.1,\,0.25)\,\text{m/s} & \texttt{STOP} \end{cases}$$

**Heading perturbation:** $\Delta\theta \mid phase \sim \mathcal{N}(0,\,\sigma_{\theta,phase})$

**Bernoulli gates:**

$$p_v \mid phase = \begin{cases} 0.75 & \texttt{WALK} \\ 0.30 & \texttt{APPROACH} \\ 0.66 & \texttt{STOP} \end{cases} \qquad p_\theta \mid phase = \begin{cases} 0.80 & \texttt{WALK} \\ 0.50 & \texttt{APPROACH} \\ 0.75 & \texttt{STOP} \end{cases}$$

**Dwell time:** $\Delta\tau \sim \text{Beta}(1.9,\,4.0) \to [11,\,180]\,\text{s}$ (mode $\approx 50\,\text{s}$)

**Next exhibit selection:**

$$E_{next} \sim \begin{cases} \text{Uniform}(\mathcal{E}_{start}) & \text{first exhibit} \\ \text{Uniform}(N_{E_{prev}} \setminus E_{visited}) & \text{unvisited neighbours exist} \\ \text{Uniform}(EXH \setminus E_{visited}) & \text{fallback} \end{cases}$$

---

## Clock Contract

The global clock $t$ is owned exclusively by `MultiSimulate`. Each tick, $t_{start}$ is snapshotted and passed to every agent.

`AdvanceOneTick(agent, t_start)` records its sample at $t_{end} = t_{start} + 1/sr$ and returns:

| Return | Meaning |
|--------|---------|
| $t_{end}$ | Normal move |
| $t_{start}$ (unchanged) | No move — see table below |

Three no-move causes all return $t_{start}$:

| Cause | Sample recorded at $t_{end}$? | Caller action |
|-------|-------------------------------|---------------|
| Done or no target | Freeze sample (WALK) | None |
| Dwell deadline: $t_{end} \geq \tau_E$ | STOP sample | Advance itinerary |
| Excursion limit exceeded | Already recorded before returning | Advance itinerary |

Caller detects "advance itinerary" with:

$$dwell\_expired = (new\_t < t_{start} + 1/sr - \varepsilon) \;\wedge\; \lnot\,done \;\wedge\; target \neq \text{None}$$

> **Critical:** `target ≠ None` is required. The excursion path keeps `target_exhibit` set and uses $\tau_E = t_{end}$ to signal expiry. Setting `target = None` (the old bug) made `dwell_expired` permanently False, freezing the agent.

---

## Algorithm

---

### def &nbsp;$Move(S, P, phase, t)$

One position-update step toward $P$ from $S$, sample recorded at timestamp $t$ (= $t_{end}$, passed by caller).

1. If $\text{Bernoulli}(p_v) = 1$: resample $v$
2. If $\text{Bernoulli}(p_\theta) = 1$: sample $\Delta\theta$; else $\Delta\theta := 0$
3. $\theta := \text{atan2}(P_y - S_y,\; P_x - S_x) + \Delta\theta$
4. $S' := S + \tfrac{v}{sr}(\cos\theta, \sin\theta)$; clamp to $\mathcal{B}$
5. Record $(t, S'_x, S'_y, phase)$

---

### def &nbsp;$ComputeObstacleWaypoints(S, T_{blocking}, L_{start}, L_{end})$

Pure geometry — returns the ordered offset-vertex waypoints around $T_{blocking}$. No side effects.

1. Classify each $\nu_i \in vertices(T)$ as left ($V^+$) or right ($V^-$) of $L_{start} \to L_{end}$
2. $V :=$ the smaller group (shorter arc); break ties by choosing the side whose nearest vertex to $S$ is closest
3. If $V = \emptyset$: return $[]$ (segment grazes vertex, no detour needed)
4. Start from $\nu_{start} := \arg\min_{V} D(\nu, S)$; walk $V$ in the direction that stays within $V$
5. Return offset vertices in walk order

---

### def &nbsp;$AdvanceOneTick(agent, t_{start})$

Advance one agent by one tick. Returns $t_{end}$ or $t_{start}$ (no-move signal).

---

**(a) Done or no target:**

```
record freeze sample at t_end (WALK)
return t_start
```

**(b) Obstacle navigation in progress:**

```
if nav_waypoints ≠ None:

    if t_end ≥ τ_E:                        ← CHECK EXPIRY FIRST
        record STOP sample at t_end
        clear nav_waypoints
        return t_start                     ← expiry signal

    old_pos := agent.pos
    Move(agent.pos, nav_waypoints[wp_idx], WALK, t_end)

    vec_before := wp - old_pos
    vec_after  := wp - agent.pos
    reached := D(agent.pos, wp) < 0.05
               OR dot(vec_before, vec_after) ≤ 0   ← overshoot check

    if reached:
        wp_idx += 1
        if wp_idx ≥ len(nav_waypoints): clear nav_waypoints

    return t_end
```

> **Overshoot check:** at FAST speed ($2.60\,\text{m/s}$, $sr=12\,\text{Hz}$), one step is $0.217\,\text{m} \gg 0.05\,\text{m}$. The agent can jump completely past a waypoint. After overshoot, the post-step distance to the waypoint is large and the proximity check fails silently — the agent circles forever. The dot-product detects the crossing: if $\vec{v}_{before} \cdot \vec{v}_{after} \leq 0$, the agent crossed the perpendicular plane through the waypoint.

**(c) Dwell timeout:**

```
if t_end ≥ τ_E:
    record STOP sample at t_end           ← prevents 2×dt gap at transitions
    return t_start                        ← expiry signal
```

**(d) Normal movement — unobstructed:**

Compute $P_w$ (movement target) by phase:

- $phase \in \{\texttt{WALK}, \texttt{APPROACH}\}$: $P_w := E_{pos}$
- $phase = \texttt{STOP}$ and $D > r_{stop}$: $P_w := E_{pos}$ (drift back)
- $phase = \texttt{STOP}$ and $D \leq r_{stop}$: disk sampling

**Disk sampling:**

$$\varphi \sim \text{Uniform}(\theta_n - \pi/2,\;\theta_n + \pi/2), \quad \rho \sim U(0,1), \quad k \sim \begin{cases} -0.80 & \text{prob } w_1 \\ 4.0 & \text{else} \end{cases}$$
$$r := r_{stop} \cdot \rho^{1/(k+1)}, \qquad P_w := E_{pos} + r(\cos\varphi, \sin\varphi)$$

Execute $Move(S, P_w, phase, t_{end})$.

**(d) Normal movement — blocked path:**

```
blocking := {T ∈ TAB | intersects(S → E_pos, T)}
T_nearest := argmin distToTable(S, T)

if distToTable(S, T_nearest) < 0.05:
    Move(S, E_pos, phase, t_end)          ← corner-clip: step through
else:
    waypoints := ComputeObstacleWaypoints(S, T_nearest, S, E_pos)
    if waypoints ≠ []:
        store on agent; take first step; apply reached check
    else:
        Move(S, E_pos, phase, t_end)      ← grazing: no detour
return t_end
```

---

**(e) Phase transitions and excursion tracking (after move):**

$D := D(agent.pos, E_{pos})$

```
if phase = WALK  AND D < 2·r_stop:  phase := APPROACH
if phase = APPROACH AND D < r_stop: phase := STOP;  τ_E := t_end + Δτ

# Excursion tracking (STOP only)
if phase = STOP AND D > r_stop AND NOT excursing:
    excursing := True;  excurse_start := t_end

if excursing AND D ≤ r_stop:
    excurse += t_end - excurse_start;  excursing := False

if excursing AND excurse + (t_end - excurse_start) > 15 s:
    τ_E := t_end          ← signal expiry; keep target_exhibit set
    clear nav_waypoints
    return t_start        ← dwell_expired will fire for caller
```

> **Why $\tau_E := t_{end}$ and not `target := None`:** see Clock Contract. Keeping `target` set allows the `dwell_expired` check to fire, triggering a normal itinerary advance for solo agents and a Phase 1/2 update for groups.

**return** $t_{end}$

---

### def &nbsp;$AdvanceExhibit(agent, E_{next})$

```
clear nav_waypoints, nav_wp_idx

if E_next = None:
    done := True;  target := None;  return

target  := E_next
visited ∪= {E_next}     ← commitment at planning time, not arrival
phase   := WALK
τ_E     := 9999
excurse := 0;  excursing := False
```

---

### def &nbsp;$BuildGroupItinerary(group, E_{first}, rng)$

Pre-build the shared exhibit sequence before the simulation starts. All members follow it in lockstep.

1. $itinerary := [E_{first}]$; $visited := \{E_{first}\}$; $E_{prev} := E_{first}$
2. While $|itinerary| < K$ and $|itinerary| < |EXH|$: select next via neighbourhood-then-fallback rule; append
3. $group.dwell\_ends := [9999, \ldots]$ *(filled at runtime)*

---

### Collision Avoidance

Agents $a_i, a_j$ form a **converging pair** iff:

$$D(\text{pos}_i, \text{pos}_j) < r_{coll} \quad \text{and} \quad \alpha_{tol} \leq |\theta_i - \theta_j|_{[0,\pi]} \leq \pi - \alpha_{tol}$$

$\mathcal{H}$ = all agents in at least one converging pair. $\mathcal{M}$ = active $\setminus\, \mathcal{H}$.

$\mathcal{M}$ moves first. Then $\mathcal{H}$ is processed in a random permutation — each agent records exactly one sample per tick.

---

### Group Synchronisation

**When:** end of each tick, after all agents have moved.

$group.dwell\_ends[idx]$ = shared deadline for the current exhibit. Starts at 9999, can only decrease.

**Phase 1 — minimum-intersection deadline** *(every tick until Phase 2)*

$$stop\_members := \{m \mid m.target = E_{current} \wedge m.phase = \texttt{STOP} \wedge \tau_{E,m} < 9999\}$$

If $stop\_members \neq \emptyset$ and $t_{end} < dwell\_ends[idx]$:

$$d_{new} := \min_m \tau_{E,m}; \quad \text{if } d_{new} < dwell\_ends[idx]: \text{ update it and set all } \tau_{E,m} := d_{new}$$

> Excursion aborts feed into this naturally: if a member's excursion fires and sets $\tau_E = t_{end}$, Phase 1 pulls the group deadline to the current tick, causing Phase 2 to fire immediately.

**Phase 2 — synchronised advance** *(fires once per exhibit)*

If $dwell\_ends[idx] < 9999$ and $t_{end} \geq dwell\_ends[idx]$:

1. $group.current\_idx += 1$
2. $E_{next} :=$ next in itinerary (or None)
3. For each member: $E_{prev} := E_{current}$; $AdvanceExhibit(m, E_{next})$

---

### main &nbsp;$MultiSimulate(n_{solo}, n_{groups}, group\_size\_range)$

**Guard:** if $\mathcal{E}_{start} = \emptyset$: raise `ValueError`.

**Initialise:**

- Solo: sample move\_type, pos, $E_{first}$; call $Initialise$
- Groups: sample size $s$; jitter members around shared base pos; call $Initialise$ each; $BuildGroupItinerary$; set `member.group`

**Main loop** ($\leq \lfloor t_{max} \cdot sr \rfloor + 1$ ticks):

```
t := 0
for each tick:
    if all done: break

    identify ℋ (converging pairs), ℳ (rest of active agents)

    for agent in ℳ:
        new_t := AdvanceOneTick(agent, t)
        if dwell_expired(new_t, t, agent) AND agent.group = None:
            agent.E_prev := agent.target
            AdvanceExhibit(agent, next_exhibit(agent))

    for agent in ℋ (random order):   ← same dwell check
        new_t := AdvanceOneTick(agent, t)
        if dwell_expired(new_t, t, agent) AND agent.group = None:
            agent.E_prev := agent.target
            AdvanceExhibit(agent, next_exhibit(agent))

    for each group:
        Phase 1 → Phase 2

    t += 1/sr
    if t ≥ t_max: break
```

> **Group members skip the solo dwell branch.** The `agent.group = None` guard is essential: group dwell expiry is handled exclusively by Phase 2, which redirects all members simultaneously. If a group member's individual expiry signal fires (e.g., from an excursion abort), the caller ignores it — Phase 1 picks up the new $\tau_E$ and Phase 2 handles the redirect.

---

**Termination:**

| Condition | Scope |
|-----------|-------|
| All agents done | Global |
| $t \geq t_{max}$ | Global |
| $\lvert visited \rvert \geq K$ | Per agent |
| $EXH \setminus visited = \emptyset$ | Per agent |

---

## Bugs Found and Fixed

### Bug 1 — Excursion abort froze solo agents permanently

**Root cause:** The excursion branch set `target_exhibit = None` then returned $t_{end}$ (a normal move return). `dwell_expired` requires `target ≠ None` → permanently False → agent froze.

**Fix:** Keep `target_exhibit` set. Set $\tau_E := t_{end}$ and return $t_{start}$. The caller's `dwell_expired` fires exactly once, triggering a normal itinerary advance. Group Phase 1 picks up the early $\tau_E$ and Phase 2 fires the same tick.

---

### Bug 2 — Nav branch ignored $\tau_E$ while mid-detour

**Root cause:** The `if nav_waypoints is not None` branch stepped unconditionally and returned $t_{end}$, skipping the `τ_E` check. A solo agent whose deadline expired mid-obstacle-detour kept navigating forever.

**Fix:** Check `t_end ≥ τ_E` at the top of the nav branch before any step. If expired: record STOP sample, clear nav state, return $t_{start}$.

---

### Bug 3 — Waypoint overshoot left nav state permanently active

**Root cause:** The waypoint-reached check was `D(pos, wp) < 0.05 m`. At FAST speed ($0.217\,\text{m/step}$), the agent jumps past a close waypoint; post-step distance is large; check never fires; agent loops.

**Fix:** Add dot-product crossing check. Record `old_pos` before the step. After the step:

$$passed := (wp - old\_pos) \cdot (wp - new\_pos) \leq 0$$

Waypoint is marked reached if `D < 0.05 OR passed`.

---

### Bug 4 — Empty `start_exhibits` crashed with opaque NumPy error

**Root cause:** `rng.integers(0, 0)` raised `ValueError: high <= 0` with no context.

**Fix:** Explicit guard at the start of `_pick_first_exhibit` raising a descriptive `ValueError`.

---

## Output

| Column | Type | Description |
|--------|------|-------------|
| `x` | float | Position x [m] |
| `y` | float | Position y [m] |
| `timestamp` | float | Global clock [s] at sample time |
| `person_id` | int | Agent index (solo first, then group members in order) |
| `phase` | string | `WALK`, `APPROACH`, or `STOP` |
