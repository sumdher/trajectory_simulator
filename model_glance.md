# Museum Simulator — High-Level Pseudocode

---

## MultiSimulate(n_solo, n_groups)

```
assert start_exhibits is not empty

for each solo agent:
    pos   ← random point in start area
    first ← random exhibit from start_exhibits
    Initialise(agent, pos, first)

for each group:
    first     ← random exhibit from start_exhibits
    itinerary ← BuildItinerary(first)       # planned up-front, shared by all members
    for each member:
        Initialise(member, pos ≈ group_base, first)

t := 0
repeat until all done or t ≥ t_max:

    H ← agents in converging collision pairs
    M ← active agents \ H

    for agent in M:           Tick(agent, t)
    for agent in H (random order): Tick(agent, t)   # same clock, random release

    for each group:
        UpdateGroupDeadline()   # Phase 1: deadline = min(tau_E) over stopped members
        if deadline reached:
            AdvanceAllMembers() # Phase 2: everyone redirects simultaneously

    t += 1/sr
```

---

## Initialise(agent, pos, first_exhibit)

```
agent.pos     := pos
agent.visited := {first_exhibit}   # commitment set, not arrival record
agent.E_prev  := first_exhibit
AdvanceExhibit(agent, first_exhibit)
record sample at t = 0
```

---

## BuildItinerary(first_exhibit)   [groups only]

```
itinerary := [first_exhibit]
visited   := {first_exhibit}

while len(itinerary) < K:
    next := pick from neighbourhood of last exhibit, excluding visited
            (fallback: any unvisited exhibit in the museum)
    itinerary.append(next)
    visited.add(next)

return itinerary
```

---

## Tick(agent, t)   →   returns t_end or t (no-move signal)

```
t_end := t + 1/sr

if agent.done or agent.target is None:
    record freeze sample at t_end
    return t                            # no-move

if navigating around obstacle:
    if t_end ≥ tau_E:
        record STOP sample at t_end
        clear nav state
        return t                        # expiry signal

    step one waypoint forward
    if waypoint reached (by proximity OR dot-product overshoot check):
        advance to next waypoint (clear nav if last)
    return t_end

if t_end ≥ tau_E:
    record STOP sample at t_end
    return t                            # expiry signal

# — normal movement —

if path to target is blocked by a table:
    compute waypoints around table (shorter side)
    store on agent; step toward first waypoint
    return t_end

move toward target (or wander in place if STOP and inside r_stop)
record sample at t_end

# phase transitions
WALK     → APPROACH  when distance < 2 * r_stop
APPROACH → STOP      when distance < r_stop  →  draw dwell duration, set tau_E

# excursion check (STOP only)
if wandered outside r_stop for > 15 s total:
    tau_E := t_end          # treat as expired; keep target set
    return t                # expiry signal — caller advances itinerary

return t_end
```

---

## After Tick — solo itinerary advance

```
dwell_expired := (new_t < t + 1/sr)  and  not done  and  target is not None

if dwell_expired and agent is solo:
    E_prev := agent.target
    AdvanceExhibit(agent, NextExhibit(agent))
```

---

## AdvanceExhibit(agent, next)

```
clear nav state

if next is None:
    agent.done := True
    return

agent.target  := next
agent.visited ∪= {next}
agent.phase   := WALK
agent.tau_E   := ∞
reset excursion counters
```

---

## NextExhibit(agent)   [solo only]

```
if len(visited) ≥ K:  return None

candidates := neighbourhood(E_prev) \ visited
if candidates is empty:
    candidates := all exhibits \ visited
if candidates is empty:  return None

return random choice from candidates
```

---

## Group sync — end of each tick

```
# Phase 1: pull shared deadline toward the earliest stopped member
for each stopped member at current exhibit with a drawn dwell:
    group.deadline := min(group.deadline, member.tau_E)
    broadcast: all members.tau_E := group.deadline

# Phase 2: advance everyone when deadline passes
if group.deadline reached:
    next := group.itinerary[++idx]
    for each member:
        E_prev := current exhibit
        AdvanceExhibit(member, next)
```
