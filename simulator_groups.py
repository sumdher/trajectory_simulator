"""
museum_simulator.py
===================
Simulates visitor trajectories through a museum with table obstacles.

Architecture
------------
AgentMoveType   : SLOW / FAST — controls which speed Beta distribution is used
AgentState      : all per-agent mutable state (pos, v, phase, traj, rng …)
Group           : 1–5 agents sharing a common exhibit itinerary
MultiSimulator  : owns the single global clock; ticks all agents in lockstep;
                  handles collision avoidance and group synchronisation
Simulator       : thin single-agent wrapper (backwards-compatible)

Clock contract
--------------
The global clock t is owned exclusively by MultiSimulator.
advance_one_tick(t) receives t_start (the clock value at the START of the
current tick) and records samples at t_end = t_start + 1/sr.

Return value of advance_one_tick:
  t_end = t_start + 1/sr   → agent moved normally this tick
  t_start (unchanged)       → no movement this tick; one of three causes:
                               (a) agent is done or has no target
                                   → freeze sample recorded at t_end
                               (b) dwell deadline reached
                                   → STOP sample recorded at t_end; caller
                                      must call advance_exhibit
                               (c) excursion limit exceeded
                                   → sample already recorded; tau_E set to
                                      t_end so caller detects expiry

The caller detects dwell expiry / excursion abandon with:
    dwell_expired = (new_t < t_start + 1/sr - ε
                     and not agent.done
                     and agent.target_exhibit is not None)

Algorithm reference
-------------------
Phases     : WALK → APPROACH → STOP (per exhibit)
Obstacles  : rectangular / polygonal tables; navigated via waypoint state
             stored on AgentState — one sub-step per outer tick (no inner loop)
Sampling   : UWB-style at sr = 12 Hz
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import beta as _scipy_beta


# ══════════════════════════════════════════════════════════════════════════════
#  Phase constants & probability tables
# ══════════════════════════════════════════════════════════════════════════════

WALK = "WALK"
APPROACH = "APPROACH"
STOP = "STOP"

# Bernoulli gate probabilities for speed & heading updates
_P_V: Dict[str, float] = {WALK: 0.750, APPROACH: 0.30, STOP: 0.66}
_P_θ: Dict[str, float] = {WALK: 0.80, APPROACH: 0.50, STOP: 0.75}


# ══════════════════════════════════════════════════════════════════════════════
#  Agent move-type
# ══════════════════════════════════════════════════════════════════════════════


class AgentMoveType(Enum):
    SLOW = "SLOW"
    FAST = "FAST"


_BETA_WALK_SLOW = _scipy_beta(a=2.0, b=10.1)  # scaled to [0.30, 1.20] m/s
_BETA_WALK_FAST = _scipy_beta(a=2.0, b=5.0)  # scaled to [0.80, 2.60] m/s
_BETA_DWELL = _scipy_beta(a=1.9, b=4.0)  # dwell time  (mode ≈ 50 s)


def _rescale(u: float, lo: float, hi: float) -> float:
    return lo + u * (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
#  Domain objects
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Table:
    table_id: int
    vertices: np.ndarray
    offset_vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        centroid = self.vertices.mean(axis=0)
        offsets = []
        for v in self.vertices:
            direction = v - centroid
            norm = float(np.linalg.norm(direction))
            offsets.append(v + 0.1 * direction / norm if norm > 1e-12 else v.copy())
        self.offset_vertices = np.array(offsets, dtype=np.float64)

    def __hash__(self) -> int:
        return hash(self.table_id)

    def __eq__(self, o) -> bool:
        return isinstance(o, Table) and self.table_id == o.table_id

    def __repr__(self) -> str:
        return f"Table(id={self.table_id}, n_verts={len(self.vertices)})"


@dataclass
class Exhibit:
    exhibit_id: int
    x: float
    y: float
    table: Table
    normal_angle: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        return hash(self.exhibit_id)

    def __eq__(self, o) -> bool:
        return isinstance(o, Exhibit) and self.exhibit_id == o.exhibit_id

    def __repr__(self) -> str:
        return f"E{self.exhibit_id}({self.x:.2f},{self.y:.2f})"


@dataclass
class MuseumConfig:
    # ── Layout ──────────────────────────────────────────────
    tables: List[Table]
    exhibits: List[Exhibit]
    neighbourhoods: Dict[int, List[Exhibit]]
    start_exhibits: List[Exhibit]
    start_area: Tuple[float, float, float, float]
    boundary: Tuple[float, float, float, float]
    # ── Physics ─────────────────────────────────────────────
    sr: float = 12.0
    r_stop: float = 0.50
    r_collision: float = 0.40
    collision_angle_tol: float = math.radians(30.0)
    sigma_theta_walk: float = 0.80
    sigma_theta_approach: float = 0.50
    sigma_theta_stop: float = 0.656
    w1: float = 0.5
    # ── Termination ─────────────────────────────────────────
    K: int = 10
    t_max: float = 3_600.0


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    phase: str


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry utilities
# ══════════════════════════════════════════════════════════════════════════════


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _signed_dist(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-12:
        return 0.0
    return float((ab[0] * (v[1] - a[1]) - ab[1] * (v[0] - a[0])) / denom)


def _segments_properly_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> bool:
    d1 = p2 - p1
    d2 = p4 - p3
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-12:
        return False
    t = float((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = float((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    e = 1e-9
    return e < t < 1.0 - e and e < u < 1.0 - e


def _segment_intersects_table(S: np.ndarray, E: np.ndarray, table: Table) -> bool:
    verts = table.vertices
    n = len(verts)
    return any(
        _segments_properly_intersect(S, E, verts[i], verts[(i + 1) % n])
        for i in range(n)
    )


def _clamp_to_boundary(
    pos: np.ndarray, bnd: Tuple[float, float, float, float]
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bnd
    return np.array(
        [float(np.clip(pos[0], xmin, xmax)), float(np.clip(pos[1], ymin, ymax))]
    )


def _headings_converging(theta_i: float, theta_j: float, tol: float) -> bool:
    delta = abs(theta_i - theta_j) % (2 * math.pi)
    if delta > math.pi:
        delta = 2 * math.pi - delta
    return tol <= delta <= math.pi - tol


def _dist_point_to_table(px: float, py: float, table: Table) -> float:
    verts = table.vertices
    n = len(verts)
    min_d = math.inf
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        ab = b - a
        t = float(np.dot(np.array([px, py]) - a, ab) / (np.dot(ab, ab) + 1e-30))
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        min_d = min(min_d, math.hypot(px - closest[0], py - closest[1]))
    inside = all(
        float(
            (verts[(i + 1) % n][0] - verts[i][0]) * (py - verts[i][1])
            - (verts[(i + 1) % n][1] - verts[i][1]) * (px - verts[i][0])
        )
        >= -1e-9
        for i in range(n)
    )
    return 0.0 if inside else min_d


def _edge_normal_angle(px: float, py: float, table: Table) -> float:
    verts = table.vertices
    n = len(verts)
    best_i = 0
    best_d = math.inf
    p = np.array([px, py], dtype=np.float64)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        ab = b - a
        t = float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-30))
        t = max(0.0, min(1.0, t))
        d = float(np.linalg.norm(p - (a + t * ab)))
        if d < best_d:
            best_d = d
            best_i = i
    e = verts[(best_i + 1) % n] - verts[best_i]
    return math.atan2(-e[0], e[1])


# ══════════════════════════════════════════════════════════════════════════════
#  AgentState
# ══════════════════════════════════════════════════════════════════════════════


class AgentState:
    """
    All per-agent mutable state.

    Clock contract
    --------------
    advance_one_tick(t_start) records its sample at t_end = t_start + 1/sr.
    Returns t_end on normal move; returns t_start to signal "no move this tick"
    (done/no-target, dwell expired, or excursion limit exceeded).

    visited semantics
    -----------------
    `visited` is a planning EXCLUSION set — an exhibit is added the moment we
    commit to going there (not on arrival) to prevent re-selection.

    Obstacle navigation
    -------------------
    Waypoints are computed once when a blocked path is first detected and stored
    in _nav_waypoints. advance_one_tick advances exactly one step per call,
    keeping all agents on the same global clock.

    Excursion abandon
    -----------------
    If a STOP-phase agent drifts > 15 s outside r_stop, the exhibit is
    abandoned by setting tau_E = t_end and returning t_start (expiry signal).
    The caller then calls advance_exhibit to pick the next exhibit.
    This keeps target_exhibit valid so the dwell_expired check succeeds.
    """

    def __init__(
        self,
        agent_id: int,
        cfg: MuseumConfig,
        move_type: AgentMoveType = AgentMoveType.FAST,
        seed: Optional[int] = None,
    ) -> None:
        self.agent_id = agent_id
        self.cfg = cfg
        self.move_type = move_type
        self.rng = np.random.default_rng(seed)

        self.group: Optional[object] = None  # set to Group by MultiSimulator

        self.pos: np.ndarray = np.zeros(2)
        self.v: float = 0.0
        self.traj: List[TrajectoryPoint] = []

        self.target_exhibit: Optional[Exhibit] = None
        self.phase: str = WALK
        self.tau_E: float = 9_999.0
        self.excurse: float = 0.0
        self.excurse_start: float = 0.0
        self.excursing: bool = False

        self.visited: Set[Exhibit] = set()
        self.E_prev: Optional[Exhibit] = None
        self.done: bool = False

        self._nav_waypoints: Optional[List[np.ndarray]] = None
        self._nav_wp_idx: int = 0

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def sample_speed(self, phase: str) -> float:
        if phase == WALK:
            if self.move_type == AgentMoveType.SLOW:
                return _rescale(
                    float(_BETA_WALK_SLOW.rvs(random_state=self.rng)), 0.30, 1.20
                )
            else:
                return _rescale(
                    float(_BETA_WALK_FAST.rvs(random_state=self.rng)), 0.80, 2.60
                )
        if phase == APPROACH:
            return float(self.rng.uniform(0.3, 0.7))
        return float(self.rng.uniform(0.1, 0.25))

    def sample_dtheta(self, phase: str) -> float:
        sigma = {
            WALK: self.cfg.sigma_theta_walk,
            APPROACH: self.cfg.sigma_theta_approach,
            STOP: self.cfg.sigma_theta_stop,
        }[phase]
        return float(self.rng.normal(0.0, sigma))

    def sample_dwell(self) -> float:
        return _rescale(float(_BETA_DWELL.rvs(random_state=self.rng)), 11.0, 180.0)

    def current_theta(self) -> float:
        if self.target_exhibit is None:
            return 0.0
        diff = self.target_exhibit.pos - self.pos
        return math.atan2(diff[1], diff[0])

    # ── _step_move ────────────────────────────────────────────────────────────

    def _step_move(self, target: np.ndarray, phase: str, t: float) -> None:
        """One movement step, sample recorded at t (= t_start + 1/sr)."""
        if self.rng.random() < _P_V[phase]:
            self.v = self.sample_speed(phase)
        dtheta = self.sample_dtheta(phase) if self.rng.random() < _P_θ[phase] else 0.0
        diff = target - self.pos
        theta = math.atan2(diff[1], diff[0]) + dtheta
        step = self.v / self.cfg.sr
        self.pos = _clamp_to_boundary(
            self.pos + step * np.array([math.cos(theta), math.sin(theta)]),
            self.cfg.boundary,
        )
        self.traj.append(
            TrajectoryPoint(t, float(self.pos[0]), float(self.pos[1]), phase)
        )

    # ── _compute_obstacle_waypoints ───────────────────────────────────────────

    def _compute_obstacle_waypoints(
        self,
        blocking_table: Table,
        L_start: np.ndarray,
        L_end: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Pure geometry: return the ordered offset-vertex waypoints that navigate
        around blocking_table from the current position toward L_end.
        Returns [] if no detour is needed (segment grazes a vertex exactly).
        """
        verts = blocking_table.vertices
        n = len(verts)
        EPS = 1e-9

        sdists = [_signed_dist(verts[i], L_start, L_end) for i in range(n)]
        pos_side = [i for i in range(n) if sdists[i] > EPS]
        neg_side = [i for i in range(n) if sdists[i] < -EPS]

        if len(pos_side) < len(neg_side):
            V_idx = pos_side
        elif len(neg_side) < len(pos_side):
            V_idx = neg_side
        else:
            if not pos_side or not neg_side:
                return []
            np_i = min(pos_side, key=lambda i: _dist(self.pos, verts[i]))
            nn_i = min(neg_side, key=lambda i: _dist(self.pos, verts[i]))
            V_idx = (
                pos_side
                if _dist(self.pos, verts[np_i]) <= _dist(self.pos, verts[nn_i])
                else neg_side
            )

        if not V_idx:
            return []

        V_set = set(V_idx)
        start_idx = min(V_set, key=lambda i: _dist(self.pos, verts[i]))
        delta = +1 if (start_idx + 1) % n in V_set else -1

        waypoints: List[np.ndarray] = []
        cur = start_idx
        for _ in range(n + 1):
            if cur not in V_set:
                break
            waypoints.append(blocking_table.offset_vertices[cur].copy())
            cur = (cur + delta) % n
        return waypoints

    # ── advance_one_tick ──────────────────────────────────────────────────────

    def advance_one_tick(self, t: float) -> float:
        """
        Advance the agent by exactly one global tick (1/sr seconds).

        Parameters
        ----------
        t : float   global clock at the START of this tick (t_start)

        Returns
        -------
        t + 1/sr    normal move; sample recorded
        t           no move this tick; one of:
                      • done / no target  → freeze sample recorded at t_end
                      • dwell expired     → STOP sample recorded at t_end
                      • excursion limit   → sample already recorded; tau_E = t_end

        Caller detects expiry / abandon with:
            dwell_expired = (new_t < t + 1/sr - ε
                             and not agent.done
                             and agent.target_exhibit is not None)
        """
        dt = 1.0 / self.cfg.sr
        t_end = t + dt

        # ── (a) Done or no target ──────────────────────────────
        if self.done or self.target_exhibit is None:
            self.traj.append(
                TrajectoryPoint(t_end, float(self.pos[0]), float(self.pos[1]), WALK)
            )
            return t

        # ── (b) Continue obstacle navigation ──────────────────
        if self._nav_waypoints is not None:
            # FIX BUG 2: check dwell expiry BEFORE taking a nav step.
            # Without this, a solo agent whose tau_E expired while mid-detour
            # would keep navigating indefinitely, never signalling expiry.
            if t_end >= self.tau_E:
                self.traj.append(
                    TrajectoryPoint(t_end, float(self.pos[0]), float(self.pos[1]), STOP)
                )
                self._nav_waypoints = None
                self._nav_wp_idx = 0
                return t  # expiry signal

            wp = self._nav_waypoints[self._nav_wp_idx]
            old_pos = self.pos.copy()
            self._step_move(wp, WALK, t_end)

            # FIX BUG H: pure distance threshold (0.05 m) fails when the agent
            # overshoots the waypoint in one step (max step ≈ 0.217 m at FAST speed).
            # After overshoot, distance to wp *increases* and the check never fires.
            # Fix: also mark waypoint reached when the agent has PASSED it —
            # detected by the dot product of (wp−old_pos) and (wp−new_pos) going ≤ 0,
            # meaning the agent crossed the perpendicular plane through the waypoint.
            vec_old = wp - old_pos
            vec_new = wp - self.pos
            passed = float(np.dot(vec_old, vec_new)) <= 0.0
            if _dist(self.pos, wp) < 0.05 or passed:
                self._nav_wp_idx += 1
                if self._nav_wp_idx >= len(self._nav_waypoints):
                    self._nav_waypoints = None
                    self._nav_wp_idx = 0
            return t_end

        # ── (c) Dwell timeout ─────────────────────────────────
        if t_end >= self.tau_E:
            self.traj.append(
                TrajectoryPoint(t_end, float(self.pos[0]), float(self.pos[1]), STOP)
            )
            return t  # expiry signal

        # ── Choose movement target ─────────────────────────────
        exhibit = self.target_exhibit
        E_pos = exhibit.pos
        r_stop = self.cfg.r_stop
        L_start = self.pos.copy()
        L_end = E_pos.copy()

        blocking = [
            tb
            for tb in self.cfg.tables
            if _segment_intersects_table(L_start, L_end, tb)
        ]

        if blocking:
            nearest = min(
                blocking,
                key=lambda tb: _dist_point_to_table(self.pos[0], self.pos[1], tb),
            )
            if _dist_point_to_table(self.pos[0], self.pos[1], nearest) < 0.05:
                # Corner-clip guard: step through rather than compute a detour
                self._step_move(E_pos, self.phase, t_end)
            else:
                waypoints = self._compute_obstacle_waypoints(nearest, L_start, L_end)
                if waypoints:
                    self._nav_waypoints = waypoints
                    self._nav_wp_idx = 0
                    wp = waypoints[0]
                    old_pos = self.pos.copy()
                    self._step_move(wp, WALK, t_end)
                    # Same overshoot fix as in the nav-continuation branch
                    vec_old = wp - old_pos
                    vec_new = wp - self.pos
                    passed = float(np.dot(vec_old, vec_new)) <= 0.0
                    if _dist(self.pos, wp) < 0.05 or passed:
                        self._nav_wp_idx = 1
                        if self._nav_wp_idx >= len(self._nav_waypoints):
                            self._nav_waypoints = None
                            self._nav_wp_idx = 0
                else:
                    self._step_move(E_pos, self.phase, t_end)
            return t_end

        # ── Unobstructed movement ─────────────────────────────
        D = _dist(self.pos, E_pos)

        if self.phase == STOP:
            if D <= r_stop:
                phi = float(
                    self.rng.uniform(
                        exhibit.normal_angle - math.pi / 2,
                        exhibit.normal_angle + math.pi / 2,
                    )
                )
                rho = float(self.rng.uniform(0.0, 1.0))
                k = -0.80 if self.rng.random() < self.cfg.w1 else 4.0
                r = r_stop * (rho ** (1.0 / (k + 1.0)))
                P_w = E_pos + r * np.array([math.cos(phi), math.sin(phi)])
                self._step_move(P_w, STOP, t_end)
            else:
                self._step_move(E_pos, STOP, t_end)
        else:
            self._step_move(E_pos, self.phase, t_end)

        D = _dist(self.pos, E_pos)

        # ── Excursion tracking ─────────────────────────────────
        if self.phase == STOP and D > r_stop and not self.excursing:
            self.excursing = True
            self.excurse_start = t_end
        if self.excursing and D <= r_stop:
            self.excurse += t_end - self.excurse_start
            self.excursing = False
        if self.excursing and (self.excurse + (t_end - self.excurse_start)) > 15.0:
            # FIX BUG 1 + 4: signal expiry instead of setting target to None.
            #
            # OLD (broken):
            #   self.target_exhibit = None   ← dwell_expired check requires
            #   return t_end                    target is not None → never fires
            #                                → agent frozen permanently
            #
            # NEW: keep target_exhibit set; set tau_E = t_end so the caller's
            # dwell_expired condition evaluates True this tick.
            # For solo agents: caller advances to next exhibit immediately.
            # For group members: Phase 1 sees tau_E < 9999, updates group
            # deadline; Phase 2 fires same tick and redirects all members.
            self.tau_E = t_end
            self._nav_waypoints = None
            self._nav_wp_idx = 0
            return t  # expiry signal (sample already recorded above)

        # ── Phase transitions ─────────────────────────────────
        if self.phase == WALK and D < r_stop * 2.0:
            self.phase = APPROACH
        if self.phase == APPROACH and D < r_stop:
            self.phase = STOP
            self.tau_E = t_end + self.sample_dwell()

        return t_end

    # ── advance_exhibit ───────────────────────────────────────────────────────

    def advance_exhibit(self, next_exhibit: Optional[Exhibit]) -> None:
        """
        Called by MultiSimulator (solo) or Group sync (group members) when
        the dwell deadline has passed.  Resets all per-exhibit state.
        Passing None marks the agent as done.
        """
        self._nav_waypoints = None
        self._nav_wp_idx = 0

        if next_exhibit is None:
            self.done = True
            self.target_exhibit = None
            return

        self.target_exhibit = next_exhibit
        self.visited.add(next_exhibit)
        self.phase = WALK
        self.tau_E = 9_999.0
        self.excurse = 0.0
        self.excurse_start = 0.0
        self.excursing = False

    # ── initialise ────────────────────────────────────────────────────────────

    def initialise(self, pos: np.ndarray, first_exhibit: Exhibit, t0: float) -> None:
        """
        Set start position, initial velocity, and first target.

        `visited = {first_exhibit}` and `E_prev = first_exhibit` are set
        immediately — before tick 1 — because `visited` is a planning
        EXCLUSION set, not a record of past movement.  Committing at
        planning time prevents first_exhibit from being re-selected.
        """
        self.pos = pos.copy()
        self.v = self.sample_speed(WALK)
        self.traj = [TrajectoryPoint(t0, float(pos[0]), float(pos[1]), WALK)]
        self.done = False
        self._nav_waypoints = None
        self._nav_wp_idx = 0

        self.visited = {first_exhibit}
        self.E_prev = first_exhibit
        self.advance_exhibit(first_exhibit)


# ══════════════════════════════════════════════════════════════════════════════
#  Group
# ══════════════════════════════════════════════════════════════════════════════


class Group:
    """
    Agents sharing a pre-built itinerary and minimum-intersection dwell
    deadlines.  All members advance simultaneously when the shared deadline
    passes (Phase 2 of group synchronisation).
    """

    def __init__(self, group_id: int, members: List[AgentState]) -> None:
        self.group_id = group_id
        self.members = members
        self.itinerary: List[Exhibit] = []
        self.dwell_ends: List[float] = []
        self.current_idx: int = 0

    @property
    def current_exhibit(self) -> Optional[Exhibit]:
        if self.current_idx < len(self.itinerary):
            return self.itinerary[self.current_idx]
        return None

    def all_done(self) -> bool:
        return all(m.done for m in self.members)


# ══════════════════════════════════════════════════════════════════════════════
#  MultiSimulator
# ══════════════════════════════════════════════════════════════════════════════


class MultiSimulator:
    """
    Runs N solo agents and G groups under a single global clock.

    Clock: t_start is snapshotted at the top of each tick. All agents see the
    same t_start. self._t advances by 1/sr only after ALL agents have been
    processed.

    Collision avoidance: converging pairs are identified; those agents are
    processed in random order after all non-colliding agents have moved.

    Group synchronisation:
    • Phase 1 (every tick): deadline = min(tau_E) over members in STOP;
      can only decrease.
    • Phase 2 (once per exhibit): when deadline passes, ALL members redirect.
    """

    def __init__(self, cfg: MuseumConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._t: float = 0.0
        self._agents: List[AgentState] = []
        self._groups: List[Group] = []

    def _new_agent(self, agent_id: int) -> AgentState:
        move_type = (
            AgentMoveType.SLOW if self.rng.random() < 0.40 else AgentMoveType.FAST
        )
        return AgentState(
            agent_id, self.cfg, move_type, int(self.rng.integers(0, 2**31))
        )

    def _random_start_pos(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.cfg.start_area
        return np.array(
            [float(self.rng.uniform(xmin, xmax)), float(self.rng.uniform(ymin, ymax))]
        )

    def _pick_first_exhibit(self) -> Exhibit:
        pool = self.cfg.start_exhibits
        if not pool:
            raise ValueError(
                "MuseumConfig.start_exhibits is empty — cannot initialise agents. "
                "Ensure at least one exhibit is reachable from the start area."
            )
        return pool[int(self.rng.integers(0, len(pool)))]

    def _next_exhibit_for(self, agent: AgentState) -> Optional[Exhibit]:
        cfg = self.cfg
        if len(agent.visited) >= cfg.K or len(agent.visited) >= len(cfg.exhibits):
            return None
        if agent.E_prev is None:
            return None
        nbrs = cfg.neighbourhoods.get(agent.E_prev.exhibit_id, [])
        unvisited_nbrs = [e for e in nbrs if e not in agent.visited]
        if unvisited_nbrs:
            return unvisited_nbrs[int(self.rng.integers(0, len(unvisited_nbrs)))]
        unvisited_all = [e for e in cfg.exhibits if e not in agent.visited]
        if unvisited_all:
            return unvisited_all[int(self.rng.integers(0, len(unvisited_all)))]
        return None

    def _build_group_itinerary(
        self,
        group: Group,
        first_exhibit: Exhibit,
        rng_agent: np.random.Generator,
    ) -> None:
        cfg = self.cfg
        visited: Set[Exhibit] = {first_exhibit}
        E_prev = first_exhibit
        itinerary: List[Exhibit] = [first_exhibit]

        while len(itinerary) < cfg.K and len(itinerary) < len(cfg.exhibits):
            nbrs = cfg.neighbourhoods.get(E_prev.exhibit_id, [])
            unvisited_nbrs = [e for e in nbrs if e not in visited]
            if unvisited_nbrs:
                nxt = unvisited_nbrs[int(rng_agent.integers(0, len(unvisited_nbrs)))]
            else:
                unvisited_all = [e for e in cfg.exhibits if e not in visited]
                if not unvisited_all:
                    break
                nxt = unvisited_all[int(rng_agent.integers(0, len(unvisited_all)))]
            itinerary.append(nxt)
            visited.add(nxt)
            E_prev = nxt

        group.itinerary = itinerary
        group.dwell_ends = [9_999.0] * len(itinerary)
        group.current_idx = 0

    def simulate(
        self,
        n_solo: int = 3,
        n_groups: int = 2,
        group_size_range: Tuple[int, int] = (2, 4),
    ) -> List[List[TrajectoryPoint]]:
        cfg = self.cfg
        dt = 1.0 / cfg.sr
        self._t = 0.0
        self._agents = []
        self._groups = []

        agent_id_counter = 0

        for _ in range(n_solo):
            agent = self._new_agent(agent_id_counter)
            agent_id_counter += 1
            agent.initialise(self._random_start_pos(), self._pick_first_exhibit(), 0.0)
            self._agents.append(agent)

        for g_idx in range(n_groups):
            size = int(self.rng.integers(group_size_range[0], group_size_range[1] + 1))
            members = [self._new_agent(agent_id_counter + i) for i in range(size)]
            agent_id_counter += size

            group = Group(group_id=g_idx, members=members)
            self._groups.append(group)

            base_pos = self._random_start_pos()
            first_exhibit = self._pick_first_exhibit()

            for member in members:
                jitter = self.rng.uniform(-0.30, 0.30, size=2)
                member.initialise(
                    _clamp_to_boundary(base_pos + jitter, cfg.boundary),
                    first_exhibit,
                    0.0,
                )
                member.group = group

            self._build_group_itinerary(group, first_exhibit, members[0].rng)
            self._agents.extend(members)

        max_ticks = int(cfg.t_max * cfg.sr) + 1

        for _tick in range(max_ticks):
            t = self._t

            if all(a.done for a in self._agents):
                break

            active = [a for a in self._agents if not a.done]
            halted_ids: Set[int] = set()

            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    ai, aj = active[ii], active[jj]
                    if _dist(ai.pos, aj.pos) < cfg.r_collision:
                        if _headings_converging(
                            ai.current_theta(),
                            aj.current_theta(),
                            cfg.collision_angle_tol,
                        ):
                            halted_ids.add(ai.agent_id)
                            halted_ids.add(aj.agent_id)

            halted = [a for a in active if a.agent_id in halted_ids]
            moving = [a for a in active if a.agent_id not in halted_ids]

            def _process(agent: AgentState) -> None:
                """Tick one agent and advance solo itinerary if dwell expired."""
                new_t = agent.advance_one_tick(t)
                dwell_expired = (
                    new_t < t + dt - 1e-9
                    and not agent.done
                    and agent.target_exhibit is not None
                )
                if dwell_expired and agent.group is None:
                    agent.E_prev = agent.target_exhibit
                    agent.advance_exhibit(self._next_exhibit_for(agent))

            for agent in moving:
                _process(agent)

            if halted:
                for idx in self.rng.permutation(len(halted)):
                    _process(halted[idx])

            # Group synchronisation
            for group in self._groups:
                if group.all_done():
                    continue
                exhibit = group.current_exhibit
                if exhibit is None:
                    for m in group.members:
                        m.done = True
                    continue

                idx = group.current_idx

                # Phase 1: minimum-intersection deadline (runs until Phase 2 fires)
                if t + dt < group.dwell_ends[idx]:
                    stop_members = [
                        m
                        for m in group.members
                        if m.target_exhibit == exhibit
                        and m.phase == STOP
                        and m.tau_E < 9_999.0
                    ]
                    if stop_members:
                        new_deadline = min(m.tau_E for m in stop_members)
                        if new_deadline < group.dwell_ends[idx]:
                            group.dwell_ends[idx] = new_deadline
                            for m in group.members:
                                m.tau_E = new_deadline

                # Phase 2: synchronised advance (fires once per exhibit)
                if group.dwell_ends[idx] < 9_999.0 and t + dt >= group.dwell_ends[idx]:
                    group.current_idx += 1
                    nxt = group.current_exhibit
                    for m in group.members:
                        m.E_prev = exhibit
                        m.advance_exhibit(nxt)

            self._t += dt
            if self._t >= cfg.t_max:
                break

        return [a.traj for a in self._agents]


# ══════════════════════════════════════════════════════════════════════════════
#  Simulator  —  backwards-compatible single-agent wrapper
# ══════════════════════════════════════════════════════════════════════════════


class Simulator:
    def __init__(self, cfg: MuseumConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.seed = seed

    def simulate(self) -> List[TrajectoryPoint]:
        return MultiSimulator(self.cfg, seed=self.seed).simulate(n_solo=1, n_groups=0)[
            0
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  Museum factory helpers
# ══════════════════════════════════════════════════════════════════════════════


def make_rect_table(table_id: int, x: float, y: float, w: float, h: float) -> Table:
    return Table(
        table_id=table_id,
        vertices=np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64
        ),
    )


def build_neighbourhoods(
    exhibits: List[Exhibit], radius: float
) -> Dict[int, List[Exhibit]]:
    return {
        ei.exhibit_id: [
            ej
            for ej in exhibits
            if ej.exhibit_id != ei.exhibit_id
            and math.hypot(ei.x - ej.x, ei.y - ej.y) <= radius
        ]
        for ei in exhibits
    }


def make_demo_museum() -> MuseumConfig:
    T1 = make_rect_table(1, 3.0, 5.0, 3.0, 1.5)
    T2 = make_rect_table(2, 13.0, 5.0, 3.0, 1.5)
    T3 = make_rect_table(3, 5.0, 10.0, 3.0, 1.5)
    T4 = make_rect_table(4, 12.0, 10.0, 3.0, 1.5)

    raw: List[Tuple[int, float, float, Table]] = [
        (1, 3.0, 5.55, T1),
        (2, 3.0, 6.00, T1),
        (3, 6.0, 5.75, T1),
        (4, 14.5, 5.00, T2),
        (5, 16.0, 5.75, T2),
        (6, 14.5, 6.50, T2),
        (7, 6.5, 10.00, T3),
        (8, 8.0, 10.75, T3),
        (9, 6.5, 11.50, T3),
        (10, 12.0, 10.75, T4),
        (11, 12.5, 11.50, T4),
        (12, 14.5, 11.50, T4),
    ]
    exhibits = [
        Exhibit(
            exhibit_id=i, x=x, y=y, table=t, normal_angle=_edge_normal_angle(x, y, t)
        )
        for i, x, y, t in raw
    ]

    return MuseumConfig(
        tables=[T1, T2, T3, T4],
        exhibits=exhibits,
        neighbourhoods=build_neighbourhoods(exhibits, radius=6.0),
        start_exhibits=[e for e in exhibits if e.y <= 6.0],
        start_area=(7.0, 13.0, 0.0, 1.0),
        boundary=(0.0, 20.0, 0.0, 14.0),
        sr=12.0,
        r_stop=0.50,
        r_collision=0.40,
        collision_angle_tol=math.radians(30.0),
        sigma_theta_walk=1.0,
        sigma_theta_approach=1.0,
        sigma_theta_stop=0.656,
        w1=0.33,
        K=8,
        t_max=1_800.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

_AGENT_COLORS = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#A65628",
    "#F781BF",
    "#999999",
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
]


def plot_trajectories(
    trajs: List[List[TrajectoryPoint]],
    cfg: MuseumConfig,
    agent_ids: Optional[List[int]] = None,
    *,
    title: str = "Simulated Visitor Trajectories",
    figsize: Tuple[float, float] = (14.0, 10.0),
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax, ymin, ymax = cfg.boundary

    ax.add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="black",
            facecolor="#F5F5F0",
            zorder=0,
        )
    )
    sxmin, sxmax, symin, symax = cfg.start_area
    ax.add_patch(
        plt.Rectangle(
            (sxmin, symin),
            sxmax - sxmin,
            symax - symin,
            linewidth=0,
            facecolor="#FFFACD",
            alpha=0.9,
            zorder=1,
        )
    )
    ax.text(
        (sxmin + sxmax) / 2,
        (symin + symax) / 2,
        "ENTRY",
        ha="center",
        va="center",
        fontsize=8,
        color="#998800",
        fontweight="bold",
        zorder=2,
    )

    for tb in cfg.tables:
        ax.add_patch(
            MplPolygon(
                tb.vertices,
                closed=True,
                facecolor="#C4A882",
                edgecolor="#7A5C2E",
                linewidth=1.5,
                zorder=2,
            )
        )
        cx, cy = tb.vertices.mean(axis=0)
        ax.text(
            cx,
            cy,
            f"T{tb.table_id}",
            ha="center",
            va="center",
            fontsize=7,
            color="#4A3520",
            fontweight="bold",
            zorder=3,
        )

    for e in cfg.exhibits:
        ax.plot(
            e.x,
            e.y,
            marker="^",
            color="#222222",
            markersize=7,
            zorder=5,
            linestyle="none",
        )
        ax.text(
            e.x + 0.18,
            e.y + 0.18,
            f"E{e.exhibit_id}",
            fontsize=7,
            color="#111111",
            zorder=6,
        )

    _LSTYLE = {WALK: "-", APPROACH: "--", STOP: ":"}
    for idx, traj in enumerate(trajs):
        color = _AGENT_COLORS[idx % len(_AGENT_COLORS)]
        if not traj:
            continue
        xs = [p.x for p in traj]
        ys = [p.y for p in traj]
        phases = [p.phase for p in traj]
        for i in range(len(xs) - 1):
            ax.plot(
                [xs[i], xs[i + 1]],
                [ys[i], ys[i + 1]],
                color=color,
                linewidth=0.9,
                alpha=0.75,
                linestyle=_LSTYLE.get(phases[i], "-"),
                zorder=4,
            )
        ax.plot(
            xs[0],
            ys[0],
            "o",
            color=color,
            markersize=8,
            zorder=10,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
        ax.plot(
            xs[-1],
            ys[-1],
            "*",
            color=color,
            markersize=12,
            zorder=10,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )

    agent_handles = [
        Line2D(
            [0],
            [0],
            color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
            linewidth=2,
            label=f"Agent {agent_ids[i] if agent_ids else i}",
        )
        for i in range(len(trajs))
    ]
    phase_handles = [
        Line2D([0], [0], color="grey", linewidth=2, linestyle=_LSTYLE[ph], label=ph)
        for ph in (WALK, APPROACH, STOP)
    ]
    ax.legend(
        handles=agent_handles + phase_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
        ncol=2,
    )
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_trajectory(
    traj, cfg, *, title="Simulated Visitor Trajectory", figsize=(13.0, 9.0)
):
    return plot_trajectories([traj], cfg, title=title, figsize=figsize)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, os, csv
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("-ns", "--n-solo", type=int, default=3)
    p.add_argument("-ng", "--n-groups", type=int, default=2)
    p.add_argument(
        "-gs", "--group-size", type=int, nargs=2, default=[2, 4], metavar=("MIN", "MAX")
    )
    p.add_argument("-o", "--outdir", default=".")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    museum = make_demo_museum()
    msim = MultiSimulator(museum, seed=args.seed)
    trajs = msim.simulate(
        n_solo=args.n_solo,
        n_groups=args.n_groups,
        group_size_range=tuple(args.group_size),
    )

    print(f"Agents simulated: {len(trajs)}")
    for i, traj in enumerate(trajs):
        if not traj:
            print(f"  agent {i:3d}: empty")
            continue
        by_phase = {
            ph: sum(1 for p in traj if p.phase == ph) for ph in (WALK, APPROACH, STOP)
        }
        print(
            f"  agent {i:3d}: {len(traj):5d} pts  t={traj[-1].t:6.1f}s  "
            + "  ".join(f"{ph}={by_phase[ph]}" for ph in (WALK, APPROACH, STOP))
        )

    stem = f"seed{args.seed}_groups{args.n_groups}_solo{args.n_solo}"
    fig = plot_trajectories(
        trajs,
        museum,
        title=f"Museum Trajectories — seed {args.seed} "
        f"({args.n_solo} solo + {args.n_groups} groups)",
    )
    plt.savefig(
        os.path.join(args.outdir, f"plot_{stem}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    with open(
        os.path.join(args.outdir, f"trajectories_{stem}.csv"), "w", newline=""
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "timestamp", "person_id", "phase"])
        for aid, traj in enumerate(trajs):
            for pt in traj:
                w.writerow(
                    [round(pt.x, 4), round(pt.y, 4), round(pt.t, 4), aid, pt.phase]
                )
    print("Done.")
