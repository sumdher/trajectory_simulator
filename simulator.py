"""
museum_simulator.py
===================
Simulates visitor trajectories through a museum with table obstacles.

Algorithm reference
-------------------
Phases     : WALK → APPROACH → STOP (per exhibit)
Obstacles  : rectangular / polygonal tables; navigated via MoveAroundObstacle
Sampling   : UWB-style at sr = 12 Hz

Usage
-----
    museum = make_demo_museum()
    sim    = Simulator(museum, seed=42)
    traj   = sim.simulate()
    fig    = plot_trajectory(traj, museum)
    plt.show()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
_P_V: Dict[str, float] = {WALK: 0.50, APPROACH: 0.30, STOP: 0.66}
_P_θ: Dict[str, float] = {WALK: 0.50, APPROACH: 0.30, STOP: 0.75}


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-built Beta distributions (frozen for repeated sampling efficiency)
# ══════════════════════════════════════════════════════════════════════════════

_BETA_WALK = _scipy_beta(a=2.0, b=10.1)  # walk-phase speed
_BETA_DWELL = _scipy_beta(a=1.9, b=4.0)  # dwell time at exhibit (mode ≈ 50 s)


def _rescale(u: float, lo: float, hi: float) -> float:
    """Linearly map u ∈ [0, 1] → [lo, hi]."""
    return lo + u * (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
#  Domain objects
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Table:
    """
    Polygonal table obstacle with CCW-ordered corner vertices.

    Parameters
    ----------
    table_id : int
        Unique identifier.
    vertices : np.ndarray, shape (n, 2)
        Corner coordinates in counter-clockwise order.
    """

    table_id: int
    vertices: np.ndarray

    def __hash__(self) -> int:
        return hash(self.table_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Table) and self.table_id == other.table_id

    def __repr__(self) -> str:
        return f"Table(id={self.table_id}, n_verts={len(self.vertices)})"


@dataclass
class Exhibit:
    """
    Museum exhibit - a visitable point of interest near one table.

    Parameters
    ----------
    exhibit_id : int
        Unique identifier.
    x, y       : float
        Exhibit coordinates [m].
    table      : Table
        The table this exhibit belongs to (potential obstacle).
    """

    exhibit_id: int
    x: float
    y: float
    table: Table

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        return hash(self.exhibit_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Exhibit) and self.exhibit_id == other.exhibit_id

    def __repr__(self) -> str:
        return f"E{self.exhibit_id}({self.x:.2f},{self.y:.2f})"


@dataclass
class MuseumConfig:
    """
    Complete museum layout and simulation hyper-parameters.

    Parameters
    ----------
    tables          : list of all Table obstacles
    exhibits        : list of all Exhibit objects
    neighbourhoods  : mapping exhibit_id → list of nearby Exhibits
    start_exhibits  : exhibits closest to the entry area (first target pool)
    start_area      : (xmin, xmax, ymin, ymax) rectangle for initial position
    boundary        : (xmin, xmax, ymin, ymax) hard museum boundary
    sr              : location-system sampling rate [Hz]  (UWB default = 12)
    r_stop          : radius within which visitor wanders during STOP phase [m]
    sigma_theta_*   : heading-perturbation std-dev per phase [rad]
    w1              : Bernoulli gate for disk sampling (< 0.5 → outer-dense)
    K               : maximum number of exhibits to visit (termination)
    t_max           : maximum trajectory duration [s] (termination)
    """

    # ── Layout ────────────────────────────────────────────────
    tables: List[Table]
    exhibits: List[Exhibit]
    neighbourhoods: Dict[int, List[Exhibit]]
    start_exhibits: List[Exhibit]
    start_area: Tuple[float, float, float, float]
    boundary: Tuple[float, float, float, float]
    # ── Physics ───────────────────────────────────────────────
    sr: float = 12.0
    r_stop: float = 0.50
    sigma_theta_walk: float = 0.30
    sigma_theta_approach: float = 0.10
    sigma_theta_stop: float = 0.50
    w1: float = 0.30
    # ── Termination ───────────────────────────────────────────
    K: int = 10
    t_max: float = 3_600.0


@dataclass
class TrajectoryPoint:
    """One recorded position sample."""

    t: float
    x: float
    y: float
    phase: str


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry utilities
# ══════════════════════════════════════════════════════════════════════════════


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance D(p, q)."""
    return float(np.linalg.norm(p - q))


def _signed_dist(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Signed distance sdist(v, L) from point *v* to the infinite line through
    *a → b*.

    Sign convention
    ---------------
    Positive  ↔  *v* is to the **left** of the directed line a→b
                 (the counter-clockwise side for CCW-wound polygons).
    Formula: cross(b−a, v−a) / |b−a|
    """
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-12:
        return 0.0
    # 2-D cross product: ab.x*(v.y-a.y) - ab.y*(v.x-a.x)
    return float((ab[0] * (v[1] - a[1]) - ab[1] * (v[0] - a[0])) / denom)


def _segments_properly_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> bool:
    """
    True iff open segments p1-p2 and p3-p4 properly intersect (endpoints
    excluded) using parametric line intersection.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-12:  # parallel / collinear
        return False
    t = float((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = float((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    _EPS = 1e-9
    return _EPS < t < 1.0 - _EPS and _EPS < u < 1.0 - _EPS


def _segment_intersects_table(S: np.ndarray, E: np.ndarray, table: Table) -> bool:
    """True iff segment S→E intersects at least one edge of *table*."""
    verts = table.vertices
    n = len(verts)
    return any(
        _segments_properly_intersect(S, E, verts[i], verts[(i + 1) % n])
        for i in range(n)
    )


def _clamp_to_boundary(
    pos: np.ndarray,
    bnd: Tuple[float, float, float, float],
) -> np.ndarray:
    """Project *pos* back inside the museum bounding box."""
    xmin, xmax, ymin, ymax = bnd
    return np.array(
        [
            float(np.clip(pos[0], xmin, xmax)),
            float(np.clip(pos[1], ymin, ymax)),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Simulator
# ══════════════════════════════════════════════════════════════════════════════


class Simulator:
    """
    Single-visitor museum trajectory simulator.

    Parameters
    ----------
    cfg  : MuseumConfig
    seed : optional int - seed for the internal ``numpy.random.Generator``
           (pass an integer for fully reproducible runs)
    """

    def __init__(self, cfg: MuseumConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Mutable simulation state - fully reset by ``simulate()``
        self._pos: np.ndarray = np.zeros(2)
        self._v: float = 0.0
        self._t: float = 0.0
        self._traj: List[TrajectoryPoint] = []

    # ──────────────────────────────────────────────────────────
    #  Sampling helpers
    # ──────────────────────────────────────────────────────────

    def _sample_speed(self, phase: str) -> float:
        """Draw v ~ (v | phase)."""
        if phase == WALK:
            return _rescale(float(_BETA_WALK.rvs(random_state=self.rng)), 0.50, 2.60)
        if phase == APPROACH:
            return float(self.rng.uniform(0.3, 0.7))
        return float(self.rng.uniform(0.1, 0.25))  # STOP

    def _sample_dtheta(self, phase: str) -> float:
        """Draw Δθ ~ N(0, σ_phase)."""
        sigma = {
            WALK: self.cfg.sigma_theta_walk,
            APPROACH: self.cfg.sigma_theta_approach,
            STOP: self.cfg.sigma_theta_stop,
        }[phase]
        return float(self.rng.normal(0.0, sigma))

    def _sample_dwell(self) -> float:
        """Draw τ_E ~ Beta(1.9, 4.0) scaled to [11, 180] seconds."""
        return _rescale(float(_BETA_DWELL.rvs(random_state=self.rng)), 11.0, 180.0)

    def _pick(self, candidates: List[Exhibit]) -> Exhibit:
        """Uniform random choice from a non-empty exhibit list."""
        return candidates[int(self.rng.integers(0, len(candidates)))]

    # ──────────────────────────────────────────────────────────
    #  Move  —  §Move(S, P, phase)
    # ──────────────────────────────────────────────────────────

    def _move(self, target: np.ndarray, phase: str) -> None:
        """
        Execute one discrete Move step.

        1. Possibly update speed v via Bernoulli gate g_v.
        2. Possibly update heading perturbation Δθ via gate g_θ.
        3. Compute new heading θ = atan2(target) + Δθ.
        4. Advance position by v/sr along θ, clamped to boundary.
        5. Advance clock by 1/sr and record the sample.
        """
        cfg = self.cfg

        # Bernoulli gates
        if self.rng.random() < _P_V[phase]:
            self._v = self._sample_speed(phase)

        dtheta = self._sample_dtheta(phase) if self.rng.random() < _P_θ[phase] else 0.0

        diff = target - self._pos
        theta = math.atan2(diff[1], diff[0]) + dtheta
        step = self._v / cfg.sr

        self._pos = _clamp_to_boundary(
            self._pos + step * np.array([math.cos(theta), math.sin(theta)]),
            cfg.boundary,
        )
        self._t += 1.0 / cfg.sr
        self._traj.append(
            TrajectoryPoint(self._t, float(self._pos[0]), float(self._pos[1]), phase)
        )

    # ──────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────
    #  Move (deterministic)  —  used during obstacle navigation
    #  MoveAroundObstacle  —  §MoveAroundObstacle(S, E, T_blocking, L)
    # ──────────────────────────────────────────────────────────

    def _move_around_obstacle(
        self,
        exhibit: Exhibit,
        blocking_table: Table,
        L_start: np.ndarray,
        L_end: np.ndarray,
    ) -> None:
        """
        Navigate around *blocking_table* towards *exhibit*.

        Algorithm
        ---------
        1. Draw line L from S to E.
        2. Classify every table vertex by which side of L it falls on
           (positive = left of S→E, negative = right).
           Vertices exactly on L are ignored.
        3. V = the side with FEWER vertices (smaller table partition).
           Tie (2 vs 2): pick the side whose nearest vertex is closer to S.
        4. Order V by CCW polygon index, starting from the V-vertex
           closest to S, in the direction that stays inside V.
        5. Walk each waypoint in order, then proceed to E.
        """
        verts = blocking_table.vertices
        n = len(verts)
        E_pos = exhibit.pos
        _EPS = 1e-9

        # 1-2. Classify vertices by side of L
        sdists = [_signed_dist(verts[i], L_start, L_end) for i in range(n)]
        pos_side = [i for i in range(n) if sdists[i] > _EPS]
        neg_side = [i for i in range(n) if sdists[i] < -_EPS]

        # 3. Pick smaller partition; tie → side whose closest vertex is nearer S
        if len(pos_side) < len(neg_side):
            V_idx = pos_side
        elif len(neg_side) < len(pos_side):
            V_idx = neg_side
        else:
            # equal — pick by which side has the vertex closest to S
            nearest_pos = min(pos_side, key=lambda i: _dist(self._pos, verts[i]))
            nearest_neg = min(neg_side, key=lambda i: _dist(self._pos, verts[i]))
            if _dist(self._pos, verts[nearest_pos]) <= _dist(
                self._pos, verts[nearest_neg]
            ):
                V_idx = pos_side
            else:
                V_idx = neg_side

        if not V_idx:
            return  # line passes exactly through vertices — no detour needed

        V_set = set(V_idx)

        # 4. Start from the V-vertex closest to S
        start_idx = min(V_set, key=lambda i: _dist(self._pos, verts[i]))

        # Determine traversal direction: whichever of CCW/CW keeps us in V
        delta = +1 if (start_idx + 1) % n in V_set else -1

        # 5. Walk waypoints in order
        _CLOSE = 0.05
        cur = start_idx
        for _ in range(n + 1):
            if cur not in V_set:
                break
            target = verts[cur]
            while _dist(self._pos, target) >= _CLOSE:
                self._move(target, WALK)
            cur = (cur + delta) % n

    #  MoveToExhibit  —  §MoveToExhibit(S, E)
    # ──────────────────────────────────────────────────────────

    def _move_to_exhibit(self, exhibit: Exhibit) -> None:
        """
        Drive the agent from its current position to *exhibit*, handling:
        - Phase transitions  WALK → APPROACH → STOP
        - Obstacle avoidance via _move_around_obstacle
        - Dwell-time sampling on arrival (STOP phase)
        - Excursion tracking and 15-second excursion limit
        - Disk-sampling wandering during STOP
        """
        cfg = self.cfg
        E_pos = exhibit.pos

        phase = WALK
        tau_E = 9_999.0  # overwritten once STOP phase begins
        D = _dist(self._pos, E_pos)
        excurse = 0.0  # accumulated off-disk excursion time
        excurse_start = 0.0
        excursing = False
        r_stop = cfg.r_stop

        # Safety cap prevents unbounded loops on degenerate layouts
        _MAX_STEPS = int(cfg.t_max * cfg.sr) + 1

        for _ in range(_MAX_STEPS):
            if self._t > tau_E:
                break

            L_start = self._pos.copy()
            L_end = E_pos.copy()

            # Check ALL tables; navigate around the closest blocker to S.
            blocking = [
                t for t in cfg.tables if _segment_intersects_table(L_start, L_end, t)
            ]

            if not blocking:
                # ── Unobstructed path ─────────────────────────────
                if phase == STOP:
                    if D <= r_stop:
                        # Disk sampling: wander inside the stop-radius
                        phi = float(self.rng.uniform(0.0, 2.0 * math.pi))
                        rho = float(self.rng.uniform(0.0, 1.0))
                        # g_k gates between inner-dense (k=-0.66) and
                        # outer-dense (k=4) concentration
                        k = -0.66 if (self.rng.random() < cfg.w1) else 4.0
                        r = r_stop * (rho ** (1.0 / (k + 1.0)))
                        P_w = E_pos + r * np.array([math.cos(phi), math.sin(phi)])
                        self._move(P_w, STOP)
                    else:
                        self._move(E_pos, STOP)
                else:
                    self._move(E_pos, phase)

                D = _dist(self._pos, E_pos)

                # ── Excursion tracking ────────────────────────────
                if phase == STOP and D > r_stop and not excursing:
                    excursing = True
                    excurse_start = self._t

                if excursing and D <= r_stop:
                    excurse += self._t - excurse_start
                    excursing = False

                if excursing and (excurse + (self._t - excurse_start)) > 15.0:
                    return  # excursion limit exceeded

                # ── Phase transitions ─────────────────────────────
                if phase == WALK and D < r_stop * 2.0:
                    phase = APPROACH

                if phase == APPROACH and D < r_stop:
                    phase = STOP
                    tau_E = self._t + self._sample_dwell()

            else:
                # Navigate around the closest blocking table.
                nearest_blocker = min(
                    blocking,
                    key=lambda t: _dist_point_to_table(self._pos[0], self._pos[1], t),
                )
                # Corner-clip guard: if the agent is already essentially ON
                # the blocking table boundary (it just rounded a corner and
                # the path still barely intersects that same corner), the
                # obstacle call would loop forever.  In that case just take
                # one direct step toward E to escape the degenerate clip.
                _CLOSE = 0.05
                if (
                    _dist_point_to_table(self._pos[0], self._pos[1], nearest_blocker)
                    < _CLOSE
                ):
                    self._move(E_pos, phase)
                else:
                    self._move_around_obstacle(exhibit, nearest_blocker, L_start, L_end)
                D = _dist(self._pos, E_pos)

    # ──────────────────────────────────────────────────────────
    #  simulate  —  §SimulateTrajectory (main)
    # ──────────────────────────────────────────────────────────

    def simulate(self) -> List[TrajectoryPoint]:
        """
        Run one complete visitor trajectory through the museum.

        Algorithm
        ---------
        1. Draw start position uniformly from the start area.
        2. Select first exhibit uniformly from ``cfg.start_exhibits``.
        3. Repeat until a termination criterion is met:
               a. MoveToExhibit(S, E)
               b. Select next exhibit from unvisited neighbours (or fallback
                  to any unvisited exhibit).
        Termination criteria (any one suffices):
               • ``|E_visited| ≥ K``
               • ``t ≥ t_max``
               • All exhibits visited

        Returns
        -------
        List[TrajectoryPoint]
            Chronological position records at ``cfg.sr`` Hz, each annotated
            with the movement phase at that instant.
        """
        cfg = self.cfg

        # ── Reset mutable state ────────────────────────────────
        self._traj = []
        self._t = 0.0
        visited: Set[Exhibit] = set()

        # ── Random start position ──────────────────────────────
        xmin, xmax, ymin, ymax = cfg.start_area
        self._pos = np.array(
            [
                float(self.rng.uniform(xmin, xmax)),
                float(self.rng.uniform(ymin, ymax)),
            ]
        )
        self._traj.append(
            TrajectoryPoint(0.0, float(self._pos[0]), float(self._pos[1]), WALK)
        )

        # ── First exhibit from start-area pool ─────────────────
        E = self._pick(cfg.start_exhibits)
        visited.add(E)
        E_prev = E

        # ── Initial speed draw ─────────────────────────────────
        self._v = self._sample_speed(WALK)

        # ── Main simulation loop ───────────────────────────────
        while True:
            # Termination criteria
            if len(visited) >= cfg.K:
                break
            if self._t >= cfg.t_max:
                break
            if len(visited) >= len(cfg.exhibits):
                break

            self._move_to_exhibit(E)

            # Select next exhibit: prefer unvisited neighbours of E_prev
            nbrs = cfg.neighbourhoods.get(E_prev.exhibit_id, [])
            unvisited_nbrs = [e for e in nbrs if e not in visited]

            if unvisited_nbrs:
                E = self._pick(unvisited_nbrs)
            else:
                # Fallback: any unvisited exhibit in the museum
                unvisited_all = [e for e in cfg.exhibits if e not in visited]
                if unvisited_all:
                    E = self._pick(unvisited_all)
                else:
                    break

            visited.add(E)
            E_prev = E

        return self._traj


# ══════════════════════════════════════════════════════════════════════════════
#  Museum factory helpers
# ══════════════════════════════════════════════════════════════════════════════


def make_rect_table(
    table_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
) -> Table:
    """
    Create an axis-aligned rectangular table.

    Vertices wound counter-clockwise starting at the bottom-left corner.
    """
    verts = np.array(
        [
            [x, y],  # BL
            [x + w, y],  # BR
            [x + w, y + h],  # TR
            [x, y + h],  # TL
        ],
        dtype=np.float64,
    )
    return Table(table_id=table_id, vertices=verts)


def _dist_point_to_table(px: float, py: float, table: Table) -> float:
    """
    Minimum Euclidean distance from point (px, py) to the *closest point on
    the boundary* of *table*.  Returns 0.0 if the point is inside the polygon.

    For convex polygons (all our tables are convex) this is computed as the
    minimum distance to each edge segment, taken as 0 when inside.
    """
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

    # Check if point is inside (all cross-products same sign ⟹ inside CCW polygon)
    inside = all(
        float(
            (verts[(i + 1) % n][0] - verts[i][0]) * (py - verts[i][1])
            - (verts[(i + 1) % n][1] - verts[i][1]) * (px - verts[i][0])
        )
        >= -1e-9
        for i in range(n)
    )
    return 0.0 if inside else min_d


def validate_exhibits(exhibits: List[Exhibit], margin: float = 0.5) -> None:
    """
    Raise ``ValueError`` for any exhibit whose distance to its associated
    table is less than *margin* metres (default 0.5 m).

    Call this after constructing your exhibit list so layout errors are caught
    immediately rather than producing silently wrong trajectories.
    """
    violations: List[str] = []
    for e in exhibits:
        d = _dist_point_to_table(e.x, e.y, e.table)
        if d < margin:
            status = "INSIDE table" if d == 0.0 else f"only {d:.3f} m from edge"
            violations.append(
                f"  E{e.exhibit_id} ({e.x}, {e.y}) → T{e.table.table_id}: {status}"
            )
    if violations:
        raise ValueError(
            f"Exhibit margin violation (required ≥ {margin} m):\n"
            + "\n".join(violations)
        )


def build_neighbourhoods(
    exhibits: List[Exhibit],
    radius: float,
) -> Dict[int, List[Exhibit]]:
    """
    Compute neighbourhood sets for every exhibit.

    ``N_j`` = all exhibits (excluding ``E_j`` itself) within *radius* metres.
    """
    nbrs: Dict[int, List[Exhibit]] = {}
    for ei in exhibits:
        nbrs[ei.exhibit_id] = [
            ej
            for ej in exhibits
            if ej.exhibit_id != ei.exhibit_id
            and math.hypot(ei.x - ej.x, ei.y - ej.y) <= radius
        ]
    return nbrs


# ══════════════════════════════════════════════════════════════════════════════
#  Demo museum layout
# ══════════════════════════════════════════════════════════════════════════════


def make_demo_museum() -> MuseumConfig:
    """
    Build a small illustrative museum (20 m × 14 m) with:
    • 4 rectangular tables
    • 12 exhibits — T1 has 2 on its left face, T4 has 2 on its top face
    • neighbourhood radius = 6 m
    • start area: a 6 m-wide patch at the centre of the bottom edge

    Layout sketch (units: metres)::

        y=14 ┌─────────────────────────┐
             │        [T3]   [T4]      │
             │                         │
             │   [T1]       [T2]       │
        y=1  │         ═══════         │  ← start area (x 7–13)
        y=0  └─────────────────────────┘
             x=0                     x=20

    Same-edge pairs
    ---------------
    T1 left face  (x = 2.5):  E1 (y=5.55) and E2 (y=6.00)
    T4 top  face  (y = 12.0): E11 (x=12.5) and E12 (x=14.5)
    """
    # ── Tables ────────────────────────────────────────────────
    T1 = make_rect_table(1, 3.0, 5.0, 3.0, 1.5)  # x=[3,6],   y=[5,6.5]
    T2 = make_rect_table(2, 13.0, 5.0, 3.0, 1.5)  # x=[13,16], y=[5,6.5]
    T3 = make_rect_table(3, 5.0, 10.0, 3.0, 1.5)  # x=[5,8],   y=[10,11.5]
    T4 = make_rect_table(4, 12.0, 10.0, 3.0, 1.5)  # x=[12,15], y=[10,11.5]
    tables = [T1, T2, T3, T4]

    # ── Exhibits ──────────────────────────────────────────────
    # All placed at exactly 0.5 m from their table's nearest face.
    #
    # Face offsets:
    #   left face:   x = table_x            − 0.5
    #   right face:  x = table_x + width    + 0.5
    #   bottom face: y = table_y            − 0.5
    #   top face:    y = table_y + height   + 0.5
    #
    # For two exhibits on the same face the perpendicular offset is identical;
    # they differ only in their coordinate along the face.
    # fmt: off
    raw: List[Tuple[int, float, float, Table]] = [
        # id    x       y       table   face
        (  1,   2.5,   5.55,   T1),   # left  face of T1, lower  ┐ same face
        (  2,   2.5,   6.00,   T1),   # left  face of T1, upper  ┘
        (  3,   6.5,   5.75,   T1),   # right face of T1

        (  4,  14.5,   4.50,   T2),   # bottom face of T2
        (  5,  16.5,   5.75,   T2),   # right  face of T2
        (  6,  14.5,   7.00,   T2),   # top    face of T2

        (  7,   6.5,   9.50,   T3),   # bottom face of T3
        (  8,   8.5,  10.75,   T3),   # right  face of T3
        (  9,   6.5,  12.00,   T3),   # top    face of T3

        ( 10,  11.5,  10.75,   T4),   # left   face of T4
        ( 11,  12.5,  12.00,   T4),   # top    face of T4, left  ┐ same face
        ( 12,  14.5,  12.00,   T4),   # top    face of T4, right ┘
    ]
    # fmt: on
    exhibits = [Exhibit(exhibit_id=i, x=x, y=y, table=t) for i, x, y, t in raw]
    validate_exhibits(exhibits, margin=0.5)  # hard guard — raises if violated

    # ── Neighbourhood sets (radius = 6 m) ─────────────────────
    neighbourhoods = build_neighbourhoods(exhibits, radius=6.0)

    # ── Start exhibits: y ≤ 6 m (nearest to entry) ────────────
    start_exhibits = [e for e in exhibits if e.y <= 6.0]

    return MuseumConfig(
        tables=tables,
        exhibits=exhibits,
        neighbourhoods=neighbourhoods,
        start_exhibits=start_exhibits,
        start_area=(7.0, 13.0, 0.0, 1.0),  # 6 m-wide central patch
        boundary=(0.0, 20.0, 0.0, 14.0),
        sr=12.0,
        r_stop=0.50,
        sigma_theta_walk=0.70,
        sigma_theta_approach=0.5,
        sigma_theta_stop=0.60,
        w1=0.30,
        K=8,
        t_max=1_800.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

_PHASE_COLORS = {WALK: "#4C72B0", APPROACH: "#DD8452", STOP: "#55A868"}


def plot_trajectory(
    traj: List[TrajectoryPoint],
    cfg: MuseumConfig,
    *,
    title: str = "Simulated Visitor Trajectory",
    figsize: Tuple[float, float] = (13.0, 9.0),
):
    """
    Plot the trajectory overlaid on the museum floor-plan.

    Colour coding
    -------------
    Blue   = WALK
    Orange = APPROACH
    Green  = STOP
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=figsize)

    # Museum boundary
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

    # Start area
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

    # Tables
    for t in cfg.tables:
        poly = MplPolygon(
            t.vertices,
            closed=True,
            facecolor="#C4A882",
            edgecolor="#7A5C2E",
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(poly)
        cx, cy = t.vertices.mean(axis=0)
        ax.text(
            cx,
            cy,
            f"T{t.table_id}",
            ha="center",
            va="center",
            fontsize=7,
            color="#4A3520",
            fontweight="bold",
            zorder=3,
        )

    # Exhibits
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

    # Trajectory — colour-code each step by its phase
    if traj:
        xs = [p.x for p in traj]
        ys = [p.y for p in traj]
        phases = [p.phase for p in traj]

        for i in range(len(xs) - 1):
            ax.plot(
                [xs[i], xs[i + 1]],
                [ys[i], ys[i + 1]],
                color=_PHASE_COLORS[phases[i]],
                linewidth=0.9,
                alpha=0.75,
                zorder=4,
            )

        ax.plot(
            xs[0],
            ys[0],
            "o",
            color="lime",
            markersize=10,
            zorder=10,
            label="Start",
            markeredgecolor="black",
        )
        ax.plot(
            xs[-1],
            ys[-1],
            "*",
            color="red",
            markersize=14,
            zorder=10,
            label="End",
            markeredgecolor="black",
        )

    # Phase legend
    phase_patches = [
        mpatches.Patch(color=c, label=ph) for ph, c in _PHASE_COLORS.items()
    ]
    start_end_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=phase_patches + start_end_handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
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


def plot_speed_distribution(traj: List[TrajectoryPoint]):
    """Diagnostic: overlay empirical speed histogram on the Beta(2,10.1) pdf."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    walk_pts = [p for p in traj if p.phase == WALK]
    if len(walk_pts) < 2:
        return None

    dt = 1.0 / 12.0
    speeds = []
    for i in range(1, len(walk_pts)):
        dx = walk_pts[i].x - walk_pts[i - 1].x
        dy = walk_pts[i].y - walk_pts[i - 1].y
        speeds.append(math.hypot(dx, dy) / dt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        speeds,
        bins=50,
        density=True,
        alpha=0.6,
        color="#4C72B0",
        label="Simulated WALK speeds",
    )
    xs = np.linspace(0, 3, 300)
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Density")
    ax.set_title("WALK Phase Speed Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point — quick demo
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  Command-line entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(
        description="Museum visitor trajectory simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num", type=int, default=1, help="Number of trajectories to generate"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Base random seed (trajectory i uses seed+i)",
    )
    parser.add_argument(
        "-o", "--outdir", default=".", help="Output directory for .png files"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    museum = make_demo_museum()

    for i in range(args.num):
        seed = args.seed + i
        sim = Simulator(museum, seed=seed)
        traj = sim.simulate()

        # ── Summary ───────────────────────────────────────────
        t_total = traj[-1].t if traj else 0.0
        n_pts = len(traj)
        by_phase = {
            ph: sum(1 for p in traj if p.phase == ph) for ph in (WALK, APPROACH, STOP)
        }

        print(
            f"[{i+1}/{args.num}] seed={seed}  "
            f"t={t_total:.1f}s  pts={n_pts}  "
            + "  ".join(f"{ph}={by_phase[ph]}" for ph in (WALK, APPROACH, STOP))
        )

        # ── Save trajectory plot ───────────────────────────────
        # Single trajectory  →  trajectory_<seed>.png
        # Multiple           →  trajectory_001.png, trajectory_002.png, …
        if args.num == 1:
            fname = f"trajectory_seed{seed}.png"
        else:
            fname = f"trajectory_{i+1:04d}_seed{seed}.png"

        out_path = os.path.join(args.outdir, fname)
        fig = plot_trajectory(traj, museum, title=f"Visitor Trajectory — seed {seed}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"         saved → {out_path}")

    print("Done.")
