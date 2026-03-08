# Simulate Trajectories

## Given

1. Tables $TAB = [T_1, \ldots, T_n]$.
Each $T_i$ has corner vertices defining its boundary, stored in fixed counter-clockwise order.
2. Exhibits $EXH = [E_1, \ldots, E_m]$.
Each $E_j$ has coordinates $(x_{E_j}, y_{E_j})$ and a table membership $T_{E_j} \in TAB$.
3. For each exhibit $E_j$, a **neighbourhood set** $N_j \subseteq EXH$ of exhibits within a fixed radius. Repetitions are allowed across sets.
4. A **starting set** $\mathcal{E}_{start} \subseteq EXH$ of exhibits closest to the start area.
5. A **start area** defined by $[x_{min}^{start}, x_{max}^{start}, y_{min}^{start}, y_{max}^{start}]$.
6. Museum boundary $\mathcal{B}$: defined by $[x_{min}, x_{max}, y_{min}, y_{max}]$.

---

## Notation

1. $D(p, q):$ Euclidean distance between points $p$ and $q$
2. $sdist(\nu, L):$ signed distance from point $\nu$ to directed line $L$ — positive if $\nu$ is to the **left** of $L$ (counter-clockwise side), negative if to the right
3. $distToTable(S, T):$ minimum Euclidean distance from point $S$ to the closest point on the **boundary** of table $T$; returns $0$ if $S$ is inside $T$
4. $intersects(L, T):$ true iff segment $L$ properly crosses at least one edge of $T$ (endpoint touches excluded)
5. $vertices(T):$ set of the corner vertices of table $T$, indexed $[\nu_0, \nu_1, \ldots]$ in counter-clockwise order
6. $vertices(T)[i]:$ $i^{th}$ element in $vertices(T)$
7. $index(\nu, T):$ the index $i$ such that $vertices(T)[i] = \nu$
8. $phase:$ "WALK", "APPROACH" or "STOP"
9. $sr:$ sampling rate of the location system. $sr = 12$ for UWB.
10. $r_{stop}:$ radius of exhibit vicinity where user wanders during $\texttt{STOP}$ phase
11. $k:$ disk sampling concentration parameter.
$k < 1:$ more toward centre.
$k > 1:$ more toward boundary.
12. $w_1:$ Bernoulli gate probability for disk sampling. if $w_1 < 0.5,$ outer-dense wandering.

---

## Distributions

**Speed $(v)$ perturbation:**

$v \mid phase = \begin{cases} \text{Beta}(2.0, 10.1) \text{ scaled to } [0.50, 2.60] & \texttt{WALK} \\ \text{Uniform}(0.3, 0.7) & \texttt{APPROACH} \\ \text{Uniform}(0.1, 0.25) & \texttt{STOP} \end{cases}$

**Heading direction $(\theta)$ perturbation:**

$\Delta\theta \mid phase = \begin{cases} \mathcal{N}(0, \sigma_{\theta,w}) & \texttt{WALK} \\ \mathcal{N}(0, \sigma_{\theta,a}) & \texttt{APPROACH} \\ \mathcal{N}(0, \sigma_{\theta,s}) & \texttt{STOP} \end{cases}$

**Bernoulli gates (determine how frequently $v,$ $\theta$ change):**

$p_v \mid phase = \begin{cases} 0.5 & \texttt{WALK} \\ 0.3 & \texttt{APPROACH} \\ 0.66 & \texttt{STOP} \end{cases} \qquad p_\theta \mid phase = \begin{cases} 0.5 & \texttt{WALK} \\ 0.3 & \texttt{APPROACH} \\ 0.75 & \texttt{STOP} \end{cases}$

**Dwell time:**

$\tau_E \sim \text{Beta}(1.9, 4.0) \text{ scaled to } [11, 180] \text{ seconds} \qquad \text{(mode} \approx 50\text{s)}$

**Start position:**

$x \sim \text{Uniform}(x_{min}^{start}, x_{max}^{start}), \quad y \sim \text{Uniform}(y_{min}^{start}, y_{max}^{start})$

**Exhibit selection:**

$E \sim \begin{cases} \text{Uniform}(\mathcal{E}_{start}) & \text{initial exhibit} \\ \text{Uniform}(N_{E_{prev}} \setminus E_{visited}) & \text{if } N_{E_{prev}} \setminus E_{visited} \neq \emptyset \\ \text{Uniform}(EXH \setminus E_{visited}) & \text{otherwise (fallback)} \end{cases}$

---

## Algorithm

### def &nbsp; $\text{Move}(S(x_s,y_s), P(x_p, y_p), phase)$

1. <span style="color:green">**if** $\text{Bernoulli}(p_v \mid phase) = 1:$</span> &nbsp; <span style="color:green">sample $v \sim (v \mid phase)$</span>
2. <span style="color:green">**if** $\text{Bernoulli}(p_\theta \mid phase) = 1:$</span> &nbsp; <span style="color:green">sample $\Delta\theta \sim (\Delta\theta \mid phase)$</span>; &nbsp; **else:** $\Delta\theta := 0$
3. $\theta := \text{atan2}(y_p - y_s,\; x_p - x_s) + \Delta\theta$
4. $x' := x_s + \dfrac{v}{sr}\cdot\cos(\theta), \qquad y' := y_s + \dfrac{v}{sr}\cdot\sin(\theta)$
5. $S := \bigl(\text{clamp}(x', x_{min}, x_{max}),\; \text{clamp}(y', y_{min}, y_{max})\bigr)$ $\texttt{\# project back inside } \mathcal{B}$
6. $t := t + \dfrac{1}{sr}$
7. Record sample $(t,\; x_S,\; y_S,\; phase)$

> **Notes:** step 3 uses $\text{atan2}(y, x)$, not $\arctan(y/x)$, so heading is correct in all four quadrants.
> Step 5 clamps the new position to the museum boundary $\mathcal{B}$ — the agent slides along walls rather than leaving the space.

---

## $\text{MoveAroundObstacle}(S, E, T_{blocking}, L)$

> $T_{blocking}$ is the table blocking segment $L$

1. Compute $sdist(\nu_i, L)$ for every $\nu_i \in vertices(T_{blocking})$
$\texttt{\# classify vertices by which side of } L \texttt{ they fall on:}$
2. $V^+ := \{\nu_i \mid sdist(\nu_i, L) > 0\}$
3. $V^- := \{\nu_i \mid sdist(\nu_i, L) < 0\}$
$\texttt{\# pick the smaller partition: fewest vertices -> shortest detour:}$
4. **if** $|V^+| < |V^-|:$ &nbsp; $V := V^+$
5. **else if** $|V^-| < |V^+|:$ &nbsp; $V := V^-$
6. **else:** &nbsp; $V :=$ whichever of $V^+, V^-$ contains the vertex nearest to $S$ $\texttt{\# tie-break}$
7. **if** $V = \emptyset:$ &nbsp; **return** $\texttt{\# L passes exactly through vertices, no detour}$
8. $\nu_{start} := \arg\min_{\nu \in V} D(\nu, S)$ $\texttt{\# start from V-vertex closest to S}$
9. $i := index(\nu_{start}, T_{blocking})$
10. **if** $vertices(T_{blocking})[(i + 1) \bmod |vertices(T_{blocking})|] \in V:$ &nbsp; $\delta := +1$ &nbsp; **else:** &nbsp; $\delta := -1$ $\texttt{\# CCW or CW}$
11. **while** $vertices(T_{blocking})[i] \in V:$ $\texttt{\# visit waypoints in sequence}$
    1. $\nu_i := vertices(T_{blocking})[i]$
    2. **while** $D(S, \nu_i) \geq 0.05:$ &nbsp; $\text{Move}(S, \nu_i, \texttt{WALK})$
    3. $i := (i + \delta) \bmod |vertices(T_{blocking})|$

> **Smaller-partition rule (steps 4–6):** line $L$ splits the table vertices into two areas. The side with fewer vertices is the shortest way. When the line cuts it into equal vertices, ($|V^+| = |V^-|$), the side whose nearest vertex is closest to $S$ is chosen.
>
> **Direction $\delta$ (step 10):** after getting the starting vertex, the traversal direction is the one that keeps the next index inside $V$. This is to make sure the waypoints are visited in correct order without skipping. CCW: $\delta = +1$; CW: $\delta = -1$

---

## $\text{MoveToExhibit}(S, E)$

1. $phase := \texttt{WALK}$
2. $\tau_E := 9999$
3. $D := D(S, E)$
4. $excurse := 0$
5. $excurse\_start := 0$
6. $excursing := \text{False}$
7. $r_{stop} :=$ (from config)

**for** at most $\lfloor t_{max} \cdot sr \rfloor + 1$ steps:

1. **if** $t > \tau_E:$ &nbsp; **break**
2. Draw segment $L$ from $S$ to $E$.
3. $blocking := \{T \in TAB \mid intersects(L, T)\}$

4. **if** $blocking = \emptyset:$
    1. **if** $phase = \texttt{STOP}:$
        1. **if** $D(S, E) \leq r_{stop}:$ $\texttt{\# walk around exhibit}$
            1. <span style="color:green">sample $\varphi \sim \text{Uniform}(0, 2\pi)$</span>
            2. <span style="color:green">sample $\rho \sim \text{Uniform}(0, 1)$</span>
            3. <span style="color:green">sample $g_k \sim \text{Bernoulli}(w_1)$</span>
            4. **if** $g_k = 1:$ &nbsp; $k := -0.66$ &nbsp; **else:** &nbsp; $k := 4$
            5. $r := r_{stop} \cdot \rho^{\frac{1}{k+1}}$
            6. $P_w := \bigl(x_E + r\cos\varphi,\; y_E + r\sin\varphi\bigr)$
            7. $\text{Move}(S, P_w, \texttt{STOP})$
        2. **else:** &nbsp; $\text{Move}(S, E, \texttt{STOP})$
    2. **else:** &nbsp; $\text{Move}(S, E, phase)$
    3. $D := D(S, E)$
    4. **if** $\bigl(phase = \texttt{STOP}\bigr)$ **and** $\bigl(D > r_{stop}\bigr)$ **and** $\bigl(excursing = \text{False}\bigr):$ $\texttt{\# first excursion step}$
        1. $excursing := \text{True}$
        2. $excurse\_start := t$
    5. **if** $\bigl(excursing = \text{True}\bigr)$ **and** $\bigl(D \leq r_{stop}\bigr):$ $\texttt{\# return from excursion}$
        1. $excurse := excurse + (t - excurse\_start)$
        2. $excursing := \text{False}$
    6. **if** $\bigl(excursing = \text{True}\bigr)$ **and** $\bigl(excurse + (t - excurse\_start) > 15\bigr):$ $\texttt{\# excursion limit}$
        1. **return**
    7. **if** $\bigl(phase = \texttt{WALK}\bigr)$ **and** $\bigl(D < r_{stop} \cdot 2\bigr):$
        1. $phase := \texttt{APPROACH}$
    8. **if** $\bigl(phase = \texttt{APPROACH}\bigr)$ **and** $\bigl(D < r_{stop}\bigr):$
        1. $phase := \texttt{STOP}$
        2. <span style="color:green">$\Delta\tau \sim \text{Beta}(1.9, 4.0)$ scaled to $[11, 180]$</span> $\texttt{\# dwell time at stop}$
        3. $\tau_E := t + \Delta\tau$

5. **else:**
    1. $T_{nearest} := \arg\min_{T \in blocking}\; distToTable(S, T)$
    2. **if** $distToTable(S, T_{nearest}) < 0.05:$
        1. $\text{Move}(S, E, phase)$
    3. **else:**
        1. $\text{MoveAroundObstacle}(S, E, T_{nearest}, L)$
    4. $D := D(S, E)$

> **Step 5.1 — nearest blocker:** when multiple tables block $L$, the one closest to $S$ is resolved first. Then, others are resolved.
>
> **Step count cap:** each iteration advances $t$ by $1/sr$ (inside $\text{Move}$). The cap of $\lfloor t_{max} \cdot sr \rfloor + 1$ outer iterations therefore bounds the trajectory to approximately $t_{max}$ seconds and prevents infinite loops.

---

## main &nbsp; $\text{SimulateTrajectory}$

1. $E_{visited} := \emptyset$
2. $t := 0$
$\texttt{\# get random starting point from a starting area:}$
3. <span style="color:green">$x \sim \text{Uniform}(x_{min}^{start}, x_{max}^{start}),\quad y \sim \text{Uniform}(y_{min}^{start}, y_{max}^{start})$</span>; &nbsp; $S := (x, y)$
4. Record initial sample $(0,\; x,\; y,\; \texttt{WALK})$
$\texttt{\# get a random exhibit from exhibits near to starting area:}$
5. <span style="color:green">$E \sim \text{Uniform}(\mathcal{E}_{start})$</span>
6. $E_{visited} := E_{visited} \cup \{E\}$; &nbsp; $E_{prev} := E$
7. <span style="color:green">Sample $v \sim (v \mid \texttt{WALK})$</span>

**while** *termination\_criteria*:

1. $\text{MoveToExhibit}(S, E)$
$\texttt{\# pick next exhibit from unvisited neighbours of the previously visited exhibit:}$
2. **if** $N_{E_{prev}} \setminus E_{visited} \neq \emptyset:$ &nbsp; <span style="color:green">$E \sim \text{Uniform}(N_{E_{prev}} \setminus E_{visited})$</span>
$\texttt{\# fallback: pick from all remaining unvisited exhibits:}$
3. **else if** $EXH \setminus E_{visited} \neq \emptyset:$ &nbsp; <span style="color:green">$E \sim \text{Uniform}(EXH \setminus E_{visited})$</span>
4. **else:** &nbsp; **break**
5. $E_{visited} := E_{visited} \cup \{E\}$; &nbsp; $E_{prev} := E$

***termination\_criteria***:

1. $|E_{visited}| \geq K$ &nbsp; $\texttt{\# `K' exhibits visited}$
2. $t \geq t_{max}$ &nbsp; $\texttt{\# trajectory max duration reached}$
3. $EXH \setminus E_{visited} = \emptyset$ &nbsp; $\texttt{\# all exhibits visited}$

<!-- > **Neighbourhood source (step 2):** the next exhibit is drawn from $N_{E_{prev}}$ — the neighbourhood of the exhibit selected in the **previous** loop iteration. Within any given iteration, $E_{prev}$ refers to the exhibit chosen two steps back in the visit sequence; it is updated to the newly selected $E$ only at step 5, after sampling is complete. -->
