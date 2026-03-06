
# Simulate Trajectories

## Given

1. Tables $TAB = [T_1, \ldots, T_n]$.
Each $T_i$ has corner vertices defining its boundary, stored in fixed counter-clockwise order.
2. Exhibits $EXH = [E_1, \ldots, E_m]$.
Each $E_j$ has coordinates $(x_{E_j}, y_{E_j})$ and a table membership $T_{E_j} \in TAB$.
3. For each exhibit $E_j$, a **neighbourhood set** $N_j \subseteq EXH$ of exhibits within a fixed radius. Repetitions are allowed across sets.
4. A **starting set** $\mathcal{E}_{start} \subseteq EXH$ of exhibits closest to the start area.
5. A **start area** defined by $[x_{min}^{start},\, x_{max}^{start},\, y_{min}^{start},\, y_{max}^{start}]$.
6. Museum boundary $\mathcal{B}$: defined by $[x_{min},\, x_{max},\, y_{min},\, y_{max}]$.

---

## Notation

1. $D(p, q):$ Euclidean distance between points $p$ and $q$
2. $sdist(\nu, L):$ signed distance from point $\nu$ to line $L$
3. $vertices(T):$ set of the corner vertices of table $T$, indexed $[\nu_0, \nu_1, \ldots]$ in counter-clockwise order
4. $vertices(T)[i]:$ $i^{th}$ element in $vertices(T)$
5. $index(\nu, T):$ the index $i$ such that $vertices(T)[i] = \nu$
6. $phase:$ "WALK", "APPROACH" or "STOP"
7. $sr:$ sampling rate of the location system. $sr = 12$ for UWB.
8. $r_{stop}:$ radius of exhibit vicinity where user wanders during $\texttt{STOP}$ phase
9. $k:$ disk sampling concentration parameter
$k=1:$ uniform over disk area
$k>1:$ more toward boundary
$k<1:$ more toward centre
10. $w_1:$ Bernoulli gate probability for disk sampling. if $w_1 < 0.5,$ outer-dense wandering.

---

## Distributions

**Speed $(v)$ perturbation:**

$v \mid phase = \begin{cases} \text{Beta}(2.0,\, 10.1) \text{ scaled to } [0.50,\, 2.60] & \texttt{WALK} \\ \text{Uniform}(0.3,\, 0.7) & \texttt{APPROACH} \\ \text{Uniform}(0.1,\, 0.25) & \texttt{STOP} \end{cases}$

**Heading direction $(\theta)$ perturbation:**

$\Delta\theta \mid phase = \begin{cases} \mathcal{N}(0,\, \sigma_{\theta,w}) & \texttt{WALK} \\ \mathcal{N}(0,\, \sigma_{\theta,a}) & \texttt{APPROACH} \\ \mathcal{N}(0,\, \sigma_{\theta,s}) & \texttt{STOP} \end{cases}$

**Bernoulli gates (determine how freqeuntly $v,$ $\theta$ change):**

$p_v \mid phase = \begin{cases} 0.5 & \texttt{WALK} \\ 0.3 & \texttt{APPROACH} \\ 0.66 & \texttt{STOP} \end{cases} \qquad p_\theta \mid phase = \begin{cases} 0.5 & \texttt{WALK} \\ 0.3 & \texttt{APPROACH} \\ 0.75 & \texttt{STOP} \end{cases}$

**Dwell time:**

$\tau_E \sim \text{Beta}(1.9,\, 4.0) \text{ scaled to } [11,\, 180] \text{ seconds} \qquad \text{(mode} \approx 50\text{s)}$

**Start position:**

$x \sim \text{Uniform}(x_{min}^{start},\, x_{max}^{start}), \quad y \sim \text{Uniform}(y_{min}^{start},\, y_{max}^{start})$

**Exhibit selection:**

$E \sim \begin{cases} \text{Uniform}(\mathcal{E}_{start}) & \text{initial exhibit} \\ \text{Uniform}(N_{E_{prev}} \setminus E_{visited}) & \text{if } N_{E_{prev}} \setminus E_{visited} \neq \emptyset \\ \text{Uniform}(EXH \setminus E_{visited}) & \text{otherwise (fallback)} \end{cases}$

---

## Algorithm

### def &nbsp; $\text{Move}(S(x_s,y_s),\, P(x_p, y_p),\, phase)$

1. <span style="color:green">Sample $g_v \sim \text{Bernoulli}(p_v \mid phase),\quad g_\theta \sim \text{Bernoulli}(p_\theta \mid phase)$</span>
2. **if** $g_v = 1:$ &nbsp; <span style="color:green">sample $v \sim (v \mid phase)$</span>
3. **if** $g_\theta = 1:$ &nbsp; <span style="color:green">sample $\Delta\theta \sim (\Delta\theta \mid phase)$</span>; &nbsp; **else:** $\Delta\theta := 0$
4. $\theta := \arctan\!\left(\tfrac{y_p - y_s}{x_p - x_s}\right) + \Delta\theta$
5. $x := x_s + \left(\dfrac{v}{sr}\cdot\cos(\theta)\right),\quad y := y_s + \left(\dfrac{v}{sr}\cdot\sin(\theta)\right)$
6. $S := (x,\, y)$
7. $t := t + \dfrac{1}{sr}$

*pending: museum boundary check*

---

## $\text{MoveAroundObstacle}(S,\, E,\, T_E,\, L)$

1. $\nu^* := \arg\min_{\nu \,\in\, vertices(T_E)} D(\nu,\, S)$ $\texttt{\# closest table vertex to S}$
2. $a := sdist(\nu^*, L)$ $\texttt{\# closest table vertex to L}$
$\texttt{\#} \;V':\; \texttt{vertices that lie on the same side of L}:$
3. **if** $a > 0:$ &nbsp; $V' := \{\nu' \in vertices(T_E) \setminus \{\nu^*\} \mid sdist(\nu', L) > 0\}$
4. **if** $a < 0:$ &nbsp; $V' := \{\nu' \in vertices(T_E) \setminus \{\nu^*\} \mid sdist(\nu', L) < 0\}$
5. $V := V' \cup \{\nu^*\}$
6. $i := index(\nu^*,\, T_E)$
7. $i_{next} := (i + 1) \bmod |vertices(T_E)|$ $\texttt{\# cyclic index}$
$\texttt{\#}\;\delta:\; \texttt{direction of the vertex indices to visit}:$
8. **if** $vertices(T_E)[i_{next}] \in V:$ &nbsp; $\delta := +1$ &nbsp; **else:** &nbsp; $\delta := -1$ 
9. **while** $vertices(T_E)[i] \in V$: $\texttt{\# visit the vertices in the sequence}$
    1. $\nu_i := vertices(T_E)[i]$
    2. **while** $D(S,\, \nu_i) \geq 0.05:$ &nbsp; $\text{Move}(S,\, \nu_i,\, \texttt{WALK})$
    3. $i := (i + \delta) \bmod |vertices(T_E)|$
<!-- 10. $\theta := \arctan\!\left(\tfrac{y_E - y_S}{x_E - x_S}\right)$ -->

---

## $\text{MoveToExhibit}(S,\, E)$

1. $phase := \texttt{WALK}$
2. $\tau_E := 9999$
3. $D := D(S, E)$
4. $excurse := 0$
5. $excurse\_start := 0$
6. $excursing := \text{False}$
7. $\theta := \arctan\!\left(\dfrac{y_E - y_S}{x_E - x_S}\right)$
<!-- 8. <span style="color:green">$v \sim \text{Beta}(2.0,\, 10.1)$ scaled to $[0.50,\, 2.60]$</span> -->
8. $r_{stop} = 0.5$

**while** $t \leq \tau_E:$

1. Draw line $L$ from $S$ to $E$.

2. **if** $L$ does not intersect $T_E$:
    1. **if** $phase = \texttt{STOP}$:
        1. **if** $D(S, E) \leq r_{stop}:$ $\texttt{\# walk around exhibit}$
            1. <span style="color:green">sample $\varphi \sim \text{Uniform}(0,\, 2\pi)$</span>
            2. <span style="color:green">sample $\rho \sim \text{Uniform}(0,\, 1)$</span>
            3. <span style="color:green">sample $g_k \sim \text{Bernoulli}(w_1)$</span>
            4. **if** $g_k = 1:$ &nbsp; $k = -0.66$ &nbsp; **else:** &nbsp; $k = 4$
            5. $r := r_{stop} \cdot {\rho}^{\frac{1}{k+1}}$
            6. $P_w := \bigl(x_E + r\cos\varphi,\; y_E + r\sin\varphi\bigr)$
            7. $\text{Move}(S,\, P_w,\, \texttt{STOP})$
        2. **else:** &nbsp; $\text{Move}(S,\, E,\, \texttt{STOP})$
    2. **else:** &nbsp; $\text{Move}(S,\, E,\, phase)$ 
    3. $D := D(S, E)$
    4. **if** $\bigl(phase = \texttt{STOP}\bigr)$ **and** $\bigl(D > r_{stop}\bigr) $ **and** $\bigl(excursing = \text{False}\bigr):$ $\texttt{\# first excursion}$
        1. $excursing := \text{True}$
        2. $excurse\_start := t$
    5. **if** $\bigl(excursing = \text{True}\bigr)$ **and** $\bigl(D \leq r_{stop}\bigr):$ $\texttt{\# return from excursion}$
        1. $excurse := excurse + (t - excurse\_start)$
        2. $excursing := \text{False}$
    6. **if** $\bigl(excursing = \text{True}\bigr)$ **and** $\bigl(excurse + (t - excurse\_start) > 15\bigr):$ $\texttt{\# excursion limit}$
        1. &nbsp; **return**
    7. **if** $\bigl(D < r_{stop}*2\bigr)$ **and** $\bigl(phase = \texttt{WALK}\bigr):$
        1. $phase := \texttt{APPROACH}$
    8. **if** $\bigl(D < r_{stop}\bigr)$ **and** $\bigl(phase = \texttt{APPROACH}\bigr):$
        1. $phase := \texttt{STOP}$
        2. <span style="color:green">$\Delta\tau \sim \text{Beta}(1.9,\, 4.0)$ scaled to $[11,\, 180]$</span> $\texttt{\# dwell time at stop}$
        3. $\tau_E := t + \Delta\tau$

3. **else if** $L$ intersects $T_E:$
    1. $\text{MoveAroundObstacle}(S,\, E,\, T_E,\, L)$

---

## main &nbsp; $\text{SimulateTrajectory}$

1. $E_{visited} := \emptyset$
2. $t := 0$
$\texttt{\# get random starting point from a starting area:}$
3. <span style="color:green">$x \sim \text{Uniform}(x_{min}^{start},\, x_{max}^{start}),\quad y \sim \text{Uniform}(y_{min}^{start},\, y_{max}^{start})$</span>; &nbsp; $S := (x, y)$
$\texttt{\# get a random exhibit from exhibits near to starting area}$
4. <span style="color:green">$E \sim \text{Uniform}(\mathcal{E}_{start})$</span>
5. $E_{visited} := E_{visited} \cup \{E\}$; &nbsp; $E_{prev} := E$
6. Sample $v \sim (v \mid \texttt{WALK})$

**while** *termination\_criteria*:

1. $\text{MoveToExhibit}(S,\, E)$
$\texttt{\# pick exhibit from non-empty neighoubor exhibit set:}$
2. **if** $N_{E_{prev}} \setminus E_{visited} \neq \emptyset:$ &nbsp; <span style="color:green">$E \sim \text{Uniform}(N_{E_{prev}} \setminus E_{visited})$</span>
$\texttt{\# pick exhibit from all exhibits set:}$
3. **else if** $EXH \setminus E_{visited} \neq \emptyset:$ &nbsp; <span style="color:green">$E \sim \text{Uniform}(EXH \setminus E_{visited})$</span>
4. **else:** &nbsp; **break**
5. $E_{visited} := E_{visited} \cup \{E\}$; &nbsp; $E_{prev} := E$

***termination\_criteria***:

1. $|E_{visited}| \geq K$ &nbsp; $\texttt{\# `K' exhibits visited}$
2. $t \geq t_{max}$ $\texttt{\# trajectory max duration reached}$
3. $EXH \setminus E_{visited} = \emptyset$ &nbsp; $\texttt{\# all exhibits visited}$
