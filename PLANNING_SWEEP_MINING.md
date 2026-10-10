# PLANNING Sweep — External Mining Report

Module: `nomorals/planning/` (5 files: `__init__`, `graph`, `route`, `estimates`, `congestion`).
Date: 2026-10-10. Written BEFORE any implementation, per sweep method.

Every significant class was compared against the best implementations outside
this repo. "Best" = mined for gold to merge. "Trash" = mined to learn what NOT
to do. Each section ends with the gaps this sweep will fill.

---

## 1. `WorldGraph` — live world graph, dependency + disruption propagation

### How the best do it

**Neo4j Graph Data Science (official docs).** The canonical toolkit for exactly
this job — dependency DAGs analyzed as graphs:
- `gds.dag.longestPath` — longest weighted path in a DAG (their "critical
  path"); weighted by an edge property, streamed per source node. Ours is an
  unweighted DFS version with a silent cycle-break. The gold we lack is not the
  algorithm but the *surrounding* analyses GDS ships next to it:
- **Centrality**: degree / betweenness / PageRank answer "which nodes are the
  most influential or critical" — nodes lying on many shortest paths are
  bottlenecks/bridges. Neo4j's guide explicitly frames betweenness as
  "highlighting bottlenecks or bridges" and the topological-sort doc notes that
  nodes at the same maximal distance from sources "have no dependencies between
  them, and can be scheduled in parallel".
- **Cycle detection**: Neo4j's topological sort is *also* a cycle detector —
  "if some of the nodes are missing from the sorting, there is a cycle." Our
  `schedule_order()` buries cycles in an appended leftover; our
  `critical_path()` silently breaks them in DFS. Neither *reports* them.

Sources: https://neo4j.com/docs/graph-data-science/current/algorithms/dag/longest-path/,
https://neo4j.com/docs/graph-data-science/current/algorithms/dag/topological-sort/,
https://neo4j.com/blog/graph-data-science/graph-algorithms/

**Supply-chain graph analysis (vectree.io).** Dependency graphs are mined for
*structural vulnerability*, not just reachability:
- **Articulation points / bridges**: nodes/edges whose removal increases the
  number of components — "single points of failure". Identified via Tarjan's
  DFS (also in CXXGraph's `tarjan()`, Goodrich & Tamassia biconnectivity
  slides). Our graph has *no* single-point-of-failure analysis at all.
- **Hub analysis via eigenvector/degree centrality**: hubs are "scale-free
  targets — their failure is catastrophic compared to the failure of a
  peripheral node". Our `dependents()` gives a blast radius *given a failure*,
  but nothing says *which node you should protect* proactively.

Source: https://vectree.io/pdf/c/supply-chain-graph-analysis

**Workflow-intelligence builds (dmerrimon/seleen INTELLIGENCE_ARCHITECTURE.md,
GitHub).** A real repo doing this exact job (study-workflow graph) exposes as
API endpoints: critical path, bottlenecks (high in+out degree, "delays here
have maximum cascade impact"), cascade impact from a node, resource contention,
and — the one we completely lack — **shortest path between two milestones**
(`GET /graph/path/:fromId/:toId`): "how is X connected to Y".

Source: https://github.com/dmerrimon/seleen/blob/HEAD/INTELLIGENCE_ARCHITECTURE.md

**What-if simulation.** The PERT Monte Carlo tools (see §3) never mutate state
— every what-if is a simulation. Our `mark_disrupted()` *writes* to the DB;
`what_breaks_if()` only returns a flat summary. There is no dry-run that
renders the *shape* of the blast radius (depth layers) without touching state.

### The gaps this sweep fills (graph.py)

1. `weak_links()` — betweenness-style "load-bearing node" ranking on the
   undirected dependency projection (Brandes' algorithm, dependency-normalized)
   plus fan-in/fan-out. Answers "what should I protect".
2. `find_cycles()` — explicit cycle reporting (iterative DFS, canonical
   rotation) instead of silent breaking. Surfaced in `/graph cycles` and in
   `schedule_order`'s leftover.
3. `path_between(a, b)` — BFS shortest dependency path: "how is the Lagos
   flight connected to my 2pm".
4. `neighborhood(node_id, depth)` — k-hop subgraph for focused views.
5. `to_mermaid()` — `graph LR` export for GodConsole/chat rendering.
6. `schedule_waves()` — topological layers: tasks at the same depth run in
   parallel (Neo4j's maximal-distance insight, implemented in pure Python).
7. `simulate_disruption()` — layered blast-radius tree as a dry run that never
   writes state.

---

## 2. `RoutePlanner` / `RouteSolver` / `CostModel` / `GeoIndex` — predict→build→solve

### How the best do it

**TSP heuristics literature.** The numbers that matter (dev.to survey;
TSPLIB eil51 benchmark by mertatmacadev/tsp-heuristic-analysis):
- Nearest-neighbor greedy: **~25% over optimal** (13.1% gap on eil51).
- Nearest-neighbor **+ 2-opt**: **~5% typical, 2.6% gap on eil51** —
  a ~10-iteration O(n²) local search that uncrosses edge pairs. Sub-50ms for
  n ≤ 60.
- OR-Tools PATH_CHEAPEST_ARC first solution + local search is the production
  default; Concorde is exact but unbounded time.

Our `RouteSolver` runs *raw* greedy when OR-Tools is absent (termux default,
and OR-Tools is almost never installed on a phone) — i.e. we routinely hand
the user a route ~25% worse than a 10ms 2-opt pass would give. **The single
highest-ROI change in this module is NN + 2-opt as the improved greedy.**

Sources: https://dev.to/malcolmlow/optimization-problems-explained-traveling-salesman-job-shop-scheduling-and-the-knights-tour-5d4i,
https://github.com/mertatmacadev/tsp-heuristic-analysis,
https://github.com/mdaiyan-dev/supply-chain-route-optimizer

**OR-Tools VRPTW (Google docs + routing-modeling skill).** The production
pattern for time windows is a **Time dimension**: travel + service time in the
transit callback, `CumulVar(index).SetRange(window)` per stop, waiting as
slack. Our solver currently *ignores* `Stop.window` except to print violations
after the fact — it confidently promises infeasible routes. The OR-Tools path
should add the Time dimension when any stop has a window; the greedy path
should become window-aware (cheapest insertion penalizing lateness).

Sources: https://developers.google.com/optimization/routing/dimensions,
https://github.com/elsiedai/skill-repair-benchmark/blob/HEAD/evaluation/data/core25/references/paratransit-routing/skills/ortools-routing-modeling/SKILL.md

**Uber H3.** The real library exposes `h3.k_ring` / `k_ring_distances` for
expanding a cell to its neighbors — the correct way to do "nearby" without
scanning the whole index. Our `GeoIndex.nearby()` iterates *every* cell bucket
(the thing the module docstring says it never does) and only narrows via the
fallback grid's adjacency. With real h3 installed we should use k-ring
expansion; the fallback keeps the grid path.

**Cost-model freshness.** Every serious learning system timestamps
observations: stale actuals mislead. Our `route_actuals` table has no
timestamp, so a road that was fast in 2024 counts forever. Gold: `recorded_at`
+ age-decay on the EMA (or a `forget_before` API), and honest reporting of
"learned from N recent trips".

### The gaps this sweep fills (route.py)

1. `RouteSolver._solve_greedy` → NN **+ 2-opt** (best-improvement, bounded
   passes, pure Python, works on termux).
2. Window-aware solving: greedy becomes window-penalized cheapest insertion;
   OR-Tools path adds a Time dimension with per-stop windows when present.
3. `return_to_origin` option (round trips — errands that end at home).
4. `GeoIndex.nearby` uses `h3.k_ring` expansion when the real library is
   present; fallback grid path unchanged.
5. `record_actual` timestamps observations; EMA update decays stale pairs;
   `stats()` reports recent-trip counts honestly.

---

## 3. `EstimateStore` / PERT / `aggregate` — honest time bands

### How the best do it

**Monte Carlo over Beta-PERT (the industry standard for schedule risk).**
Tools like everydaybudd's PERT Project Risk Calculator, patrickdls'
timeshifted-risk-mcs, and evgenivinogradov's Monte Carlo simulator all do the
same thing our `aggregate()` *approximates* with ±2σ normal math:
- Sample each task's **Beta-PERT** distribution (the standard
  three-point distribution: α = 1+4(M−O)/(P−O), β = 1+4(P−M)/(P−O)).
- Per iteration, the project duration = the **max over all paths** of sampled
  durations (the *simulated* critical path — note the deterministic critical
  path is wrong ~half the time).
- Output **P10/P50/P80/P90** ("quote P80 to the stakeholder"), plus per-task
  **criticality index** = fraction of iterations where the task lies on the
  critical path, plus **tornado/sensitivity** ranking of variance drivers, plus
  contingency = P80 − P50.

Our `aggregate()` assumes normal ±2σ and sums only the deterministic critical
path — it misses the "near-critical path overtakes" effect entirely, and it
can never name *which step* drives the risk. The canonical validation gate
(javiermontano-sofka SKILL.md): "minimum 10,000 iterations… confidence levels
clearly presented… sensitivity analysis identifies top 5 variance drivers…
results interpretable — S-curves, not just tables."

Sources: https://www.everydaybudd.com/tools/data-ops/project-monte-carlo-risk,
https://github.com/patrickdls/timeshifted-risk-mcs,
https://github.com/javiermontano-sofka/sofka-discovery-framework-public/blob/HEAD/sdf/skills/monte-carlo-simulation/SKILL.md,
https://github.com/evgenivinogradov/monte-carlo-simulation

**Reference-class forecasting (Kahneman/Tversky/Flyvbjerg).** The outside view
beats the inside view: UK Treasury Green Book *mandates* optimism-bias uplifts;
Flyvbjerg's 258-project audit found 9/10 over budget. Operational form: anchor
the estimate to the empirical distribution of *similar past projects*, then
apply a percentile uplift. Our `EstimateStore` *has* the outside view (the
EMA bias ratio per task type) but applies it silently inside `estimate()` —
the user never sees "your inside view says 20m; your history says ×1.6".
Surfacing the comparison as an explicit check is the gold.

Sources: http://arXIV.org/pdf/1302.3642 (Flyvbjerg 2006 PMI paper),
https://github.com/catcorner22/claude_skills_2/blob/HEAD/plugins/decision-science-skills/skills/reference-class-forecasting/SKILL.md,
https://github.com/stepowskieric/grimoirestack/blob/HEAD/app/public/skills/judgment-and-routing/reference-class-forecasting/SKILL.md

**Calibration (Tetlock / Superforecasting).** A forecaster who quotes 80%
confidence should be right ~80% of the time. We track hits (in-band) vs misses
per task type but never compute the implied quote vs reality. The gold is a
`calibration()` report: quoted band-hit-rate vs observed hit-rate per task
type, with an explicit "your bands are over/under-confident" verdict.

### The gaps this sweep fills (estimates.py)

1. `monte_carlo(parts, n=10000)` — Beta-PERT sampling (stdlib
   `random.betavariate` with the canonical Vose α/β), per-iteration simulated
   critical path over the dependency DAG, P10/P50/P80/P90, per-step criticality
   index, sensitivity ranking by rank-correlation with total, honest seed +
   iteration count. Pure stdlib, termux-safe (n scales down on small graphs).
2. `reference_class_check(task_type, inside_minutes)` — explicit outside-view
   uplift: learned bias ratio + empirical miss-rate → "quote ×1.6 / add P80".
3. `calibration()` — per-task-type quoted-vs-observed hit rates.
4. `/eta sim <type> <o/m/p>...` chat entry rendering P50/P80/P90 + drivers.

---

## 4. `ContentionMonitor` — multi-agent congestion prediction

### How the best do it

**Distributed locking: a lock is a lease (Redis/Redlock literature).**
The production consensus (dev.to serifcolakel; kosokodaniel; lynricsy
hyperskills Redis skill):
- `SET key token NX EX ttl` — the TTL is the **safety net against the dead
  owner**: "a crashed process can leave the system permanently locked… The
  TTL solves the dead-owner problem."
- The trade-off is explicit: no TTL → dead worker blocks everyone forever;
  TTL → a live-but-slow worker can become stale. Mitigation: unique tokens +
  compare-and-delete release, and watchdogs that renew while work continues.
- Our holds have **no expiry at all**: an agent that crashes between
  `acquire()` and `release()` leaves `queue_depth()` inflated forever, and
  every future `predict_jam`/`advise` on that resource is polluted. This is
  the single biggest correctness gap in the module.

Sources: http://dev.to/serifcolakel/distributed-locks-in-go-correctness-failure-modes-and-production-patterns-4mdg,
https://medium.com/@kosokodaniel/distributed-locks-are-not-enough-understanding-leases-fencing-tokens-and-idempotency-0262f0440825,
https://github.com/lynricsy/hyperskills/blob/HEAD/skills/redis/SKILL.md

**Circuit breaker (Resilience4j — the Hystrix successor).** Three states,
count-based sliding window, no direct OPEN→CLOSED:
- CLOSED: normal, counting failures against `failureRateThreshold` over the
  last N calls (with `minimumNumberOfCalls` so one early failure can't trip).
- OPEN: fail fast to fallback for `waitDurationInOpenState`.
- HALF-OPEN: limited probes; success closes, failure re-opens.
- Also trips on *slow* calls (`slowCallRateThreshold`), not just failures.

Our `advise()` is stateless: it re-derives everything per call and has no
memory of "this resource has been jammed for the last 20 minutes". The gold
is a per-resource breaker driven by our jam predictions as the failure
signal, persisted so it survives across advise() calls.

Sources: https://github.com/pradhansoumik/microservice-design-patterns/blob/HEAD/01-circuit-breaker/CB_INTERVIEW-REVISION-NOTES.md,
https://github.com/harman-04/spring-boot-resilience-circuit-breaker,
https://github.com/jacksosa/jacksosa.github.io/blob/HEAD/_posts/2025-07-18-resilience4j-spring-boot-circuit-breaker.md

**Queueing theory: Little's law.** L = λW — expected wait = queue depth /
service rate. We already measure both (depth via holds, service rate via
release events in `_rates()`), but `advise()` never converts them into the
one number an orchestrator actually wants: "expect ~Xs before a slot frees".
The gold is an honest wait estimate in the advice detail.

### The gaps this sweep fills (congestion.py)

1. **Hold leases**: `acquire(..., ttl_s=600)` stores `expires_at`;
   `queue_depth()`/`status()` count live holds only; `reap_stale()` purges
   expired holds (+ event log entry). `hold()` context manager accepts a TTL.
2. **Circuit breaker per resource** (closed/open/half-open, count-based window
   over jam observations, cooldown, probe): `breaker_state(name)`,
   `advise()` short-circuits to backoff when OPEN, HALF-OPEN admits a probe.
   Backed by the DB so it persists.
3. **Expected wait** via Little's law in `advise()` detail ("~45s per slot").
4. `status()` gains breaker state + oldest-hold age (stale-hold visibility).

---

## Cross-cutting: what the sweep will NOT do

- No new external dependencies. Beta-PERT via `random.betavariate` (stdlib),
  Brandes betweenness in pure Python, 2-opt in pure Python, Time dimension
  only inside the existing optional OR-Tools path. numpy stays out — the
  phone is the deployment target.
- No deletion of classes, modules, or tools. Additive gaps + upgrades only.
- Every public method keeps its never-raises contract.
- Chat controls gain subcommands; existing ones keep their exact behavior.
