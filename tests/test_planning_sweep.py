"""Planning sweep tests — mined-then-built upgrades across graph/route/estimates/congestion.

Real tests for breakable behavior: betweenness ranking, cycle reporting,
path/neighborhood/waves/simulation, mermaid export, 2-opt improvement,
window-aware solving, round trips, cost-model freshness, Beta-PERT sampling,
Monte Carlo percentiles/criticality determinism, reference-class checks,
calibration, hold leases, circuit-breaker state machine, wait estimates.
No network, no LLM.
"""
from __future__ import annotations

import math
import os
import random
import statistics
import tempfile
import time
import unittest


# ── graph analytics ────────────────────────────────────────────────────

def _chain_graph():
    from nomorals.planning.graph import WorldGraph
    g = WorldGraph(db_path=":memory:")
    nodes = {}
    for t, label in [("asset", "Lagos flight"), ("schedule", "2pm meeting"),
                     ("commitment", "deploy"), ("deadline", "launch")]:
        n = g.add_node(t, label)
        nodes[label] = n
    g.add_edge(nodes["2pm meeting"].node_id, nodes["Lagos flight"].node_id, "depends_on")
    g.add_edge(nodes["deploy"].node_id, nodes["2pm meeting"].node_id, "depends_on")
    g.add_edge(nodes["launch"].node_id, nodes["deploy"].node_id, "depends_on")
    return g, nodes


class TestWeakLinks(unittest.TestCase):
    def test_middle_of_chain_is_load_bearing(self):
        g, nodes = _chain_graph()
        ranked = g.weak_links()
        self.assertTrue(ranked)
        top2 = {r["label"] for r in ranked[:2]}
        self.assertEqual(top2, {"2pm meeting", "deploy"})
        for r in ranked:
            self.assertGreaterEqual(r["betweenness"], 0.0)
            self.assertLessEqual(r["betweenness"], 1.0)

    def test_empty_graph(self):
        from nomorals.planning.graph import WorldGraph
        g = WorldGraph(db_path=":memory:")
        self.assertEqual(g.weak_links(), [])


class TestCycles(unittest.TestCase):
    def test_acyclic(self):
        g, _ = _chain_graph()
        self.assertEqual(g.find_cycles(), [])

    def test_two_cycle_reported(self):
        from nomorals.planning.graph import WorldGraph
        g = WorldGraph(db_path=":memory:")
        a = g.add_node("schedule", "A")
        b = g.add_node("schedule", "B")
        g.add_edge(a.node_id, b.node_id, "depends_on")
        g.add_edge(b.node_id, a.node_id, "depends_on")
        cycles = g.find_cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), {a.node_id, b.node_id})

    def test_self_loop_rejected(self):
        from nomorals.planning.graph import WorldGraph
        g = WorldGraph(db_path=":memory:")
        a = g.add_node("schedule", "A")
        self.assertFalse(g.add_edge(a.node_id, a.node_id, "depends_on"))
        self.assertEqual(g.find_cycles(), [])


class TestPathsAndViews(unittest.TestCase):
    def test_path_between(self):
        g, nodes = _chain_graph()
        path = g.path_between(nodes["launch"].node_id, nodes["Lagos flight"].node_id)
        self.assertEqual([n.label for n in path],
                         ["launch", "deploy", "2pm meeting", "Lagos flight"])

    def test_path_between_unconnected(self):
        from nomorals.planning.graph import WorldGraph
        g = WorldGraph(db_path=":memory:")
        a = g.add_node("schedule", "A")
        b = g.add_node("schedule", "B")
        self.assertEqual(g.path_between(a.node_id, b.node_id), [])

    def test_neighborhood_depth(self):
        g, nodes = _chain_graph()
        hood = g.neighborhood(nodes["2pm meeting"].node_id, depth=1)
        labels = {n.label for n in hood}
        self.assertEqual(labels, {"2pm meeting", "Lagos flight", "deploy"})

    def test_schedule_waves_parallel(self):
        from nomorals.planning.graph import WorldGraph
        g = WorldGraph(db_path=":memory:")
        indep = [g.add_node("commitment", f"task{i}") for i in range(3)]
        later = g.add_node("deadline", "later")
        for n in indep:
            g.add_edge(later.node_id, n.node_id, "depends_on")
        waves = g.schedule_waves()
        self.assertEqual(len(waves), 2)
        self.assertEqual({n.label for n in waves[0]},
                         {f"task{i}" for i in range(3)})
        self.assertEqual([n.label for n in waves[1]], ["later"])

    def test_schedule_waves_cycle_leftover(self):
        from nomorals.planning.graph import WorldGraph
        g = WorldGraph(db_path=":memory:")
        a = g.add_node("schedule", "A")
        b = g.add_node("schedule", "B")
        g.add_edge(a.node_id, b.node_id, "depends_on")
        g.add_edge(b.node_id, a.node_id, "depends_on")
        waves = g.schedule_waves()
        self.assertEqual(len(waves), 1)  # the cyclic wave
        self.assertEqual({n.label for n in waves[0]}, {"A", "B"})

    def test_simulate_is_dry_run(self):
        g, nodes = _chain_graph()
        r = g.simulate_disruption(nodes["Lagos flight"].node_id)
        self.assertTrue(r["ok"])
        self.assertEqual(r["total_affected"], 3)
        self.assertEqual(r["max_depth"], 3)
        self.assertEqual(sorted(r["layers"]), ["1", "2", "3"])
        # nothing was marked disrupted — pure simulation
        self.assertEqual(g.disrupted(), [])

    def test_to_mermaid(self):
        g, _ = _chain_graph()
        out = g.to_mermaid()
        self.assertTrue(out.startswith("graph LR"))
        self.assertIn("depends_on", out)
        self.assertIn("Lagos flight", out)


class TestGraphChat(unittest.TestCase):
    def test_new_commands(self):
        from nomorals.planning.graph import control_graph
        g, nodes = _chain_graph()
        a = nodes["Lagos flight"].node_id
        d = nodes["launch"].node_id
        self.assertIn("load-bearing", control_graph("weak", graph=g))
        self.assertIn("no dependency cycles", control_graph("cycles", graph=g))
        self.assertIn("2pm meeting", control_graph(f"hood {a}", graph=g))
        self.assertIn("launch", control_graph(f"between {d} {a}", graph=g))
        self.assertIn("wave 1", control_graph("waves", graph=g))
        self.assertIn("dry run", control_graph(f"sim {a}", graph=g))
        self.assertIn("```mermaid", control_graph("map", graph=g))
        self.assertIn("usage", control_graph("between", graph=g))


# ── route: 2-opt, windows, roundtrip, freshness ────────────────────────

def _planner():
    from nomorals.planning.route import RoutePlanner
    return RoutePlanner(db_path=":memory:", backend="greedy")


class TestTwoOpt(unittest.TestCase):
    def test_two_opt_never_worsens(self):
        from nomorals.planning.route import RouteSolver, CostModel, Stop
        rng = random.Random(11)
        cm = CostModel(db_path=":memory:")
        sv = RouteSolver("greedy")
        pts = [Stop(f"s{i}", f"p{i}", 6.3 + rng.random() * 0.3, 3.3 + rng.random() * 0.3)
               for i in range(14)]
        origin = Stop("o", "o", 6.45, 3.39)
        t0 = time.time()
        raw = sv._solve_greedy([origin] + pts, cm, t0)

        def tour_len(order):
            return sum(cm.estimate(a, b)[0] for a, b in zip(order, order[1:]))

        improved = sv._two_opt(raw, cm, t0=t0)
        self.assertLessEqual(tour_len(improved), tour_len(raw) + 1e-6)
        # origin stays fixed, all stops still present exactly once
        self.assertEqual(improved[0].stop_id, "o")
        self.assertEqual({s.stop_id for s in improved},
                         {s.stop_id for s in raw})

    def test_two_opt_improves_crossed_tour(self):
        # Deliberately crossed order on a line: 2-opt must uncross it.
        from nomorals.planning.route import RouteSolver, CostModel, Stop
        cm = CostModel(db_path=":memory:")
        sv = RouteSolver("greedy")
        order = [Stop(f"s{i}", f"p{i}", 6.40 + i * 0.01, 3.39)
                 for i in (0, 2, 1, 3)]
        t0 = time.time()

        def tour_len(o):
            return sum(cm.estimate(a, b)[0] for a, b in zip(o, o[1:]))

        before = tour_len(order)
        after = tour_len(sv._two_opt(order, cm, t0=t0))
        self.assertLess(after, before)


class TestWindowsAndRoundtrip(unittest.TestCase):
    def test_window_violation_reported(self):
        p = _planner()
        a = p.add_stop("home", lat=6.45, lng=3.39)
        b = p.add_stop("bank", lat=6.46, lng=3.40)
        p.set_window("bank", "00:00", "00:01")  # long past
        r = p.plan([a, b], start_time=time.time())
        self.assertTrue(r.window_violations)
        self.assertIn("bank", r.window_violations[0])

    def test_window_respected_when_feasible(self):
        from nomorals.planning.route import Stop
        import datetime as dt
        p = _planner()
        now = time.time()
        # bank open for the next 6 hours — easily reachable
        t0 = now
        a = Stop("a", "home", 6.45, 3.39)
        w0 = t0 - 3600
        w1 = t0 + 6 * 3600
        b = Stop("b", "bank", 6.451, 3.391, window=(w0, w1))
        r = p.plan([a, b], start_time=t0)
        self.assertEqual(r.window_violations, [])

    def test_return_to_origin(self):
        p = _planner()
        stops = [p.add_stop(n, lat=6.45 + i * 0.01, lng=3.39)
                 for i, n in enumerate(["home", "a", "b"])]
        r = p.plan(stops, return_to_origin=True)
        self.assertIs(r.order[-1], r.order[0])
        self.assertEqual(len(r.legs), 3)
        r2 = p.plan(stops)
        self.assertEqual(len(r2.legs), 2)

    def test_window_chat_command(self):
        from nomorals.planning.route import control_route, RoutePlanner
        p = RoutePlanner(db_path=":memory:", backend="greedy")
        p.add_stop("home", lat=6.45, lng=3.39)
        p.add_stop("bank", lat=6.46, lng=3.40)
        out = control_route("window bank 09:00-17:00", planner=p)
        self.assertIn("window set", out)
        stop = p.find_stop("bank")
        self.assertIsNotNone(stop.window)
        out = control_route("plan home; bank roundtrip", planner=p)
        self.assertIn("back home", out)


class TestCostModelFreshness(unittest.TestCase):
    def test_learned_estimate(self):
        from nomorals.planning.route import CostModel, Stop
        cm = CostModel(db_path=":memory:")
        a, b = Stop("a", "A", 6.45, 3.39), Stop("b", "B", 6.46, 3.40)
        self.assertTrue(cm.record_actual(a, b, 12.0, 5000))
        minutes, cost, source = cm.estimate_detail(a, b)
        self.assertEqual(source, "learned")
        self.assertAlmostEqual(minutes, 12.0, places=1)

    def test_stale_pair_decays(self):
        from nomorals.planning.route import CostModel, Stop
        cm = CostModel(db_path=":memory:")
        a, b = Stop("a", "A", 6.45, 3.39), Stop("b", "B", 6.46, 3.40)
        cm.record_actual(a, b, 60.0, 0)
        # age the observation 60 days, then record a fresh one
        cm._db.execute("UPDATE route_actuals SET recorded_at = ?",
                       (time.time() - 60 * 86400,))
        cm._db.commit()
        cm.record_actual(a, b, 10.0, 0)
        minutes, _, _ = cm.estimate_detail(a, b)
        # stale 60m observation should be mostly forgotten → near 10m
        self.assertLess(minutes, 20.0)

    def test_stats_reports_recent(self):
        from nomorals.planning.route import CostModel, Stop
        cm = CostModel(db_path=":memory:")
        a, b = Stop("a", "A", 6.45, 3.39), Stop("b", "B", 6.46, 3.40)
        cm.record_actual(a, b, 12.0, 0)
        s = cm.stats()
        self.assertEqual(s["recorded_trips"], 1)
        self.assertEqual(s["recent_trips_30d"], 1)


# ── estimates: Monte Carlo, reference class, calibration ───────────────

class TestPertSample(unittest.TestCase):
    def test_bounds(self):
        from nomorals.planning.estimates import pert_sample
        rng = random.Random(3)
        for _ in range(200):
            x = pert_sample(10, 20, 45, rng)
            self.assertGreaterEqual(x, 10)
            self.assertLessEqual(x, 45)

    def test_mean_converges_to_te(self):
        from nomorals.planning.estimates import pert_sample
        rng = random.Random(3)
        xs = [pert_sample(10, 20, 45, rng) for _ in range(20000)]
        te = (10 + 4 * 20 + 45) / 6
        self.assertAlmostEqual(statistics.mean(xs), te, delta=0.5)

    def test_degenerate(self):
        from nomorals.planning.estimates import pert_sample
        self.assertEqual(pert_sample(5, 5, 5), 5)
        self.assertEqual(pert_sample("x", "y", "z"), 0.0)


def _mc_steps():
    return [
        {"id": "api", "label": "API build",
         "optimistic": 20, "most_likely": 40, "pessimistic": 90},
        {"id": "ui", "label": "UI build",
         "optimistic": 10, "most_likely": 20, "pessimistic": 45},
        {"id": "integ", "label": "Integration",
         "optimistic": 15, "most_likely": 30, "pessimistic": 60,
         "depends_on": ["api", "ui"]},
    ]


class TestMonteCarlo(unittest.TestCase):
    def test_percentiles_ordered(self):
        from nomorals.planning.estimates import monte_carlo
        r = monte_carlo("launch", _mc_steps(), iterations=3000, seed=42)
        self.assertTrue(r.ok)
        self.assertLessEqual(r.p10, r.p50)
        self.assertLessEqual(r.p50, r.p80)
        self.assertLessEqual(r.p80, r.p90)
        self.assertGreater(r.p90, 0)

    def test_seed_deterministic(self):
        from nomorals.planning.estimates import monte_carlo
        a = monte_carlo("launch", _mc_steps(), iterations=2000, seed=9)
        b = monte_carlo("launch", _mc_steps(), iterations=2000, seed=9)
        self.assertEqual(a.p50, b.p50)
        self.assertEqual(a.p90, b.p90)
        self.assertEqual(a.seed, b.seed)

    def test_criticality_sensible(self):
        from nomorals.planning.estimates import monte_carlo
        r = monte_carlo("launch", _mc_steps(), iterations=3000, seed=42)
        # api dominates ui (bigger, wider); integ is the sink — always critical
        self.assertGreater(r.criticality["api"], r.criticality["ui"])
        self.assertAlmostEqual(r.criticality["integ"], 1.0, places=1)
        # sensitivity top driver should be api
        self.assertEqual(r.sensitivity[0][0], "api")

    def test_empty_steps(self):
        from nomorals.planning.estimates import monte_carlo
        r = monte_carlo("x", [], iterations=100)
        self.assertFalse(r.ok)

    def test_format(self):
        from nomorals.planning.estimates import monte_carlo
        r = monte_carlo("launch", _mc_steps(), iterations=1000, seed=1)
        text = r.format()
        self.assertIn("P50", text)
        self.assertIn("P90", text)
        self.assertIn("contingency", text)


class TestReferenceClass(unittest.TestCase):
    def _store(self):
        from nomorals.planning.estimates import EstimateStore
        path = os.path.join(tempfile.mkdtemp(), "e.db")
        return EstimateStore(db_path=path), path

    def test_no_history_honest(self):
        from nomorals.planning.estimates import reference_class_check
        r = reference_class_check("novel-task", 30.0,
                                  db_path=os.path.join(tempfile.mkdtemp(), "e.db"))
        self.assertTrue(r["ok"])
        self.assertIn("no reference class", r["verdict"])

    def test_optimism_uplift(self):
        from nomorals.planning.estimates import reference_class_check
        st, path = self._store()
        for actual in (45, 50, 55, 48, 52, 60):
            st.record_actual("research", 30.0, float(actual))
        r = reference_class_check("research", 30.0, db_path=path)
        self.assertTrue(r["ok"])
        self.assertGreater(r["uplift_ratio"], 1.25)
        self.assertGreater(r["outside_anchor"], r["inside"])
        self.assertIn("optimism", r["verdict"])

    def test_calibration(self):
        from nomorals.planning.estimates import calibration
        st, path = self._store()
        for actual in (45, 50, 55, 48, 52, 60):  # all outside the 30±25% band
            st.record_actual("research", 30.0, float(actual))
        rows = calibration(db_path=path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_type"], "research")
        self.assertLess(rows[0]["hit_rate"], 0.6)
        self.assertIn("overconfident", rows[0]["verdict"])


class TestEtaChat(unittest.TestCase):
    def test_sim_command(self):
        from nomorals.planning.estimates import control_eta
        out = control_eta("sim launch api:20/40/90 ui:10/20/45", db_path=":memory:")
        self.assertIn("Monte Carlo", out)
        self.assertIn("P50", out)

    def test_sim_with_deps(self):
        from nomorals.planning.estimates import control_eta
        out = control_eta("sim launch api:20/40/90 integ:15/30/60>api",
                          db_path=":memory:")
        self.assertIn("Monte Carlo", out)

    def test_outside_no_history(self):
        from nomorals.planning.estimates import control_eta
        out = control_eta("outside research 30", db_path=":memory:")
        self.assertIn("no reference class", out)

    def test_calibrate_no_history(self):
        from nomorals.planning.estimates import control_eta
        out = control_eta("calibrate", db_path=":memory:")
        self.assertIn("no task types", out)


# ── congestion: leases, breaker, wait ──────────────────────────────────

def _monitor():
    from nomorals.planning.congestion import ContentionMonitor
    return ContentionMonitor(in_memory=True)


class TestHoldLeases(unittest.TestCase):
    def test_expired_hold_ignored(self):
        mon = _monitor()
        mon.register("gpu", capacity=2)
        mon.acquire("gpu", "a1", ttl_s=0.15)
        mon.acquire("gpu", "a2", ttl_s=300)
        self.assertEqual(mon.queue_depth("gpu"), 2)
        time.sleep(0.2)
        self.assertEqual(mon.queue_depth("gpu"), 1)

    def test_reap_stale(self):
        mon = _monitor()
        mon.register("gpu", capacity=2)
        mon.acquire("gpu", "a1", ttl_s=0.1)
        time.sleep(0.15)
        self.assertEqual(mon.reap_stale(), 1)
        self.assertEqual(mon.queue_depth("gpu"), 0)

    def test_no_expiry_hold(self):
        mon = _monitor()
        mon.register("gpu", capacity=2)
        mon.acquire("gpu", "a1", ttl_s=0)  # managed manually
        time.sleep(0.05)
        self.assertEqual(mon.queue_depth("gpu"), 1)
        self.assertTrue(mon.release("gpu", "a1"))

    def test_hold_context_manager(self):
        from nomorals.planning.congestion import hold
        mon = _monitor()
        mon.register("gpu", capacity=2)
        with hold(mon, "gpu", "a1"):
            self.assertEqual(mon.queue_depth("gpu"), 1)
        self.assertEqual(mon.queue_depth("gpu"), 0)


class TestCircuitBreaker(unittest.TestCase):
    def test_closed_to_open(self):
        mon = _monitor()
        mon.register("hot", capacity=1)
        rid = mon._resolve("hot").resource_id
        self.assertEqual(mon.breaker_state("hot"), "closed")
        for _ in range(10):
            mon._breaker_observe(rid, True)
        self.assertEqual(mon.breaker_state("hot"), "open")

    def test_open_fails_fast_in_advise(self):
        mon = _monitor()
        mon.register("hot", capacity=1)
        rid = mon._resolve("hot").resource_id
        for _ in range(10):
            mon._breaker_observe(rid, True)
        a = mon.advise("hot")
        self.assertEqual(a["action"], "backoff")
        self.assertIn("circuit is OPEN", a["detail"])

    def test_half_open_probe_then_close(self):
        mon = _monitor()
        mon.register("hot", capacity=1)
        rid = mon._resolve("hot").resource_id
        for _ in range(10):
            mon._breaker_observe(rid, True)
        # force cooldown expiry
        row = mon._breaker_row(rid)
        row["opened_at"] = time.time() - 61
        mon._breaker_save(rid, row)
        self.assertEqual(mon.breaker_state("hot"), "half_open")
        a = mon.advise("hot")
        self.assertEqual(a["action"], "stagger")
        self.assertEqual(a.get("wave_size"), 1)
        # two healthy probes close it
        mon._breaker_observe(rid, False)
        mon._breaker_observe(rid, False)
        self.assertEqual(mon.breaker_state("hot"), "closed")

    def test_half_open_failure_reopens(self):
        mon = _monitor()
        mon.register("hot", capacity=1)
        rid = mon._resolve("hot").resource_id
        for _ in range(10):
            mon._breaker_observe(rid, True)
        row = mon._breaker_row(rid)
        row["opened_at"] = time.time() - 61
        mon._breaker_save(rid, row)
        self.assertEqual(mon.breaker_state("hot"), "half_open")
        mon._breaker_observe(rid, True)
        self.assertEqual(mon.breaker_state("hot"), "open")

    def test_healthy_observations_keep_closed(self):
        mon = _monitor()
        mon.register("cool", capacity=4)
        rid = mon._resolve("cool").resource_id
        for _ in range(10):
            mon._breaker_observe(rid, False)
        self.assertEqual(mon.breaker_state("cool"), "closed")
        a = mon.advise("cool")
        self.assertEqual(a["action"], "proceed")


class TestWaitEstimate(unittest.TestCase):
    def test_wait_none_when_clear(self):
        mon = _monitor()
        mon.register("gpu", capacity=4)
        self.assertEqual(mon.wait_estimate("gpu"), 0.0)

    def test_wait_none_without_rate(self):
        mon = _monitor()
        mon.register("gpu", capacity=1)
        mon.acquire("gpu", "a1", ttl_s=300)
        self.assertIsNone(mon.wait_estimate("gpu"))

    def test_wait_with_service_rate(self):
        mon = _monitor()
        mon.register("gpu", capacity=1)
        now = time.time()
        rid = mon._resolve("gpu").resource_id
        # 6 releases in the last 5 minutes → ~1.2/min service rate
        for i in range(6):
            mon._db.execute(
                "INSERT INTO congestion_events (resource_id, kind, ts) "
                "VALUES (?, 'release', ?)", (rid, now - i * 40))
        mon._db.commit()
        mon.acquire("gpu", "a1", ttl_s=300)
        mon.acquire("gpu", "a2", ttl_s=300)
        w = mon.wait_estimate("gpu")
        self.assertIsNotNone(w)
        self.assertGreater(w, 0)


class TestCongestionChat(unittest.TestCase):
    def test_breaker_and_reap_commands(self):
        from nomorals.planning.congestion import control_congestion
        mon = _monitor()
        mon.register("hot", capacity=1)
        out = control_congestion("breaker hot", monitor=mon)
        self.assertIn("CLOSED", out)
        mon.acquire("hot", "a1", ttl_s=0.05)
        time.sleep(0.08)
        out = control_congestion("reap", monitor=mon)
        self.assertIn("1 expired", out)
        out = control_congestion("status", monitor=mon)
        self.assertIn("circuit closed", out)


if __name__ == "__main__":
    unittest.main()
