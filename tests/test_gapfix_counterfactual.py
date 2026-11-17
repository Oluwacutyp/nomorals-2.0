"""Gapfix R2-35 — counterfactual "what if" wired into the decisions flow.

Failing-first: on the pristine tree the `whatif` verb does not exist, the
`add` verb ignores a 5th `|` field, and `whatif_for_options` is not
importable — every new-behavior test below fails before the wiring lands.

Every test drives the PRODUCTION entry points (`_control_decisions` on a
bare RuntimeReachMixin host, the shared `_world_graph()` the /graph
surface uses) — never the dormant organ in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _reach_host(tmp_path):
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _graph_with_launch(host):
    """Launch <- Website <- Ads (depends_on): Launch blast 2, Ads blast 0."""
    graph = host._world_graph()
    a = graph.add_node("project", "Launch")
    b = graph.add_node("project", "Website")
    c = graph.add_node("project", "Ads")
    graph.add_edge(b.node_id, a.node_id, "depends_on")
    graph.add_edge(c.node_id, b.node_id, "depends_on")
    return graph, a, b, c


# ── the pure bridge ────────────────────────────────────────────────────


def test_whatif_for_options_matches_by_id_and_label():
    from nomorals.planning.counterfactual import whatif_for_options
    from nomorals.planning.graph import WorldGraph

    graph = WorldGraph(":memory:")
    a = graph.add_node("project", "Launch")
    b = graph.add_node("project", "Website")
    graph.add_edge(b.node_id, a.node_id, "depends_on")
    out = whatif_for_options(graph, ["launch", a.node_id])
    assert out["unmatched"] == []
    # The label is the owner's phrasing ("launch"); the node id is the match.
    assert out["recommendation"]["label"] == "launch"
    assert out["recommendation"]["node_id"] == a.node_id
    labels = [o["option"]["label"] for o in out["options"]]
    assert labels == ["launch", a.node_id]


def test_whatif_for_options_reports_unmatched_never_drops():
    from nomorals.planning.counterfactual import whatif_for_options
    from nomorals.planning.graph import WorldGraph

    graph = WorldGraph(":memory:")
    _a = graph.add_node("project", "Launch")
    out = whatif_for_options(graph, ["Launch", "Ghost"])
    assert [o["option"]["label"] for o in out["options"]] == ["Launch"]
    assert len(out["unmatched"]) == 1
    assert out["unmatched"][0]["option"] == "Ghost"
    assert "no world-graph node" in out["unmatched"][0]["reason"]
    assert out["recommendation"]["label"] == "Launch"


def test_whatif_for_options_ambiguous_label_is_not_guessed():
    from nomorals.planning.counterfactual import whatif_for_options
    from nomorals.planning.graph import WorldGraph

    graph = WorldGraph(":memory:")
    graph.add_node("project", "Launch")
    graph.add_node("project", "Launch")
    out = whatif_for_options(graph, ["Launch"])
    assert out["options"] == []
    assert out["recommendation"] is None
    assert len(out["unmatched"]) == 1
    assert "2 nodes" in out["unmatched"][0]["reason"]


def test_whatif_for_options_empty_options_is_honest():
    from nomorals.planning.counterfactual import whatif_for_options
    from nomorals.planning.graph import WorldGraph

    out = whatif_for_options(WorldGraph(":memory:"), [])
    assert out["options"] == []
    assert out["recommendation"] is None
    assert out["unmatched"] == []


# ── the /decisions verbs ───────────────────────────────────────────────


def test_add_with_options_runs_comparison_before_decided(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    _graph_with_launch(host)
    out = host._control_decisions(
        "add Launch the site | delay | churn stays low | 70% | Launch; Ads")
    assert out.startswith("recorded:")
    # Each option simulated, comparison shown, recommendation named + reasoned.
    assert "what-if" in out.lower()
    assert "Launch" in out and "Ads" in out
    assert "blast 2" in out  # Launch: Website + Ads depend on it
    assert "blast 0" in out  # Ads: nothing depends on it
    assert "least disruptive" in out.lower()
    assert "Ads" in out.split("least disruptive")[-1]
    # The owner's pick stands: they chose "delay" (the high-blast option),
    # the recommendation did not overwrite it.
    d = host._decisions_journal().open_decisions()[0]
    assert d.chosen == "delay"
    assert d.options == ["Launch", "Ads"]


def test_add_without_options_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    out = host._control_decisions(
        "add Raise prices | raise | churn stays low | 70%")
    assert out.startswith("recorded:")
    assert "what-if" not in out.lower()
    d = host._decisions_journal().open_decisions()[0]
    assert d.options == []


def test_add_with_unmatched_options_degrades_honestly(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    out = host._control_decisions(
        "add Do the thing | go | it works | 60% | Ghost; Phantom")
    assert out.startswith("recorded:")
    assert "no world-graph node matches" in out
    assert "blast" not in out
    d = host._decisions_journal().open_decisions()[0]
    assert d.options == ["Ghost", "Phantom"]
    assert d.chosen == "go"


def test_whatif_verb_reruns_comparison(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    _graph_with_launch(host)
    host._control_decisions(
        "add Launch the site | delay | churn stays low | 70% | Launch; Ads")
    did = host._decisions_journal().open_decisions()[0].id
    out = host._control_decisions(f"whatif {did[:11]}")
    assert "what-if" in out.lower()
    assert "blast 2" in out and "blast 0" in out
    assert "least disruptive" in out.lower()


def test_whatif_unknown_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    out = host._control_decisions("whatif dc-deadbeef")
    assert "no decision matching" in out


def test_whatif_decision_without_options(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    host._control_decisions("add Raise prices | raise | churn stays low | 70%")
    did = host._decisions_journal().open_decisions()[0].id
    out = host._control_decisions(f"whatif {did[:11]}")
    assert "no options recorded" in out
    assert "blast" not in out


def test_recommendation_is_advisory_owner_can_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    _graph_with_launch(host)
    # Owner picks Launch (blast 2) over the recommended Ads (blast 0).
    out = host._control_decisions(
        "add Launch the site | Launch | it lands | 70% | Launch; Ads")
    assert "Ads" in out.split("least disruptive")[-1]
    did = host._decisions_journal().open_decisions()[0].id
    resolved = host._control_decisions(f"resolve {did[:11]} good it landed")
    assert resolved.startswith("resolved:")
    d = host._decisions_journal().decisions[did]
    assert d.chosen == "Launch"
    assert d.status == "resolved"
    assert d.outcome == "it landed"


def test_simulation_is_readonly(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    graph, _a, _b, _c = _graph_with_launch(host)
    host._control_decisions(
        "add Launch the site | delay | churn stays low | 70% | Launch; Ads")
    journal_path = tmp_path / "data" / "decisions.json"
    before_journal = journal_path.read_bytes()
    before_nodes = [(n.node_id, n.label, n.disrupted)
                    for n in graph.list_nodes()]
    before_edges = [(e.from_id, e.to_id, e.type) for e in graph.edges()]
    did = host._decisions_journal().open_decisions()[0].id
    host._control_decisions(f"whatif {did[:11]}")
    assert journal_path.read_bytes() == before_journal
    assert [(n.node_id, n.label, n.disrupted)
            for n in graph.list_nodes()] == before_nodes
    assert [(e.from_id, e.to_id, e.type) for e in graph.edges()] == before_edges
    assert all(not n.disrupted for n in graph.list_nodes())
    # The journal on disk still holds exactly the one recorded decision.
    rows = json.loads(journal_path.read_text())["decisions"]
    assert len(rows) == 1 and rows[0]["chosen"] == "delay"


def test_ambiguous_label_reported_in_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    host = _reach_host(tmp_path)
    graph = host._world_graph()
    graph.add_node("project", "Launch")
    graph.add_node("project", "Launch")
    out = host._control_decisions(
        "add Launch the site | delay | it lands | 70% | Launch")
    assert out.startswith("recorded:")
    assert "matches 2 nodes" in out
    assert "blast" not in out
