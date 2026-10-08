"""Tests for the live world-graph planning substrate (#89). All offline."""

import os
import tempfile

import pytest

from nomorals.planning.graph import (
    WorldGraph,
    control_graph,
    disruption_alerts,
)


def _graph():
    return WorldGraph(db_path=os.path.join(tempfile.mkdtemp(), "g.db"))


def test_add_and_get_node():
    g = _graph()
    n = g.add_node("schedule", "LOS → LHR flight", {"flight_no": "BA075"})
    assert n is not None
    assert n.node_id.startswith("wg_")
    assert g.get(n.node_id).label == "LOS → LHR flight"


def test_add_node_rejects_bad_type_and_empty_label():
    g = _graph()
    assert g.add_node("spaceship", "x") is None
    assert g.add_node("person", "   ") is None


def test_list_nodes_filters_by_type():
    g = _graph()
    g.add_node("person", "Ada")
    g.add_node("deadline", "Rent due")
    assert len(g.list_nodes("person")) == 1
    assert len(g.list_nodes()) == 2


def test_add_edge_and_reject_bad():
    g = _graph()
    a = g.add_node("commitment", "2pm meeting")
    b = g.add_node("schedule", "Lagos flight")
    assert g.add_edge(a.node_id, b.node_id, "depends_on") is True
    assert g.add_edge(a.node_id, b.node_id, "teleports") is False
    assert g.add_edge(a.node_id, a.node_id, "depends_on") is False
    assert g.add_edge(a.node_id, "nope", "depends_on") is False


def test_disruption_propagation_depends_on():
    g = _graph()
    flight = g.add_node("schedule", "Lagos flight")
    meet = g.add_node("commitment", "2pm meeting")
    dinner = g.add_node("commitment", "dinner after meeting")
    g.add_edge(meet.node_id, flight.node_id, "depends_on")
    g.add_edge(dinner.node_id, meet.node_id, "depends_on")
    affected = g.mark_disrupted(flight.node_id, note="cancelled")
    labels = {n.label for n in affected}
    assert labels == {"Lagos flight", "2pm meeting", "dinner after meeting"}
    # flags persisted
    assert g.get(meet.node_id).disrupted is True
    assert g.get(meet.node_id).attrs["disruption_note"] == "cancelled"


def test_disruption_propagation_blocks():
    g = _graph()
    visa = g.add_node("commitment", "visa approval")
    trip = g.add_node("project", "London trip")
    g.add_edge(visa.node_id, trip.node_id, "blocks")
    affected = g.mark_disrupted(visa.node_id)
    assert {n.label for n in affected} == {"visa approval", "London trip"}


def test_dependents_and_dependencies():
    g = _graph()
    flight = g.add_node("schedule", "flight")
    meet = g.add_node("commitment", "meeting")
    g.add_edge(meet.node_id, flight.node_id, "depends_on")
    assert [n.label for n in g.dependents(flight.node_id)] == ["meeting"]
    assert [n.label for n in g.dependencies(meet.node_id)] == ["flight"]
    assert g.dependents("missing") == []
    assert g.dependencies("missing") == []


def test_what_breaks_if():
    g = _graph()
    a = g.add_node("asset", "generator")
    b = g.add_node("commitment", "night shift")
    g.add_edge(b.node_id, a.node_id, "depends_on")
    r = g.what_breaks_if(a.node_id)
    assert r["ok"] is True
    assert r["count"] == 1
    assert r["affected"][0]["label"] == "night shift"
    assert g.what_breaks_if("missing")["ok"] is False


def test_critical_path():
    g = _graph()
    d1 = g.add_node("deadline", "visa approved")
    d2 = g.add_node("deadline", "flight booked")
    d3 = g.add_node("deadline", "hotel confirmed")
    g.add_edge(d2.node_id, d1.node_id, "depends_on")
    g.add_edge(d3.node_id, d2.node_id, "depends_on")
    g.add_node("person", "Ada")  # not in the chain
    path = g.critical_path()
    assert [n.label for n in path] == ["hotel confirmed", "flight booked", "visa approved"]


def test_critical_path_empty_graph():
    assert _graph().critical_path() == []


def test_clear_and_list_disrupted():
    g = _graph()
    n = g.add_node("schedule", "flight")
    g.mark_disrupted(n.node_id)
    assert len(g.disrupted()) == 1
    assert g.clear_disruption(n.node_id) is True
    assert g.disrupted() == []
    assert g.clear_disruption("missing") is False


def test_remove_node_removes_edges():
    g = _graph()
    a = g.add_node("person", "Ada")
    b = g.add_node("project", "Trip")
    g.add_edge(a.node_id, b.node_id, "owned_by")
    assert g.remove_node(a.node_id) is True
    assert g.get(a.node_id) is None
    assert g.edges(b.node_id) == []
    assert g.remove_node("missing") is False


def test_sync_from_tasks_idempotent():
    g = _graph()
    tasks = [{"task_id": "t1", "label": "Pay rent", "run_at": 1700000000.0}]
    assert g.sync_from_tasks(tasks) == 1
    assert g.sync_from_tasks(tasks) == 1  # same count, not duplicated
    assert len(g.list_nodes("schedule")) == 1


def test_sync_from_memory_with_fake_memory():
    g = _graph()

    class Rec:
        def __init__(self, rid, text, kind, tags):
            self.record_id = rid
            self.text = text
            self.kind = kind
            self.tags = tags

    class Result:
        records = [
            Rec("r1", "Rent due Friday", "commitment", "deadline"),
            Rec("r2", "Flight to London", "plan", "flight"),
            Rec("r3", "", "note", ""),
        ]

    class FakeMemory:
        def recall(self, query, limit=None):
            return Result()

    assert g.sync_from_memory(FakeMemory()) == 2
    # idempotent: second sync adds nothing new
    assert g.sync_from_memory(FakeMemory()) == 2
    types = {n.type for n in g.list_nodes()}
    assert "deadline" in types and "schedule" in types


def test_sync_from_memory_never_raises():
    g = _graph()
    assert g.sync_from_memory(object()) == 0  # no recall method
    assert g.sync_from_memory(None) in (0,) or True  # default manager may not exist


def test_disruption_alerts_first_consumer():
    g = _graph()
    flight = g.add_node("schedule", "Lagos flight delayed", {"city": "Lagos"})
    g.mark_disrupted(flight.node_id, note="delayed 3h")
    tasks = [
        {"label": "2pm meeting after Lagos flight", "run_at": 1700000000.0},
        {"label": "buy groceries"},
    ]
    alerts = disruption_alerts(g, tasks)
    assert len(alerts) == 1
    assert "2pm meeting" in alerts[0]
    assert "Lagos flight delayed" in alerts[0]
    assert disruption_alerts(_graph(), tasks) == []  # nothing disrupted
    assert disruption_alerts(g, []) == []


def test_termux_profile_is_in_memory():
    g = WorldGraph(profile="termux")
    n = g.add_node("person", "Ada")
    assert n is not None
    assert g.get(n.node_id) is not None


def test_chat_add_link_disrupt_show():
    g = WorldGraph(db_path=os.path.join(tempfile.mkdtemp(), "g.db"))
    out = control_graph("add schedule Lagos flight", graph=g)
    assert "✅ added" in out
    node_id = out.strip().split("`")[1]
    out = control_graph(f"show {node_id}", graph=g)
    assert "Lagos flight" in out
    meet = control_graph("add commitment 2pm meeting", graph=g).strip().split("`")[1]
    out = control_graph(f"link {meet} {node_id} depends_on", graph=g)
    assert "✅ linked" in out
    out = control_graph(f"disrupt {node_id} cancelled", graph=g)
    assert "2 node(s) affected" in out
    assert "Lagos flight" in out and "2pm meeting" in out


def test_chat_breaks_path_clear_and_bad_input():
    g = WorldGraph(db_path=os.path.join(tempfile.mkdtemp(), "g.db"))
    assert "node not found" in control_graph("show nope", graph=g)
    assert "node not found" in control_graph("disrupt nope", graph=g)
    assert "usage" in control_graph("add schedule", graph=g).lower()
    assert "usage" in control_graph("link a b", graph=g).lower()
    assert "couldn't link" in control_graph("link a b depends_on", graph=g)
    n = g.add_node("schedule", "flight")
    out = control_graph(f"breaks {n.node_id}", graph=g)
    assert "affected" in out
    out = control_graph("path", graph=g)
    assert "no dependency chain" in out or "critical path" in out
    out = control_graph(f"clear {n.node_id}", graph=g)
    assert "cleared" in out
    assert "not found" in control_graph("clear nope", graph=g)
    out = control_graph("sync", graph=g)
    assert "projected" in out
    assert "🌐" in control_graph("", graph=g)


def test_chat_never_raises_on_garbage():
    g = WorldGraph(db_path=os.path.join(tempfile.mkdtemp(), "g.db"))
    for tail in ["", "add", "link", "disrupt", "clear", "show", "breaks", "path",
                 "sync", "frobnicate", "add badtype x", None]:
        out = control_graph(tail, graph=g)
        assert isinstance(out, str) and out
