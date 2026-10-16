"""Commitment ledger -> world-graph projection.

Covers nomorals/planning/commitment_sync.py against the real
WorldGraph + CommitmentLedger: extraction -> node, idempotency,
person + deadline edges, kept-status refresh, overdue detection,
and the empty-ledger case.
"""

import time

import pytest

from nomorals.commitments.ledger import Commitment, CommitmentLedger
from nomorals.planning.commitment_sync import (
    overdue_commitments_in_graph,
    sync_commitment_to_graph,
    sync_extracted_to_graph,
    sync_ledger_to_graph,
)
from nomorals.planning.graph import WorldGraph

NOW = 1792000000.0  # fixed clock so deadline maths is deterministic


def _graph():
    return WorldGraph(":memory:")


def _node_by_ref(graph, ref):
    for node in graph.list_nodes():
        if node.external_ref == ref:
            return node
    return None


def _edges_from(graph, node_id, etype):
    return [e for e in graph.edges(node_id)
            if e.from_id == node_id and e.type == etype]


# ── extract -> node ──────────────────────────────────────────────────

def test_extract_to_node_appears_in_graph():
    ledger = CommitmentLedger()
    graph = _graph()
    result = sync_extracted_to_graph(
        {"text": "I'll send you the deck by Friday",
         "sender": "owner", "recipient": "Alex",
         "timestamp": NOW, "id": "m1"},
        ledger, graph)
    assert result["ok"] is True
    assert result["added"] == 1
    assert len(result["commitment_nodes"]) == 1
    node = graph.get(result["commitment_nodes"][0])
    assert node is not None
    assert node.type == "commitment"
    assert node.external_ref.startswith("commitment:cm-")
    assert "send" in node.label
    assert node.attrs["status"] == "open"


def test_sync_extracted_accepts_message_list():
    ledger = CommitmentLedger()
    graph = _graph()
    result = sync_extracted_to_graph(
        [{"text": "I will call you tomorrow",
          "sender": "owner", "recipient": "Sam",
          "timestamp": NOW, "id": "m2"}],
        ledger, graph)
    assert result["ok"] is True
    assert result["added"] == 1


def test_sync_extracted_no_promise_syncs_nothing():
    ledger = CommitmentLedger()
    graph = _graph()
    result = sync_extracted_to_graph(
        {"text": "Nice weather today", "sender": "owner",
         "timestamp": NOW, "id": "m3"},
        ledger, graph)
    assert result["ok"] is True
    assert result["added"] == 0
    assert graph.list_nodes() == []


# ── idempotency ───────────────────────────────────────────────────────

def test_idempotent_sync_twice_same_node_count():
    graph = _graph()
    commitment = Commitment(
        id="cm-test1", promisor="owner", promisee="Alex",
        what="send you the deck", deadline=NOW + 86400,
        source_text="I'll send you the deck by Friday",
        created_at=NOW)
    first = sync_commitment_to_graph(graph, commitment)
    count_after_first = len(graph.list_nodes())
    assert first["ok"] is True
    assert count_after_first == 3  # commitment + person + deadline
    second = sync_commitment_to_graph(graph, commitment)
    assert second["ok"] is True
    assert second["commitment_node"] == first["commitment_node"]
    assert second["person_node"] == first["person_node"]
    assert second["deadline_node"] == first["deadline_node"]
    assert len(graph.list_nodes()) == count_after_first
    assert len(graph.edges()) == 2


def test_idempotent_ledger_sync_twice():
    ledger = CommitmentLedger()
    ledger.add(promisor="owner", promisee="Alex", what="send the deck",
               deadline=NOW + 86400, source_text="I'll send the deck")
    graph = _graph()
    assert sync_ledger_to_graph(ledger, graph) == 1
    count = len(graph.list_nodes())
    assert sync_ledger_to_graph(ledger, graph) == 1
    assert len(graph.list_nodes()) == count


# ── person + deadline edges ───────────────────────────────────────────

def test_person_and_deadline_edges_exist():
    graph = _graph()
    commitment = Commitment(
        id="cm-edges", promisor="owner", promisee="Alex",
        what="send you the deck", deadline=NOW + 86400,
        source_text="I'll send you the deck by Friday",
        created_at=NOW)
    result = sync_commitment_to_graph(graph, commitment)
    assert result["ok"] is True

    person = _node_by_ref(graph, "person:Alex")
    assert person is not None
    assert person.type == "person"
    assert person.label == "Alex"
    assert result["person_node"] == person.node_id
    owned = _edges_from(graph, result["commitment_node"], "owned_by")
    assert len(owned) == 1
    assert owned[0].to_id == person.node_id

    deadline = _node_by_ref(graph, "deadline:cm-edges")
    assert deadline is not None
    assert deadline.type == "deadline"
    assert deadline.attrs["due_ts"] == pytest.approx(NOW + 86400)
    assert result["deadline_node"] == deadline.node_id
    due = _edges_from(graph, result["commitment_node"], "due")
    assert len(due) == 1
    assert due[0].to_id == deadline.node_id


def test_no_person_or_deadline_when_absent():
    graph = _graph()
    commitment = Commitment(
        id="cm-bare", promisor="owner", promisee=None,
        what="review the notes", deadline=None, created_at=NOW)
    result = sync_commitment_to_graph(graph, commitment)
    assert result["ok"] is True
    assert result["person_node"] is None
    assert result["deadline_node"] is None
    assert len(graph.list_nodes()) == 1
    assert graph.edges() == []


# ── status refresh ────────────────────────────────────────────────────

def test_kept_commitment_updates_node_status():
    ledger = CommitmentLedger()
    added = ledger.add(promisor="owner", promisee="Alex",
                       what="send the deck", deadline=NOW + 86400,
                       source_text="I'll send the deck")
    commitment_id = added.id
    graph = _graph()
    assert sync_ledger_to_graph(ledger, graph) == 1
    node = _node_by_ref(graph, f"commitment:{commitment_id}")
    assert node.attrs["status"] == "open"
    assert node.attrs["closed"] is False

    ledger.mark_kept(commitment_id, evidence="sent")
    # Re-sync must update the SAME node, not create a second one.
    assert sync_ledger_to_graph(ledger, graph) == 1
    assert len([n for n in graph.list_nodes()
                if n.external_ref == f"commitment:{commitment_id}"]) == 1
    node = _node_by_ref(graph, f"commitment:{commitment_id}")
    assert node.attrs["status"] == "kept"
    assert node.attrs["closed"] is True
    assert node.attrs["closed_at"] is not None


def test_broken_commitment_marks_closed_via_direct_sync():
    graph = _graph()
    commitment = Commitment(
        id="cm-broken", promisor="owner", promisee="Sam",
        what="call you", deadline=NOW + 100, status="broken",
        closed_at=NOW, created_at=NOW - 100)
    result = sync_commitment_to_graph(graph, commitment)
    assert result["ok"] is True
    node = _node_by_ref(graph, "commitment:cm-broken")
    assert node.attrs["status"] == "broken"
    assert node.attrs["closed"] is True


# ── ledger sync ───────────────────────────────────────────────────────

def test_sync_ledger_open_plus_closed():
    ledger = CommitmentLedger()
    open_c = ledger.add(promisor="owner", promisee="Alex",
                        what="send the deck", deadline=NOW + 86400)
    kept_c = ledger.add(promisor="owner", promisee="Sam",
                        what="call you", deadline=NOW + 86400)
    ledger.mark_kept(kept_c.id, evidence="called")
    graph = _graph()
    assert sync_ledger_to_graph(ledger, graph) == 2
    assert _node_by_ref(graph, f"commitment:{open_c.id}").attrs["status"] == "open"
    assert _node_by_ref(graph, f"commitment:{kept_c.id}").attrs["status"] == "kept"


def test_sync_from_empty_ledger_is_zero():
    ledger = CommitmentLedger()
    graph = _graph()
    assert sync_ledger_to_graph(ledger, graph) == 0
    assert graph.list_nodes() == []


# ── overdue ───────────────────────────────────────────────────────────

def test_overdue_detection():
    graph = _graph()
    overdue_c = Commitment(
        id="cm-overdue", promisor="owner", promisee="Alex",
        what="send you the deck", deadline=NOW - 3600,
        created_at=NOW - 7200)
    future_c = Commitment(
        id="cm-future", promisor="owner", promisee="Sam",
        what="call you", deadline=NOW + 3600, created_at=NOW)
    kept_c = Commitment(
        id="cm-kept-late", promisor="owner", promisee="Sam",
        what="pay the invoice", deadline=NOW - 7200, status="kept",
        closed_at=NOW - 100, created_at=NOW - 9000)
    no_deadline = Commitment(
        id="cm-nodeadline", promisor="owner", promisee="Sam",
        what="review notes", deadline=None, created_at=NOW)
    for commitment in (overdue_c, future_c, kept_c, no_deadline):
        assert sync_commitment_to_graph(graph, commitment)["ok"] is True

    overdue = overdue_commitments_in_graph(graph, NOW)
    assert [n.external_ref for n in overdue] == ["commitment:cm-overdue"]
    # Kept-but-late is NOT overdue (status filters it out); future and
    # no-deadline are not overdue either.
    assert overdue_commitments_in_graph(graph, NOW - 10000) == []


def test_overdue_empty_graph():
    assert overdue_commitments_in_graph(_graph(), NOW) == []


# ── never raises ──────────────────────────────────────────────────────

def test_never_raises_on_bad_inputs():
    graph = _graph()
    assert sync_commitment_to_graph(None, None)["ok"] is False
    assert sync_commitment_to_graph(graph, None)["ok"] is False
    assert sync_commitment_to_graph(graph, Commitment(id=""))["ok"] is False
    assert sync_ledger_to_graph(None, graph) == 0
    assert sync_ledger_to_graph(CommitmentLedger(), None) == 0
    assert sync_extracted_to_graph({}, None, graph)["ok"] is False
    assert sync_extracted_to_graph({}, CommitmentLedger(), None)["ok"] is False
    assert overdue_commitments_in_graph(None, NOW) == []
    assert overdue_commitments_in_graph(graph, "not-a-time") == []


def test_commitment_without_promisee_still_syncs():
    graph = _graph()
    commitment = Commitment(id="cm-nop", what="finish the report",
                            deadline=NOW + 5000, created_at=NOW)
    result = sync_commitment_to_graph(graph, commitment)
    assert result["ok"] is True
    assert result["person_node"] is None
    assert result["deadline_node"] is not None
