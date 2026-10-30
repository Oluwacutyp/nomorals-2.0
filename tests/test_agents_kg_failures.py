"""Tail-56 pins for nomorals/agents/kg.py —
0 converts / 7 exclusions (67, 94, 586, 617, 648, 678, 732).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

kg.py calls no failures-ledger record function anywhere
(grep-proven: no `record_` call in the file), so NOT-recording
is pinned against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-56 SURVEY.md):
- 67: GraphNode.from_row corrupt properties JSON -> {}; the
  same value a row with no properties yields, and id/label/
  type/counts survive from their own columns (the tail-25
  from_row shape). Nothing is fabricated.
- 94: GraphEdge.from_row — mirror of 67 for edges; relation
  and weight survive from their own columns.
- 586: curate_from_text source-node upsert failure -> the
  zero result dict; every counter in it is literally true
  (nothing was added), and the docstring's contract is
  "Never raises — curation is an enhancement".
- 617: known-hash enrichment failure -> pass; the item is
  still curated, it simply carries no enrichment properties
  and no decodes_to link is claimed.
- 648: per-item curation failure -> continue; added_nodes /
  added_links / items only ever count successes, so the
  returned tallies are literally true.
- 678: knowledge_context failure -> ""; the docstring's
  declared contract — an empty graph, a missing table, or a
  bad config all yield "" so callers can concatenate
  unconditionally. Consumers prepend the block; "" is an
  omission, never a claim.
- 732: register's kg action malformed properties JSON ->
  {}; the tool's lenient parameter parsing (weight/depth/
  limit degrade to defaults the same way), and the returned
  node dict shows properties={} truthfully.
"""

import json
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.kg as kgmod
import nomorals.core.decoder as decmod
from nomorals.agents.failures import recent_failures
from nomorals.agents.kg import GraphEdge, GraphNode, KnowledgeGraph, knowledge_context
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _snap():
    return list(recent_failures(200))


class _CountingLoads:
    """json.loads wrapper logging every argument, delegating."""

    def __init__(self):
        self.args = []
        self.real = json.loads

    def __call__(self, s, *a, **k):
        self.args.append(s)
        return self.real(s, *a, **k)


# ── 67: GraphNode.from_row corrupt properties -> {} ────────────────────

def test_67_node_from_row_corrupt_properties_become_empty():
    before = _snap()
    counter = _CountingLoads()
    row = {"id": "n1", "label": "L", "type": "person",
           "properties": "{corrupt", "created_at": 1.0,
           "last_access": 2.0, "access_count": 3}
    with mock.patch.object(json, "loads", counter):
        node = GraphNode.from_row(row)
    assert "{corrupt" in counter.args  # the parse was actually attempted
    assert node.properties == {}
    assert node.label == "L" and node.access_count == 3  # own columns survive
    ok = GraphNode.from_row({**row, "properties": '{"a": 1}'})
    assert ok.properties == {"a": 1}                     # control
    empty = GraphNode.from_row({**row, "properties": None})
    assert empty.properties == {}                        # genuine empty
    assert _snap() == before


# ── 94: GraphEdge.from_row corrupt properties -> {} ────────────────────

def test_94_edge_from_row_corrupt_properties_become_empty():
    before = _snap()
    counter = _CountingLoads()
    row = {"id": "e1", "src": "n1", "dst": "n2", "relation": "knows",
           "weight": 0.5, "properties": "!!bad", "created_at": 1.0}
    with mock.patch.object(json, "loads", counter):
        edge = GraphEdge.from_row(row)
    assert "!!bad" in counter.args  # the parse was actually attempted
    assert edge.properties == {}
    assert edge.relation == "knows" and edge.weight == 0.5
    ok = GraphEdge.from_row({**row, "properties": '{"via": "x"}'})
    assert ok.properties == {"via": "x"}                 # control
    assert _snap() == before


# ── 586: curate source upsert failure -> zero result ───────────────────

def test_586_curate_source_failure_returns_zero_counts():
    before = _snap()
    db = _db()
    graph = KnowledgeGraph(db)

    class BoomDB:
        def __init__(self, real):
            self.real, self.log = real, []

        def query_one(self, sql, params=()):
            self.log.append(sql)
            raise RuntimeError("kg db boom")

        def __getattr__(self, name):
            return getattr(self.real, name)

    boom = BoomDB(db)
    graph.db = boom
    out = graph.curate_from_text("see https://example.com/x", source="probe")
    assert len(boom.log) == 1  # the source upsert SELECT was attempted
    assert out["added_nodes"] == 0 and out["added_links"] == 0
    assert out["items"] == []
    graph.db = db
    out2 = graph.curate_from_text("see https://example.com/x", source="probe")
    assert out2["added_nodes"] >= 1                      # control
    assert _snap() == before


# ── 617: digest enrichment failure -> item still curated ───────────────

def test_617_digest_enrichment_failure_still_curates_item():
    before = _snap()
    db = _db()
    graph = KnowledgeGraph(db)
    digest = "a" * 64
    attempts = []
    real_ident = decmod.identify_hash

    def boom_ident(h):
        attempts.append(h)
        raise RuntimeError("decoder boom")

    with mock.patch.object(decmod, "identify_hash", boom_ident):
        out = graph.curate_from_text(f"hash {digest} end", source="probe617")
    assert attempts == [digest]  # enrichment was actually attempted
    assert out["added_nodes"] == 1
    node = graph.find_node(digest)
    assert node is not None and "algorithm_candidates" not in node.properties

    db2 = _db()
    graph2 = KnowledgeGraph(db2)
    out2 = graph2.curate_from_text(f"hash {digest} end", source="probe617")
    node2 = graph2.find_node(digest)
    assert out2["added_nodes"] == 1                      # control
    assert node2 is not None and "algorithm_candidates" in node2.properties
    assert real_ident is decmod.identify_hash
    assert _snap() == before


# ── 648: per-item curation failure -> only that item skipped ───────────

def test_648_curate_item_failure_skips_only_that_item():
    before = _snap()
    db = _db()
    graph = KnowledgeGraph(db)
    real_upsert = graph.upsert_node
    calls = []

    def flaky_upsert(label, **k):
        calls.append(label)
        if "bad.example" in label:  # both its url and domain items
            raise RuntimeError("item boom")
        return real_upsert(label, **k)

    with mock.patch.object(graph, "upsert_node", flaky_upsert):
        out = graph.curate_from_text(
            "go https://good.example and https://bad.example",
            source="probe648")
    assert "https://bad.example" in calls  # failing item was attempted
    assert [i["item"] for i in out["items"]] == [
        "https://good.example", "good.example"]
    assert out["added_nodes"] == 2         # counts only successes

    graph2 = KnowledgeGraph(_db())
    out2 = graph2.curate_from_text(
        "go https://good.example and https://bad.example", source="probe648")
    assert out2["added_nodes"] == 4                      # control
    assert _snap() == before


# ── 678: knowledge_context failure -> "" ───────────────────────────────

def test_678_knowledge_context_failure_returns_empty_string():
    before = _snap()

    class BoomCtxDB:
        def __init__(self):
            self.log = []

        def query_one(self, sql, params=()):
            self.log.append(sql)
            raise RuntimeError("no kg table")

        def __getattr__(self, name):
            raise RuntimeError("boom")

    boom = BoomCtxDB()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(reasoning_knowledge="on"), db=boom)
    assert knowledge_context(ctx, "quantum computing research") == ""
    assert boom.log, "the stats query was actually attempted"

    db = _db()
    KnowledgeGraph(db).upsert_node("quantum computing", type="concept")
    ctx2 = SimpleNamespace(
        settings=SimpleNamespace(reasoning_knowledge="on"), db=db)
    res = knowledge_context(ctx2, "quantum computing research")
    assert "quantum computing" in res                    # control
    ctx3 = SimpleNamespace(
        settings=SimpleNamespace(reasoning_knowledge="on"), db=_db())
    assert knowledge_context(ctx3, "quantum computing research") == ""
    assert _snap() == before


# ── 732: register malformed properties JSON -> {} ──────────────────────

def test_732_register_malformed_properties_become_empty():
    before = _snap()
    captured = {}

    class FakeRegistry:
        context = SimpleNamespace(db=_db())

        def register(self, *a, **k):
            def deco(fn):
                captured["fn"] = fn
                return fn
            return deco

    kgmod.register(FakeRegistry())
    fn = captured["fn"]
    counter = _CountingLoads()
    with mock.patch.object(json, "loads", counter):
        res = fn(action="add", label="thing732", properties="{not json")
    assert "{not json" in counter.args  # the parse was actually attempted
    assert res["ok"] is True and res["node"]["properties"] == {}
    res2 = fn(action="add", label="thing732b", properties='{"k": 5}')
    assert res2["node"]["properties"] == {"k": 5}        # control
    assert _snap() == before
