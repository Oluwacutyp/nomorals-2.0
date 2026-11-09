"""Mongo program P3: domain-port conformance.

Every P3 domain port is exercised through its PUBLIC api on three rungs:

* ``native``     — the SQLite default (no store attached): the historical
                   SQL path, byte-identical behavior;
* ``sqlite_doc`` — a SQLiteDocumentStore attached to the same Database:
                   the seam over the local backend;
* ``mongo``      — a real MongoDocumentStore against a throwaway mongod
                   replica set started by the fixture (skipped when no
                   mongod binary is available).

On the two seam rungs each domain also gets ONE raw read-back proving the
documents in the store are the carry-v1 shape for the same data (the P2
mapping): ``_id`` is the stringified primary key, an ``expires_at`` data
column rides as ``sql_expires_at``, JSON-text columns stay strings.

Plus: carry round-trip pins and the completeness pin — every table the
P3 census assigned is either in ``carry.PORTED_TABLES`` or in
``carry.PORT_RESIDUE`` with its reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.storage import carry
from nomorals.storage.db import Database
from nomorals.storage.docstore import (
    MongoDocumentStore,
    SQLiteDocumentStore,
)

pymongo = pytest.importorskip("pymongo")


# ── mongod fixture (the test_docstore.py pattern) ────────────────────────────

def _find_mongod() -> str | None:
    env = os.environ.get("NM_MONGOD_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("mongod")
    if found:
        return found
    candidate = Path.home() / "workspace" / "tools" / "mongodb" / "bin" / "mongod"
    return str(candidate) if candidate.exists() else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def mongo_uri(tmp_path_factory) -> str:
    binary = _find_mongod()
    if binary is None:
        pytest.skip("no mongod binary available for the mongo conformance run")
    port = _free_port()
    dbpath = tmp_path_factory.mktemp("mongodata")
    log = open(tmp_path_factory.mktemp("mongolog") / "mongod.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [binary, "--dbpath", str(dbpath), "--bind_ip", "127.0.0.1",
         "--port", str(port), "--replSet", "rs0",
         "--wiredTigerCacheSizeGB", "0.25"],
        stdout=log, stderr=log,
    )
    base = f"mongodb://127.0.0.1:{port}"
    try:
        direct = pymongo.MongoClient(base + "/?directConnection=true",
                                     serverSelectionTimeoutMS=500)
        deadline = time.time() + 30
        while True:
            try:
                direct.admin.command("ping")
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("mongod did not come up") from None
                time.sleep(0.25)
        with contextlib.suppress(pymongo.errors.OperationFailure):
            direct.admin.command("replSetInitiate")
        client = pymongo.MongoClient(base + "/?replicaSet=rs0",
                                     serverSelectionTimeoutMS=500)
        deadline = time.time() + 30
        while True:
            try:
                if client.admin.command("hello").get("isWritablePrimary"):
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError("mongod replica set did not elect a primary")
            time.sleep(0.25)
        yield base + "/?replicaSet=rs0"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


# ── rung plumbing ────────────────────────────────────────────────────────────

RUNGS = ["native", "sqlite_doc", "mongo"]


class Rung:
    """One database on one rung, plus raw read-back into the store."""

    def __init__(self, kind: str, mongo_uri: str | None) -> None:
        self.kind = kind
        self.db = Database(":memory:")
        self.db.migrate()
        self.mongo_dbname: str | None = None
        self._client = None
        if kind == "sqlite_doc":
            self.db.document_store = SQLiteDocumentStore(self.db)
        elif kind == "mongo":
            assert mongo_uri is not None
            self.mongo_dbname = f"p3_{uuid.uuid4().hex[:12]}"
            self.db.document_store = MongoDocumentStore(
                mongo_uri, self.mongo_dbname,
                server_selection_timeout_ms=3000)
            self._client = pymongo.MongoClient(mongo_uri)

    def raw(self, collection: str, doc_id: str) -> dict | None:
        """The stored document for ``doc_id`` on a seam rung, read back
        WITHOUT the organ — the carry-v1 shape evidence."""
        if self.kind == "sqlite_doc":
            return self.db.document_store.get(collection, doc_id)
        if self.kind == "mongo":
            assert self._client is not None and self.mongo_dbname
            return self._client[self.mongo_dbname][collection].find_one(
                {"_id": doc_id})
        raise AssertionError("raw read-back exists only on seam rungs")

    @property
    def seam(self) -> bool:
        return self.kind != "native"


@pytest.fixture(params=RUNGS)
def rung(request, mongo_uri) -> Rung:
    if request.param == "mongo":
        return Rung("mongo", mongo_uri)
    return Rung(request.param, None)


def _ctx(db, **attrs):
    base = {"db": db, "settings": SimpleNamespace(
        workspace_dir="", timezone="UTC", home="",
        trading=SimpleNamespace(live_enabled=True, unlock_ttl_hours=1.0,
                                max_daily_loss_pct=3.0),
        news=SimpleNamespace(feeds=""),
        briefing=SimpleNamespace(topics=[]),
        research=SimpleNamespace(domains="lifestyle,tech,cyber"),
        improvement=SimpleNamespace(tolerance=0.05),
        tools=SimpleNamespace(user_agent="p3-test", http_timeout=5.0,
                              proxy_url=""),
    )}
    base.update(attrs)
    return SimpleNamespace(**base)


# ── conversations (storage/conversations.py) ─────────────────────────────────

def test_conversations_conformance(rung: Rung):
    from nomorals.storage.conversations import ConversationLog

    log = ConversationLog(rung.db)
    with log.transaction():
        log.ensure_conversation("c1", title="First", agent="devon",
                                channel="telegram", now=100.0)
        log.append_message(row_id="m1", conversation_id="c1", role="user",
                           content="hello", name="death", created_at=101.0)
        log.append_message(row_id="m2", conversation_id="c1",
                           role="assistant", content="hi there",
                           created_at=102.0)
    # the CLI's statement never retitles an existing conversation
    log.ensure_conversation("c1", title="Changed", agent="partner",
                            channel="local", now=103.0, bump_title=False)
    assert log.get_conversation("c1")["title"] == "First"
    # the brain's statement does (non-empty title wins)
    log.ensure_conversation("c1", title="Retitled", agent="devon",
                            channel="telegram", now=104.0)
    assert log.get_conversation("c1")["title"] == "Retitled"
    log.ensure_conversation("c2", title="", agent="partner",
                            channel="local", now=105.0, bump_title=False)

    hist = log.history("c1", limit=10)
    assert [m["content"] for m in hist] == ["hi there", "hello"]
    assert [m["content"] for m in log.messages_for("c1")] == [
        "hello", "hi there"]
    after = log.messages_after("c1", 101.0, limit=10)
    assert [m["content"] for m in after] == ["hi there"]
    counts = log.participant_counts("c1", limit=10)
    assert counts == [{"name": "death", "n": 1}]
    assert log.recent_user_contents("c1", limit=5) == ["hello"]
    assert {c["id"] for c in log.conversations()} == {"c1", "c2"}
    assert [m["id"] for m in log.messages_since(101.5)] == ["m2"]
    assert [m["id"] for m in log.recent_messages(limit=1)] == ["m2"]

    chat = log.upsert_chat(chat_id="chat-9", platform="telegram",
                           platform_chat_id="999", kind="group",
                           title="Peer", peer="peer-1", is_owner=0,
                           in_us=1, now=110.0)
    assert chat is not None and chat["title"] == "Peer"
    log.upsert_chat(chat_id="chat-9", platform="telegram",
                    platform_chat_id="999", kind="dm", title="",
                    peer="", is_owner=0, in_us=1, now=111.0)
    chat = log.get_chat("chat-9")
    assert chat is not None and chat["title"] == "Peer"
    assert chat["kind"] == "dm" and chat["last_active"] == 111.0
    assert log.latest_chat()["id"] == "chat-9"
    log.touch_chat("chat-9", 112.0)
    assert log.get_chat("chat-9")["last_active"] == 112.0

    log.rekey("c2", "c2-renamed")
    assert log.get_conversation("c2") is None
    assert log.get_conversation("c2-renamed") is not None

    if rung.seam:
        doc = rung.raw("conversations", "c1")
        assert doc is not None and doc["_id"] == "c1"
        assert doc["title"] == "Retitled" and doc["agent"] == "devon"
        mdoc = rung.raw("messages", "m2")
        assert mdoc is not None and mdoc["conversation_id"] == "c1"
        assert mdoc["content"] == "hi there"
        cdoc = rung.raw("chats", "chat-9")
        assert cdoc is not None and cdoc["chat_id"] == "999"


# ── side chats (social/chat/side_chats.py) ───────────────────────────────────

def test_side_chats_conformance(rung: Rung):
    from nomorals.social.chat.side_chats import SideChatManager

    mgr = SideChatManager(rung.db)

    async def scenario():
        thread = await mgr.create(
            "user-1", "Mongo thread", description="d", tags=["x"])
        await mgr.add_message(thread.thread_id, "user", "first note")
        await mgr.add_message(thread.thread_id, "assistant", "second note")
        other = await mgr.create("user-1", "Other thread")
        await mgr.add_message(other.thread_id, "user", "unrelated")
        history = await mgr.get_history(thread.thread_id)
        assert [m.content for m in history] == ["first note", "second note"]
        listed = await mgr.list_threads("user-1")
        assert {t.thread_id for t in listed} == {thread.thread_id,
                                                 other.thread_id}
        found = await mgr.search("user-1", "second")
        assert [m.content for m in found] == ["second note"]
        await mgr.archive(other.thread_id)
        listed = await mgr.list_threads("user-1")
        assert [t.thread_id for t in listed] == [thread.thread_id]
        await mgr.delete(other.thread_id)
        assert await mgr.get_history(other.thread_id) == []
        return thread

    thread = asyncio.run(scenario())
    if rung.seam:
        doc = rung.raw("threads", thread.thread_id)
        assert doc is not None and doc["topic"] == "Mongo thread"
        assert doc["message_count"] == 2 and doc["is_archived"] == 0


# ── core memory blocks (memory/manager.py CoreBlocks) ────────────────────────

def test_core_blocks_conformance(rung: Rung):
    from nomorals.memory.manager import CoreBlocks

    blocks = CoreBlocks(rung.db)
    assert blocks.get("persona") == ""
    out = blocks.replace("persona", "I am Devon.")
    assert out["ok"] is True
    blocks.insert("persona", "I build things.")
    assert blocks.get("persona") == "I am Devon.\nI build things."
    assert blocks.exists("persona") is True
    assert blocks.all() == {"persona": "I am Devon.\nI build things."}
    blocks.rethink("user", "Owner: death.")
    assert blocks.get("user") == "Owner: death."
    assert blocks.clear("user") is True
    assert blocks.get("user") == ""
    if rung.seam:
        doc = rung.raw("memory_core_blocks", "persona")
        assert doc is not None and doc["_id"] == "persona"
        assert doc["text"] == "I am Devon.\nI build things."


# ── trading (tools/trading.py) ───────────────────────────────────────────────

def test_trading_conformance(rung: Rung):
    from nomorals.tools import trading

    ctx = _ctx(rung.db)
    started = trading.paper_start(ctx, "BTCUSDT", "crypto", 5000.0)
    assert started["ok"] is True and started["resumed"] is False
    sid = started["session_id"]
    again = trading.paper_start(ctx, "BTCUSDT", "crypto", 5000.0)
    assert again["resumed"] is True and again["session_id"] == sid
    state = trading._paper_broker_state(rung.db, sid)
    assert state["cash"] == 5000.0
    stopped = trading.paper_stop(ctx, sid)
    assert stopped["ok"] is True

    unlock = trading.live_unlock(ctx, "I understand")
    assert unlock["ok"] is True
    valid = trading._valid_unlock(rung.db)
    assert valid is not None and valid["id"] == unlock["unlock_id"]
    killed = trading._auto_kill(ctx, "conformance")
    assert killed["ok"] is True
    assert trading._valid_unlock(rung.db) is None

    if rung.seam:
        doc = rung.raw("paper_sessions", sid)
        assert doc is not None and doc["status"] == "closed"
        assert doc["symbol"] == "BTCUSDT"
        jdoc = rung.raw("trade_journal", killed["journal_id"])
        assert jdoc is not None and jdoc["kind"] == "live"
        assert jdoc["side"] == "kill"


# ── watchers (agents/watchers.py WatcherStore) ───────────────────────────────

def test_watchers_conformance(rung: Rung):
    from nomorals.agents import watchers as W

    store = W.WatcherStore(rung.db)
    watcher = W.Watcher(id="w1", name="gold-watch", kind="url",
                        target={"url": "https://example.com"},
                        interval_s=60.0, expires_at=9_999_999_999.0,
                        created_at=100.0, updated_at=100.0)
    store.create(watcher)
    got = store.get("w1")
    assert got is not None and got.name == "gold-watch"
    assert store.get("gold-watch") is not None
    assert [w.id for w in store.list()] == ["w1"]
    assert [w.id for w in store.list("active")] == ["w1"]
    assert [w.id for w in store.due(200.0)] == ["w1"]
    got.last_value = "v2"
    store.save(got)
    assert store.get("w1").last_value == "v2"

    store.record_check("w1", changed=True, summary="moved",
                       value_hash="h1", now=300.0)
    store.record_check("w1", changed=False, summary="same",
                       value_hash="h1", now=400.0)
    history = store.check_history("w1")
    assert [h["checked_at"] for h in history] == [400.0, 300.0]
    assert history[1]["changed"] == 1

    aid = store.record_alert("w1", severity="high", channel="owner",
                             status="held", title="t", body="b", now=500.0)
    assert [a["id"] for a in store.held_alerts()] == [aid]
    store.mark_digested([aid])
    assert store.held_alerts() == []
    assert [a["id"] for a in store.alert_log("w1")] == [aid]

    store.set_state("digest_day", "2026-11-09")
    assert store.get_state("digest_day") == "2026-11-09"
    assert store.get_state("missing", "fallback") == "fallback"

    assert store.delete("w1") is True
    assert store.get("w1") is None
    assert store.alert_log("w1") == []

    if rung.seam:
        store.create(watcher)
        store.record_check("w1", changed=True, summary="moved",
                           value_hash="h1", now=300.0)
        doc = rung.raw("watchers", "w1")
        assert doc is not None and doc["name"] == "gold-watch"
        # expires_at is a DATA column: carried as sql_expires_at, and the
        # store-managed expiry stays empty.
        assert doc["sql_expires_at"] == 9_999_999_999.0
        assert doc["expires_at"] is None
        sdoc = rung.raw("watcher_state", "digest_day")
        assert sdoc is not None and sdoc["value"] == "2026-11-09"


# ── monitors (agents/monitor.py) ─────────────────────────────────────────────

def test_monitors_conformance(rung: Rung, tmp_path):
    from nomorals.agents.monitor import MonitorAgent

    target_file = tmp_path / "watched.txt"
    target_file.write_text("v1", encoding="utf-8")
    ctx = _ctx(rung.db, settings=SimpleNamespace(
        workspace_dir=str(tmp_path), timezone="UTC"))
    agent = MonitorAgent(ctx)
    added = agent.add(str(target_file), kind="file", interval=60.0)
    assert added["created"] is True
    mid = added["id"]
    again = agent.add(str(target_file), kind="file", interval=120.0)
    assert again["created"] is False and again["id"] == mid
    rows = agent.list()
    assert len(rows) == 1 and rows[0]["interval_s"] == 120.0
    assert agent.status()["total"] == 1

    ticked = agent.tick(now=time.time() + 1000.0)
    assert ticked["checked"] == 1 and ticked["errors"] == []
    target_file.write_text("v2-longer", encoding="utf-8")
    ticked = agent.tick(now=time.time() + 2000.0)
    assert ticked["checked"] == 1
    assert len(ticked["changed"]) == 1

    off = agent.set_enabled(str(target_file), False)
    assert off is not None and off["enabled"] is False
    assert agent.remove(str(target_file)) is True
    assert agent.list() == []

    if rung.seam:
        readded = agent.add(str(target_file), kind="file", interval=60.0)
        doc = rung.raw("monitors", readded["id"])
        assert doc is not None and doc["target"] == str(target_file)
        assert doc["kind"] == "file"


# ── news (agents/news.py) ────────────────────────────────────────────────────

def _feed_item(url: str, title: str, ts: float) -> dict:
    """A feed-entry dict, the shape _store_fresh consumes (the organ
    generates the row id and stamps created_at itself)."""
    return {"source": "feed-1", "source_name": "Pub",
            "source_url": "https://pub.example", "title": title,
            "url": url, "summary": "s", "source_ts": ts}


def test_news_conformance(rung: Rung):
    from nomorals.agents.news import NewsAgent

    agent = NewsAgent(_ctx(rung.db))
    stored = agent._store_fresh([
        _feed_item("https://a.example/1", "Gold rallies", 1000.0),
        _feed_item("https://a.example/2", "AI advances", 1001.0)])
    ids = {r["url"]: r["id"] for r in stored}
    assert len(ids) == 2
    # INSERT OR IGNORE: the same url is skipped on the second pass.
    agent._store_fresh([_feed_item("https://a.example/1", "Dup", 1002.0)])
    recent = agent.recent(10)
    assert {r["url"] for r in recent} == {"https://a.example/1",
                                          "https://a.example/2"}
    assert {r["url"] for r in agent.undelivered()} == {
        "https://a.example/1", "https://a.example/2"}
    status = agent.status()
    assert status["stored"] == 2
    assert status["by_state"].get("undelivered") == 2

    assert agent.mark_delivered(["https://a.example/1"]) == 1
    assert [r["url"] for r in agent.undelivered()] == ["https://a.example/2"]
    # mark_delivered marks DIGESTED, not PUSHED: today's push count
    # stays zero on every rung.
    assert agent._pushes_today() == 0

    profile = dict(agent._load_profile())
    profile["beats"] = [{"topic": "gold", "weight": 1.0, "hits": 0}]
    agent._save_profile(profile)
    after = agent._load_profile()
    assert after["beats"][0]["topic"] == "gold"

    if rung.seam:
        doc = rung.raw("news_items", ids["https://a.example/1"])
        assert doc is not None and doc["title"] == "Gold rallies"
        assert doc["delivery_state"] == "digested"
        assert doc["publisher"] == "Pub"
        pdoc = rung.raw("news_interest", "profile")
        assert pdoc is not None and "gold" in pdoc["value"]


# ── research jobs (agents/search/jobs.py) ────────────────────────────────────

def test_research_jobs_conformance(rung: Rung):
    from nomorals.agents.search import jobs

    ctx = _ctx(rung.db)
    job = {"id": "job-1", "query": "gold outlook", "mode": "deep",
           "params": {"max": 3}, "state": "queued", "progress": "",
           "report": {}, "error": "", "created_at": 100.0}
    jobs._store(ctx, job)
    loaded = jobs._load(ctx, "job-1")
    assert loaded is not None and loaded["query"] == "gold outlook"
    assert loaded["params"] == {"max": 3}
    jobs._set(ctx, "job-1", state="running", progress="half")
    moved = jobs._cas_state(ctx, "job-1", {"running"}, "done",
                            report={"summary": "ok"})
    assert moved is not None and moved["state"] == "done"
    assert jobs._cas_state(ctx, "job-1", {"running"}, "queued") is None
    listed = jobs._list(ctx)
    assert [j["id"] for j in listed] == ["job-1"]
    if rung.seam:
        doc = rung.raw("research_jobs", "job-1")
        assert doc is not None and doc["state"] == "done"
        # params/result ride as JSON text, exactly as the SQL row stores
        assert isinstance(doc["params"], str)
        assert isinstance(doc["result"], str)


# ── search engine (agents/search/engine.py) ──────────────────────────────────

def test_search_engine_conformance(rung: Rung):
    from nomorals.agents.search.engine import SearchEngine

    engine = SearchEngine(_ctx(rung.db))
    engine._cache_put({"url": "https://x.example/a", "title": "A",
                       "text": "body text"})
    cached = engine._cache_get("https://x.example/a")
    assert cached is not None and cached["text"] == "body text"
    engine._cache_put({"url": "https://x.example/a", "title": "A2",
                       "text": "new body"})
    assert engine._cache_get("https://x.example/a")["title"] == "A2"
    engine._journal({"id": "s1", "query": "gold", "mode": "quick",
                     "results": [{"url": "https://x.example"}],
                     "summary": "found", "seconds": 0.5})
    history = engine.history()
    assert [h["id"] for h in history] == ["s1"]
    assert history[0]["query"] == "gold"
    if rung.seam:
        doc = rung.raw("page_cache", "https://x.example/a")
        assert doc is not None and doc["_id"] == "https://x.example/a"
        assert doc["text"] == "new body"
        ldoc = rung.raw("search_log", "s1")
        assert ldoc is not None and ldoc["summary"] == "found"


# ── research log (agents/researcher.py) ──────────────────────────────────────

def test_research_log_conformance(rung: Rung):
    from nomorals.agents.researcher import ResearchAgent
    from nomorals.storage.rowstore import attached_rows

    # Seed one proposal row in carry-v1 shape (the journal insert inside
    # run_cycle writes exactly this row dict through the same seam).
    seed = {"id": "prop-1", "domain": "tech", "topic": "compilers",
            "digest": "d", "suggestion": "adopt the thing",
            "sources": "[]", "delivered": 0, "created_at": 500.0,
            "score": 0.9, "score_detail": "{}", "status": "pending"}
    if rung.seam:
        table = attached_rows(rung.db, "research_log")
        assert table is not None
        table.put(seed)
    else:
        rung.db.execute(
            "INSERT INTO research_log (id, domain, topic, digest, "
            "suggestion, sources, delivered, created_at, score, "
            "score_detail, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            tuple(seed[k] for k in ("id", "domain", "topic", "digest",
                                    "suggestion", "sources", "delivered",
                                    "created_at", "score", "score_detail",
                                    "status")))
    agent = ResearchAgent(_ctx(rung.db))
    proposals = agent.list_proposals(limit=5)
    assert [p["id"] for p in proposals] == ["prop-1"]
    assert agent._recent_topics(limit=5) == ["compilers"]
    assert agent._used_topics("tech") == {"compilers"}
    row, err = agent.resolve_proposal_detailed("prop-1")
    assert err == "" and row is not None and row["topic"] == "compilers"
    row2, _ = agent.resolve_proposal_detailed("latest")
    assert row2 is not None and row2["id"] == "prop-1"
    assert agent._set_status("prop-1", "approved") is True
    after = agent.list_proposals(limit=5, status="approved")
    assert [p["id"] for p in after] == ["prop-1"]
    if rung.seam:
        doc = rung.raw("research_log", "prop-1")
        assert doc is not None and doc["status"] == "approved"


# ── training registry (training/registry.py) ─────────────────────────────────

def test_training_registry_conformance(rung: Rung):
    from nomorals.training.registry import RunStatus, TrainingRegistry

    reg = TrainingRegistry(rung.db)
    run = reg.start(name="run-a", base_model="phi", config={"lr": 0.1})
    assert reg.get(run.id).name == "run-a"
    assert reg.get("run-a").id == run.id
    assert reg.get(run.id).status == RunStatus.RUNNING
    reg.complete(run, metrics={"loss": 0.25, "steps": 10})
    after = reg.get(run.id)
    assert after.status == RunStatus.DONE
    assert after.metrics == {"loss": 0.25, "steps": 10}
    assert after.config == {"lr": 0.1}
    other = reg.start(name="run-b", base_model="phi", config={})
    listed = reg.list()
    assert {r.id for r in listed} == {run.id, other.id}
    assert [r.id for r in reg.interrupted()] == [other.id]
    stats = reg.stats()
    assert stats["total"] == 2
    assert stats["by_status"].get(RunStatus.DONE) == 1
    if rung.seam:
        doc = rung.raw("training_runs", run.id)
        assert doc is not None and doc["status"] == RunStatus.DONE
        assert doc["config"] == '{"lr": 0.1}'
        assert isinstance(doc["metrics"], str)


# ── dataset registry (training/dataset.py) ───────────────────────────────────

def test_dataset_registry_conformance(rung: Rung, tmp_path):
    from nomorals.core.errors import NotFound
    from nomorals.training.dataset import DatasetRegistry

    data_file = tmp_path / "ds.jsonl"
    data_file.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    reg = DatasetRegistry(rung.db)
    ds = reg.register("ds-a", data_file, schema_={"fields": ["a"]})
    assert ds.rows == 3
    assert reg.get(ds.id).name == "ds-a"
    assert reg.get("ds-a").id == ds.id
    assert [d.id for d in reg.list()] == [ds.id]
    stats = reg.stats()
    assert stats["datasets"] == 1 and stats["rows"] == 3
    assert reg.delete(ds.id) == 1
    with pytest.raises(NotFound):
        reg.get(ds.id)
    if rung.seam:
        ds2 = reg.register("ds-b", data_file)
        doc = rung.raw("datasets", ds2.id)
        assert doc is not None and doc["name"] == "ds-b"
        assert isinstance(doc["schema_"], str)
        assert doc["rows"] == 3


# ── benchmark (agents/benchmark.py) ──────────────────────────────────────────

def test_benchmark_conformance(rung: Rung):
    from types import SimpleNamespace as NS

    from nomorals.agents.benchmark import get_run, list_runs, save_run

    score = NS(name="accuracy", passed=8, total=10,
               as_dict=lambda: {"name": "accuracy", "passed": 8,
                                "total": 10})
    report = NS(run_id="bench-1", scoreboard="sb", suite="core",
                mode="offline", provider="unit", measurable=True,
                overall=0.8, seconds=1.5, scores={"accuracy": score})
    assert save_run(rung.db, report) == "bench-1"
    runs = list_runs(rung.db)
    assert [r["id"] for r in runs] == ["bench-1"]
    assert runs[0]["overall"] == 0.8
    assert runs[0]["passed"] == 8 and runs[0]["total"] == 10
    full = get_run(rung.db, "bench-1")
    assert full is not None
    assert full["dimensions"] == {"accuracy": {"name": "accuracy",
                                               "passed": 8, "total": 10}}
    assert get_run(rung.db, "missing") is None
    if rung.seam:
        doc = rung.raw("benchmark_runs", "bench-1")
        assert doc is not None and doc["suite"] == "core"
        assert doc["measurable"] == 1
        assert isinstance(doc["dimensions"], str)


# ── skill canary (agents/skill_canary.py) ────────────────────────────────────

def test_skill_canary_conformance(rung: Rung):
    from nomorals.agents.skill_canary import CanaryRollout

    roll = CanaryRollout(_ctx(rung.db))
    vhash = roll.record_version("skill-x", "body-one")
    assert roll.latest_hash("skill-x") == vhash
    assert roll.get_body("skill-x", vhash) == "body-one"
    versions = roll.versions("skill-x")
    assert [v["version_hash"] for v in versions] == [vhash]
    started = roll.start("skill-x", "body-two", baseline_body="body-one")
    assert started["ok"] is True
    run_id = started["id"]
    active = roll.active("skill-x")
    assert active is not None and active.id == run_id
    assert active.status == "running"
    roll.observe("skill-x", started["canary_hash"], True)
    roll.observe("skill-x", started["canary_hash"], False)
    roll.observe("skill-x", started["baseline_hash"], True)
    counts = roll._counts(active)
    assert counts == {"canary": (1, 2), "baseline": (1, 1)}
    history = roll.history()
    assert [h.id for h in history] == [run_id]
    if rung.seam:
        doc = rung.raw("canary_runs", run_id)
        assert doc is not None and doc["skill_name"] == "skill-x"
        assert doc["status"] == "running"


# ── improvement loop (agents/improvement.py) ─────────────────────────────────

def test_improvement_conformance(rung: Rung):
    from nomorals.agents.improvement import ImprovementLoop

    loop = ImprovementLoop(_ctx(rung.db))
    rec = loop._record("cyc-1", "docs", 0.5, 0.4, -0.1, "p-1", "edit",
                       "why", "what", "pending", {})
    assert rec.id == "cyc-1"
    loop._record("cyc-2", "docs", 0.5, 0.3, -0.2, "p-2", "edit",
                 "why2", "what2", "regressed", {})
    pending = loop.pending_approvals()
    assert [p.proposal_id for p in pending] == ["p-1"]
    assert loop._recently_regressed("docs") is True
    assert loop._recently_regressed("other") is False
    assert loop.deny("cyc-1") is True
    assert loop.pending_approvals() == []
    history = loop.history()
    assert {h.id for h in history} == {"cyc-1", "cyc-2"}
    denied = [h for h in history if h.id == "cyc-1"][0]
    # deny lands as status 'skipped' with action 'denied' — the organ's
    # own vocabulary, identical on every rung
    assert denied.status == "skipped" and denied.action == "denied"
    if rung.seam:
        doc = rung.raw("improvement_runs", "cyc-1")
        assert doc is not None and doc["status"] == "skipped"
        assert doc["action"] == "denied"
        assert doc["dimension"] == "docs"


# ── evolution outcomes (agents/evolution.py) ─────────────────────────────────

def test_evolution_outcomes_conformance(rung: Rung):
    from nomorals.agents.evolution import EvolutionAgent, _outcomes_table

    agent = EvolutionAgent(_ctx(rung.db))
    one = agent.record_outcome("prop-aaaa", applied=True, reason="gate ok",
                               commit="abc123",
                               before={"tests": 10, "prod_lines": 100},
                               after={"tests": 12, "prod_lines": 120})
    two = agent.record_outcome("prop-bbbb", applied=False,
                               reason="gate failed")
    assert one["applied"] is True and two["applied"] is False
    # the organ's own context read sees both, newest first
    context_text = agent._outcome_context("anything")
    assert "Recent evolution outcomes" in context_text
    assert "applied: " in context_text and "reverted: " in context_text
    if rung.seam:
        table = _outcomes_table(rung.db)
        assert table is not None
        rows = table.all()
        assert {r["proposal_id"] for r in rows} == {"prop-aaaa", "prop-bbbb"}
        doc = rung.raw("evolution_outcomes", one["id"])
        assert doc is not None and doc["applied"] == 1
        assert doc["tests_after"] == 12 and doc["commit_id"] == "abc123"
    else:
        rows = rung.db.query(
            "SELECT proposal_id, applied FROM evolution_outcomes")
        assert {r["proposal_id"] for r in rows} == {"prop-aaaa", "prop-bbbb"}


# ── projects acceptance runs (agents/projects.py) ────────────────────────────

def test_projects_acceptance_conformance(rung: Rung):
    from types import SimpleNamespace as NS

    from nomorals.agents.projects import ProjectManager

    mgr = ProjectManager(_ctx(rung.db))
    project = NS(id="proj-1")
    mgr._record_acceptance_run(project, "s1", "make check", "fine output")
    mgr._record_acceptance_run(project, "s2", "make test", "green output")
    history = mgr.acceptance_history("proj-1")
    assert [h["cmd"] for h in history] == ["make check", "make test"]
    assert all(h["ok"] == 1 for h in history)
    assert all(len(h["output_hash"]) == 16 for h in history)
    prior = mgr._prior_acceptance_cmds(project, exclude_step_id="s2")
    assert prior == ["make check"]
    if rung.seam:
        doc = rung.raw("acceptance_runs", history[0]["id"])
        assert doc is not None and doc["project_id"] == "proj-1"
        assert doc["cmd"] == "make check"


# ── trial flow (agents/trial/flow.py) ────────────────────────────────────────

def test_trial_flow_conformance(rung: Rung, tmp_path):
    from nomorals.agents.trial.flow import TrialFlow

    settings = SimpleNamespace(home=str(tmp_path), timezone="UTC",
                               workspace_dir=str(tmp_path))
    ctx = _ctx(rung.db, settings=settings)
    now = time.time()
    flow = TrialFlow(ctx)
    flow._assist_runs["run-1"] = {
        "platform": "telegram", "chat_key": "chat-1",
        "started": now - 60.0, "state": "done", "note": "finished"}
    flow._persist_assist_run("run-1")
    flow._sms_watches["watch-1"] = {
        "number": "+100", "number_info": {"cc": "NG"},
        "chat_key": "chat-1", "started": now - 100.0,
        "deadline": now - 10.0, "timeout": 30.0, "state": "watching",
        "code": "", "note": ""}
    flow._persist_sms_watch("watch-1")

    # a fresh flow on the same db reloads both durable stores
    flow2 = TrialFlow(ctx)
    assert "run-1" in flow2._assist_runs
    assert flow2._assist_runs["run-1"]["state"] == "done"
    assert "watch-1" in flow2._sms_watches
    assert flow2._sms_watches["watch-1"]["number_info"] == {"cc": "NG"}

    # the watch is past its deadline: recovery closes it as timeout
    recovered = TrialFlow.recover_interrupted_watches(ctx)
    assert [w["watch_id"] for w in recovered] == ["watch-1"]
    assert recovered[0]["state"] == "timeout"
    if rung.seam:
        doc = rung.raw("trial_assist_runs", "run-1")
        assert doc is not None and doc["state"] == "done"
        assert doc["platform"] == "telegram"
        wdoc = rung.raw("trial_sms_watches", "watch-1")
        assert wdoc is not None and wdoc["state"] == "timeout"
        assert isinstance(wdoc["number_info"], str)  # JSON text, as SQL


# ── deep research journal (agents/search/deep.py) ────────────────────────────

def test_deep_research_journal_conformance(rung: Rung):
    from nomorals.agents.search.deep import DeepResearcher
    from nomorals.agents.search.engine import SearchEngine

    researcher = DeepResearcher(_ctx(rung.db))
    researcher._journal({"id": "deep-1", "query": "gold deep dive",
                         "mode": "deep",
                         "results": [{"url": "https://x.example"}],
                         "summary": "two findings", "seconds": 1.25})
    # the journal lands in search_log, readable through the engine
    history = SearchEngine(_ctx(rung.db)).history()
    assert [h["id"] for h in history] == ["deep-1"]
    assert history[0]["summary"] == "two findings"
    if rung.seam:
        doc = rung.raw("search_log", "deep-1")
        assert doc is not None and doc["mode"] == "deep"
        assert isinstance(doc["results"], str)  # JSON text, as the SQL row


# ── media library + playback queue (media/) ──────────────────────────────────

def test_media_library_conformance(rung: Rung, tmp_path):
    from nomorals.core.errors import ToolError
    from nomorals.media.library import MusicLibrary

    ctx = _ctx(rung.db, settings=SimpleNamespace(
        workspace_dir=str(tmp_path)))
    lib = MusicLibrary(ctx)
    lib.create_playlist("gym")
    with pytest.raises(ToolError):
        lib.create_playlist("gym")
    lib.playlist_add("gym", "https://cdn.example/a.mp3",
                     "https://cdn.example/b.mp3")
    lib.rename_playlist("gym", "gym2")
    info = lib.playlist("gym2")
    assert info["name"] == "gym2" and info["tracks"] == 2
    assert [t["path"] for t in info["items"]] == [
        "https://cdn.example/a.mp3", "https://cdn.example/b.mp3"]
    lib.record_played("https://cdn.example/a.mp3", title="A", artist="Art")
    again = lib.record_played("https://cdn.example/a.mp3", title="A",
                              artist="Art")
    assert again == {"recorded": False, "reason": "dedup"}
    history = lib.history(10)
    assert len(history) == 1  # the repeat play inside the window is skipped
    top = lib.top_played(5)
    assert top[0]["path"] == "https://cdn.example/a.mp3"
    assert top[0]["plays"] == 1
    lib.like("https://cdn.example/a.mp3", title="A")
    assert lib.is_liked("https://cdn.example/a.mp3") is True
    assert [f["path"] for f in lib.favorites()] == [
        "https://cdn.example/a.mp3"]
    found = lib.search("cdn.example")
    assert {f["path"] for f in found["favorites"]} == {
        "https://cdn.example/a.mp3"}
    assert {f["path"] for f in found["history"]} == {
        "https://cdn.example/a.mp3"}
    stats = lib.stats()
    assert stats["playlists"] == 1 and stats["favorites"] == 1
    lib.unlike("https://cdn.example/a.mp3")
    assert lib.is_liked("https://cdn.example/a.mp3") is False
    lib.delete_playlist("gym2")
    assert lib.playlists() == []
    if rung.seam:
        doc = rung.raw("media_favorites", "https://cdn.example/a.mp3")
        assert doc is None  # unliked above
        lib.like("https://cdn.example/a.mp3", title="A")
        doc = rung.raw("media_favorites", "https://cdn.example/a.mp3")
        assert doc is not None and doc["_id"] == "https://cdn.example/a.mp3"
        assert doc["title"] == "A"


def test_playback_queue_conformance(rung: Rung, tmp_path):
    from nomorals.media.playback import PlaybackEngine

    ctx = _ctx(rung.db, settings=SimpleNamespace(
        workspace_dir=str(tmp_path)))
    engine = PlaybackEngine(ctx)
    engine._enqueue("https://cdn.example/1.mp3", "url", "One")
    engine._enqueue("https://cdn.example/2.mp3", "url", "Two")
    engine._enqueue("https://cdn.example/3.mp3", "url", "Three")
    rows = engine._queue_rows()
    assert [r["path"] for r in rows] == [
        "https://cdn.example/1.mp3", "https://cdn.example/2.mp3",
        "https://cdn.example/3.mp3"]
    engine.remove(1)
    assert [r["path"] for r in engine._queue_rows()] == [
        "https://cdn.example/1.mp3", "https://cdn.example/3.mp3"]
    engine.clear()
    assert engine._queue_rows() == []
    if rung.seam:
        engine._enqueue("https://cdn.example/9.mp3", "url", "Nine")
        rows = engine._queue_rows()
        doc = rung.raw("media_queue", rows[0]["id"])
        assert doc is not None and doc["path"] == "https://cdn.example/9.mp3"


# ── image index (tools/imagedb.py) ───────────────────────────────────────────

def test_image_index_conformance(rung: Rung, tmp_path):
    from nomorals.tools import imagedb

    image = tmp_path / "pic.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 64)
    ctx = _ctx(rung.db, settings=SimpleNamespace(
        workspace_dir=str(tmp_path)))
    result = imagedb._lookup_impl(ctx, str(image), "chat-1")
    assert result["ok"] is True and result["seen_before"] is False
    digest = result["hash"]
    again = imagedb._lookup_impl(ctx, str(image), "chat-2")
    assert again["ok"] is True and again["seen_before"] is True
    assert again["seen_where"] == "chat-1"
    dupes = imagedb.find_dupes(ctx)
    assert dupes["ok"] is True  # scanned count depends on Pillow/dhash
    stats = imagedb.image_stats(ctx)
    assert stats["ok"] is True and stats["count"] == 1
    if rung.seam:
        doc = rung.raw("image_index", digest)
        assert doc is not None and doc["_id"] == digest
        assert doc["seen_in"] == "chat-2"


# ── carry-v1 round-trip + the P3 completeness pin ────────────────────────────

def test_carry_round_trip_shapes():
    """document_to_row inverts row_to_document for the ported shapes:
    an INTEGER primary key reads back as an int, an ``expires_at`` data
    column round-trips through ``sql_expires_at``, JSON text stays a
    string, and a blob stays tagged."""
    db = Database(":memory:")
    db.migrate()
    mappings = {m.table: m for m in carry.inventory(db)}

    watchers = mappings["watchers"]
    row = {"id": "w9", "name": "n", "kind": "url", "target": "{}",
           "condition": "{}", "interval_s": 60.0, "cooldown_s": 0.0,
           "severity": "info", "channels": "[]", "quiet_hours": "{}",
           "expires_at": 123.5, "state": "active", "last_check": 0.0,
           "last_value": None, "last_alert_ts": 0.0,
           "last_alert_severity": "", "error_streak": 0, "last_changed": 0,
           "flip_times": "[]", "created_at": 1.0, "updated_at": 2.0}
    doc_id, fields, managed = carry.row_to_document(watchers, row)
    # the stored document, as the seam assembles it from the triple
    doc = {"_id": doc_id, **fields, "expires_at": managed}
    assert doc_id == "w9"
    assert doc["sql_expires_at"] == 123.5 and doc["expires_at"] is None
    back = carry.document_to_row(watchers, doc)
    assert back == row

    # a blob column rides as the tagged base64 envelope, and back
    synthetic = carry.TableMapping(
        table="synthetic", collection="synthetic",
        columns=[carry.Column("id", "INTEGER", 1),
                 carry.Column("payload", "BLOB", 0),
                 carry.Column("note", "TEXT", 0)])
    srow = {"id": 7, "payload": b"\x00\x01\x02", "note": "n"}
    sid, sfields, smanaged = carry.row_to_document(synthetic, srow)
    sdoc = {"_id": sid, **sfields, "expires_at": smanaged}
    assert sid == "7"  # INTEGER pk stringifies into _id
    assert sdoc["payload"] == {"$blob_b64": "AAEC"}
    assert carry.document_to_row(synthetic, sdoc) == srow


def test_p3_scope_completeness(tmp_path):
    """The P3 decision record is complete and consistent: every ported
    table is real (present once migrations and the organs' own runtime
    ensures have run), ported and residue never overlap, and every
    residue entry carries its reason. (Tables outside the census's P3
    assignment belong to other phases and are not P3's to classify.)"""
    from nomorals.agents import news as news_mod
    from nomorals.agents.trial.flow import TrialFlow
    from nomorals.agents.watchers import WatcherStore
    from nomorals.memory.manager import CoreBlocks
    from nomorals.social.chat.side_chats import SideChatManager

    db = Database(":memory:")
    db.migrate()
    # runtime tables created by ported organs on first use
    SideChatManager(db)
    CoreBlocks(db)
    WatcherStore(db)
    news_mod._ensure_attribution_columns(db)
    news_mod._ensure_delivery_columns(db)
    settings = SimpleNamespace(home=str(tmp_path), timezone="UTC",
                               workspace_dir=str(tmp_path))
    TrialFlow(_ctx(db, settings=settings))

    tables = {m.table for m in carry.inventory(db)}
    unported_real = set(carry.PORTED_TABLES) - tables
    assert unported_real == set(), (
        f"ported tables that do not exist: {sorted(unported_real)}")
    overlap = set(carry.PORTED_TABLES) & set(carry.PORT_RESIDUE)
    assert overlap == set(), f"ported AND residue: {sorted(overlap)}"
    for table, reason in carry.PORT_RESIDUE.items():
        assert reason.strip(), f"residue without a reason: {table}"
    # the headline domains of the census P3 list are all ported
    for table in ("conversations", "messages", "threads", "watchers",
                  "monitors", "news_items", "research_jobs", "search_log",
                  "page_cache", "training_runs", "datasets",
                  "benchmark_runs", "canary_runs", "improvement_runs",
                  "evolution_outcomes", "acceptance_runs",
                  "trial_assist_runs", "media_playlists", "media_queue",
                  "image_index", "trade_journal", "paper_sessions",
                  "live_unlocks", "memory_core_blocks"):
        assert table in carry.PORTED_TABLES, f"{table} not ported"
