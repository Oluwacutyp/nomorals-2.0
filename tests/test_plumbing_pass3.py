"""Pass 3 targeted fixes — dead organs & broken plumbing.

One production-path test group per row of the Pass 3 grouping in
~/workspace/next-families-scope-2026-11-06.md (plus the scheduler
_parse_depends by-catch).  TEETH vs PINS are marked per test: TEETH
fail on the pristine base (e86bf5f) for the row's own reason; PINs
pass on both trees and hold behaviour the fix must not change.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest

# ── 2B.14 osint_browser: the browser OSINT adapters are wired ────


class _FakeBrowserRegistry:
    """A ToolRegistry-shaped double carrying a working browser tool."""

    def __init__(self):
        self.calls = []

    def get(self, name):
        return object() if name == "browser" else None

    def call(self, name, actor="system", **kw):
        self.calls.append((name, actor, kw))
        action = kw.get("action")
        if action == "open":
            value = {"ok": True}
        elif action == "text":
            value = {"text": "profile https://example.com/jane here",
                     "url": "https://webmii.com/people?n=Jane+Doe"}
        elif action == "extract":
            value = {"links": [{
                "href": "https://web.archive.org/web/2020/https://example.com/",
                "text": "cap"}]}
        else:
            value = {}
        return SimpleNamespace(ok=True, value=value, error=None)


def test_osint_people_adapter_searches_via_registry(monkeypatch):
    # TEETH: on pristine, _browser() looks up a module-level
    # ``registry.get`` that does not exist -> None -> the adapter
    # returns [] unconditionally.
    from nomorals.search import osint_browser as ob

    monkeypatch.setattr(ob, "_rendered_ok", lambda: (True, ""))
    reg = _FakeBrowserRegistry()
    adapter = ob.BrowserPeopleSearchAdapter(reg)
    hits = adapter.search("Jane Doe", limit=3)
    assert len(hits) >= 1
    assert hits[0].title.startswith("Jane Doe")
    # the calls really went through registry.call("browser", ...)
    assert reg.calls and all(c[0] == "browser" for c in reg.calls)
    assert reg.calls[0][2]["action"] == "open"
    assert reg.calls[0][2]["session"] == "osint"


def test_osint_federated_search_threads_context_registry(monkeypatch):
    # TEETH (production path): federated_search hands the context's
    # own tool registry to the lazily built browser adapters.
    from nomorals.search import federated as federated_mod
    from nomorals.search import osint_browser as ob

    monkeypatch.setattr(ob, "_rendered_ok", lambda: (True, ""))
    reg = _FakeBrowserRegistry()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/tmp/ws"), tools=reg)
    resp = federated_mod.federated_search(
        "Jane Doe", context=ctx, sources=["osint_people_browser"],
        limit=3)
    assert resp.sources_searched == ["osint_people_browser"]
    assert len(resp.hits) >= 1
    assert reg.calls, "browser tool never invoked through the registry"


def test_osint_adapter_without_browser_tool_returns_empty(monkeypatch):
    # PIN: absence is still the failure shape — a registry without
    # the browser tool leaves the adapters returning [], never a
    # fabricated hit (passes on both trees in spirit; the constructor
    # argument itself is new, so this runs on the fixed tree).
    from nomorals.search import osint_browser as ob
    from nomorals.tools.registry import ToolRegistry

    monkeypatch.setattr(ob, "_rendered_ok", lambda: (True, ""))
    adapter = ob.BrowserPeopleSearchAdapter(ToolRegistry())
    assert adapter.search("Jane Doe", limit=3) == []


# ── 2.14 forge: the unified-plan branch in create() is live ──────


class _PlanResp:
    ok = True

    def __init__(self, t):
        self.text = t


def test_forge_create_uses_unified_plan():
    # TEETH: on pristine, create() imports plan_book from .outline
    # (which has no such function) -> ImportError -> the plan branch
    # is dead and the fallback outline path runs instead, so the
    # planned title/chapters never land.
    import nomorals.books.forge as F

    payload = json.dumps({"title": "Trust Again", "chapters": [
        {"title": "One", "beats": ["a"]}, {"title": "Two", "beats": ["b"]},
        {"title": "Three", "beats": ["c"]}]})

    class _Brain:
        def chat(self, msgs, params, task_kind=None):
            return _PlanResp(payload)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
        router=object())
    with mock.patch("nomorals.llm.power.model_usable", lambda c: True), \
            mock.patch.object(F, "brain_for", lambda c: _Brain()):
        book = F.BookForge(ctx).create("rebuilding trust after betrayal",
                                       chapters=3, research=False)
    assert book.title == "Trust Again"
    assert [c.title for c in book.chapters] == ["One", "Two", "Three"]


# ── 1.4 route: stops persist and restore ─────────────────────────


def test_route_stops_persist_and_restore(tmp_path):
    # TEETH (both halves): on pristine, add_stop's 5-value INSERT
    # into the 7-column table fails (nothing persists) and the
    # restore loop's row["stop_id"] raises TypeError on the
    # row_factory-less connection (nothing restores).
    from nomorals.planning import route

    db_file = tmp_path / "routes.db"
    p1 = route.RoutePlanner(db_path=str(db_file))
    stop = p1.add_stop("corner shop", lat=6.5, lng=3.3, dwell_minutes=20)
    assert stop is not None
    conn = sqlite3.connect(str(db_file))
    rows = conn.execute(
        "SELECT stop_id, label, lat, lng, dwell_minutes "
        "FROM route_stops").fetchall()
    conn.close()
    assert len(rows) == 1 and rows[0][1] == "corner shop"
    assert p1.set_window("corner shop", "16:00", "17:30") is True

    p2 = route.RoutePlanner(db_path=str(db_file))
    found = p2.find_stop("corner shop")
    assert found is not None
    assert found.lat == 6.5 and found.lng == 3.3
    assert found.window is not None


# ── 1.5 congestion: the profile consult really runs ─────────────


def test_congestion_profile_kind_detects_termux(monkeypatch):
    # TEETH: on pristine, the three-dot import always raises, the
    # fallback answers from NM_PROFILE/"workstation", and a termux
    # device without NM_PROFILE set gets the full model.
    from nomorals.planning import congestion as cg

    monkeypatch.delenv("NM_PROFILE", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    assert cg._profile_kind() == "termux"
    assert cg._full_model() is False


def test_congestion_profile_kind_env_pin_still_wins(monkeypatch):
    # PIN: an explicit NM_PROFILE is honoured on both trees.
    from nomorals.planning import congestion as cg

    monkeypatch.setenv("NM_PROFILE", "workstation")
    assert cg._profile_kind() == "workstation"
    assert cg._full_model() is True


# ── 1.7 scheduler: add() dedupes an exact replay ─────────────────


def _sched_ctx():
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    tools = SimpleNamespace(
        call=lambda tool, capabilities=None, **kw2: SimpleNamespace(
            ok=True, value="ok", error=None))
    return SimpleNamespace(db=db, tools=tools,
                           settings=SimpleNamespace(), extras={})


def test_scheduler_add_exact_replay_dedupes():
    # TEETH: on pristine, add() inserts unconditionally — the replay
    # mints a second job id and the table holds two copies.
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_sched_ctx())
    first = sched.add("pulse", "every 1h", "message", {"text": "hi"})
    replay = sched.add("pulse", "every 1h", "message", {"text": "hi"})
    assert replay["id"] == first["id"]
    assert replay.get("deduped") is True
    assert len(sched.list_jobs()) == 1


def test_scheduler_add_distinct_jobs_still_insert():
    # PIN: dedupe is exact-replay only — a different spec or payload
    # under the same name is a different job (both trees insert it;
    # the id assertions hold on pristine too, the count differs
    # there only for the replay case above).
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_sched_ctx())
    a = sched.add("pulse", "every 1h", "message", {"text": "hi"})
    b = sched.add("pulse", "every 2h", "message", {"text": "hi"})
    c = sched.add("pulse", "every 1h", "message", {"text": "other"})
    assert len({a["id"], b["id"], c["id"]}) == 3
    assert len(sched.list_jobs()) == 3


# ── by-catch: _parse_depends gate inversion ──────────────────────


def test_dependency_gate_open_for_garbage_stored_depends():
    # TEETH: on pristine, the stored garbage "[not json" parses to
    # ["[not json"], the lookup misses, and _dependency_ok keeps the
    # gate shut forever. After the fix it parses to [] (no
    # dependencies) and the gate is open, per the docstring contract.
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_sched_ctx())
    job = sched.add("gated", "every 1h", "message", {"text": "x"})
    sched.db.execute(
        "UPDATE schedule_jobs SET depends_on = ? WHERE id = ?",
        ("[not json", job["id"]))
    row = sched._find(job["id"])
    assert sched._dependency_ok(row) is True


def test_dependency_gate_shut_for_deleted_dependency():
    # PIN: a well-formed dependency on a job that does not exist
    # keeps the gate shut on both trees ("a deleted dependency keeps
    # the gate shut" is the documented, intended half).
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_sched_ctx())
    job = sched.add("gated", "every 1h", "message", {"text": "x"})
    sched.db.execute(
        "UPDATE schedule_jobs SET depends_on = ? WHERE id = ?",
        (json.dumps(["no-such-job-id"]), job["id"]))
    row = sched._find(job["id"])
    assert sched._dependency_ok(row) is False


# ── 2B.12 trading_desk: the daily-loss gate fails closed ─────────

_NOW = 1_700_000_000.0


class _FlakyConnector:
    """Snapshot works for the position-count call, then goes down."""

    def __init__(self):
        self.calls = 0

    def get_snapshot(self):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("snapshot down")
        return {"positions": [],
                "account_state": {"equity": 10000.0}}

    def get_account_details(self):
        return {"account": "demo"}


class _GoodDeskConnector:
    def get_snapshot(self):
        return {"positions": [],
                "account_state": {"equity": 10000.0}}

    def get_account_details(self):
        return {"account": "demo"}


def _live_desk(tmp_path, connector, name):
    from nomorals.finance.trading_desk import TradingDesk

    return TradingDesk(connector, mode="live",
                       journal_path=tmp_path / f"{name}.jsonl",
                       state_path=tmp_path / f"{name}.json",
                       now=lambda: _NOW)


def test_check_risk_refuses_when_daily_pnl_unverifiable(tmp_path):
    # TEETH: on pristine, daily_pnl swallows the snapshot failure
    # and returns the journal-only total (0.0), so the gate never
    # sees the failure as a refusal — the desk's own DeskError
    # "cannot verify" path does not exist there.
    from nomorals.finance.trading_desk import DeskError

    desk = _live_desk(tmp_path, _FlakyConnector(), "flaky")
    with pytest.raises(DeskError, match="cannot verify daily"):
        desk._check_risk("XAUUSD", "buy", 0.01, 1990.0, 2030.0, 2000.0)


def test_check_risk_healthy_live_desk_passes(tmp_path):
    # PIN: a healthy live desk passes the gate exactly as before.
    desk = _live_desk(tmp_path, _GoodDeskConnector(), "good")
    facts = desk._check_risk("XAUUSD", "buy", 0.01, 1990.0, 2030.0,
                             2000.0)
    assert facts["instrument"] == "XAUUSD"
    assert desk.daily_pnl() == 0.0


def test_cli_renders_unknown_daily_pnl():
    # TEETH: on pristine, formatting a None daily_pnl with :+.2f
    # raises TypeError — the stats/risk lines crash instead of
    # rendering the honest "unknown".
    from nomorals.cmdline.commands.trade import _render_exness

    out = {"total_trades": 2, "win_rate": 0.5, "realized_pnl": 3.0,
           "daily_pnl": None}
    line = _render_exness("stats", out)
    assert "daily_pnl=unknown" in line
    risk = _render_exness("risk", {"policy": {}, "daily_pnl": None})
    assert "daily pnl=unknown" in risk
    # control: a real number still renders as a number
    out["daily_pnl"] = -1.5
    assert "daily_pnl=-1.50" in _render_exness("stats", out)


# ── 2B.13 app_builder: remove() carries the stop outcome ─────────


def _app_builder(tmp_path):
    from nomorals.builders import app_builder as AB
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(workspace_dir=str(tmp_path))
    # the served/deployed registries live in the context db's KV
    # store — without a db they read as permanently empty.
    return AB.AppBuilder(SimpleNamespace(settings=settings, db=db))


def _make_app(builder, slug="demo-app"):
    app_dir = builder._apps_dir / slug
    app_dir.mkdir(parents=True)
    (app_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return app_dir


def test_remove_reports_failed_stop(tmp_path):
    # TEETH: on pristine, the stop ToolError is swallowed and the
    # result is a bare {"removed": True} — no stopped/stop_errors
    # keys exist at all.
    from nomorals.core.errors import ToolError

    builder = _app_builder(tmp_path)
    app_dir = _make_app(builder)
    builder._set_served({"demo-app": {"pid": 0, "port": 0}})

    def stop_boom(name):
        raise ToolError("kill failed")

    builder.stop = stop_boom  # type: ignore[method-assign]
    result = builder.remove("demo-app")
    assert result["removed"] is True
    assert not app_dir.exists()
    assert result["stopped"] is False
    assert any("server" in e and "kill failed" in e
               for e in result["stop_errors"])


def test_remove_reports_still_alive_process(tmp_path):
    # TEETH: a stop that "succeeds" but leaves the process alive is
    # a failed stop, and the result must say so.
    builder = _app_builder(tmp_path)
    _make_app(builder)
    builder._set_served({"demo-app": {"pid": 424242, "port": 0}})
    builder.stop = lambda name: {"app": "demo-app", "stopped": False,
                                 "was_pid": 424242}  # type: ignore[method-assign]
    result = builder.remove("demo-app")
    assert result["removed"] is True
    assert result["stopped"] is False
    assert any("still alive" in e for e in result["stop_errors"])


def test_remove_clean_when_stop_lands(tmp_path):
    # PIN (fixed tree): the real stop path — a registered entry
    # whose pid is already dead stops cleanly, outcome honest.
    builder = _app_builder(tmp_path)
    _make_app(builder)
    builder._set_served({"demo-app": {"pid": 99999999, "port": 0}})
    result = builder.remove("demo-app")
    assert result["removed"] is True
    assert result["stopped"] is True
    assert result["stop_errors"] == []


# ── 2.20 matchmaking: the sweep honours the live-game guard ──────


def test_sweep_skips_players_in_live_chats():
    # TEETH: on pristine, sweep pairs everyone pairable — the
    # player whose chat has a live game gets a relay duel started
    # in it anyway (enqueue would have refused the mirror-image
    # situation).
    from types import SimpleNamespace as NS

    from nomorals.games import matchmaking as M
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    live: dict = {}  # chat -> live room; filled after enqueue below

    class _Relay:
        def __init__(self):
            self.invites = []

        def create_invite(self, chat, player, game):
            self.invites.append(chat)
            return NS(code=f"c{len(self.invites)}")

        def accept_invite(self, code, chat, player):
            return True

    sent = []
    eng = NS(db=db, relay=_Relay(),
             _send=lambda chat, text: sent.append(chat),
             live=lambda chat: live.get(chat))
    mm = M.Matchmaker(eng)
    pa = NS(key="a", name="A", platform="local")
    pb = NS(key="b", name="B", platform="local")
    pc = NS(key="c", name="C", platform="local")
    pd = NS(key="d", name="D", platform="local")
    mm.enqueue(pa, "ca", "pvp")
    mm.enqueue(pb, "cb", "pvp")
    mm.enqueue(pc, "cc", "pvp")
    # the gap the guard covers: a queued FIRST, then a game starts
    # in a's chat (enqueue itself would have refused had the game
    # already been live) — the sweep must not now start a duel there.
    live["ca"] = NS()
    assert mm.sweep() == 1
    # b and c were paired; a is still queued, untouched
    assert M.match_status(db, "a") is not None
    assert M.match_status(db, "b") is None
    assert M.match_status(db, "c") is None
    assert "ca" not in eng.relay.invites
    # once the live game ends, the next sweep pairs a normally
    live.clear()
    mm.enqueue(pd, "cd", "pvp")
    assert mm.sweep() == 1
    assert M.match_status(db, "a") is None


def test_sweep_pairs_normally_when_nothing_live():
    # PIN: with no live games the sweep pairs exactly as before
    # (passes on both trees).
    from types import SimpleNamespace as NS

    from nomorals.games import matchmaking as M
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()

    class _Relay:
        def create_invite(self, chat, player, game):
            return NS(code="x")

        def accept_invite(self, code, chat, player):
            return True

    eng = NS(db=db, relay=_Relay(), _send=lambda chat, text: None,
             live=lambda chat: None)
    mm = M.Matchmaker(eng)
    mm.enqueue(NS(key="a", name="A", platform="local"), "ca", "pvp")
    mm.enqueue(NS(key="b", name="B", platform="local"), "cb", "pvp")
    assert mm.sweep() == 1
    assert M.match_status(db, "a") is None
    assert M.match_status(db, "b") is None
