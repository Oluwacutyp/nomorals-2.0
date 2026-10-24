"""Pins for Pass #114: the betdesk failure ledger.

Every converted site is driven through the real function; fallbacks
are asserted held and the ledger entry asserted recorded. The two
deliberate probe exclusions are asserted as the exact subsystem
remainder (by file + function, never by line number).
"""

import ast
import json
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.agents.betdesk.failures import (
    Failure,
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(500)]


# ── ledger basics ──────────────────────────────────────────────────────

def test_ledger_basics_and_bound():
    assert failure_count() == 0
    assert recent_failures() == []
    assert format_failures() == "no betdesk failures recorded."
    record_failure("betdesk.pin", ValueError("boom"))
    assert failure_count() == 1
    assert recent_failures()[0].site == "betdesk.pin"
    assert recent_failures()[0].exc_type == "ValueError"
    assert "boom" in format_failures()
    for i in range(250):
        record_failure(f"site.{i}", RuntimeError(str(i)))
    assert failure_count() == 251
    kept = recent_failures(500)
    assert len(kept) == 200
    assert kept[0].site == "site.249"
    assert kept[-1].site == "site.50"


def test_ledger_unprintable_exc():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    record_failure("betdesk.pin", BadStr())
    entry = recent_failures()[0]
    assert "unprintable" in entry.message


def test_direct_import_surface():
    import nomorals.agents.betdesk.failures as f

    assert f.__all__ == ["Failure", "record_failure", "recent_failures",
                         "failure_count", "clear_failures", "format_failures"]
    assert issubclass(Failure, object)


# ── real converted sites ───────────────────────────────────────────────

def test_books_move_done_failure(tmp_path):
    from nomorals.agents.betdesk.books import SessionBookCollector

    book = SessionBookCollector(str(tmp_path), None)
    os.makedirs(book.inbox, exist_ok=True)
    path = os.path.join(book.inbox, "round.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{}")
    with mock.patch("os.replace", side_effect=OSError("disk full")):
        book._move_done("round.json", path)
    assert os.path.exists(path)  # the file stays, per the docstring
    assert _sites() == ["books._move_done"]


def test_collectors_record_failures():
    from nomorals.agents.betdesk.collectors import _record, _record_domain

    class Raising:
        def for_domain(self, domain):
            return object()

        def record_outcome(self, *a, **k):
            raise RuntimeError("registry down")

    spec = SimpleNamespace(domain="example.com")
    _record(Raising(), spec, True, "")
    _record_domain(Raising(), "example.com", True)
    assert _sites() == ["collectors._record_domain", "collectors._record"]

    class Healthy:
        def for_domain(self, domain):
            return object()

        def record_outcome(self, *a, **k):
            return None

    clear_failures()
    _record(Healthy(), spec, True, "")
    _record_domain(Healthy(), "example.com", True)
    assert _sites() == []


def test_desk_notifier_construction_failure(tmp_path):
    from nomorals.agents.betdesk.desk import BettingDesk

    key = "nomorals.agents.notifier"
    saved = sys.modules.get(key)
    stub = types.ModuleType(key)  # no Notifier attribute -> ImportError
    sys.modules[key] = stub
    try:
        desk = BettingDesk(context=SimpleNamespace(), desk_dir=str(tmp_path))
    finally:
        if saved is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved
    assert desk.signals.notifier is None
    assert _sites() == ["desk.__init__.notifier"]


def test_desk_library_corrupt(tmp_path):
    from nomorals.agents.betdesk.desk import BettingDesk

    desk = BettingDesk(desk_dir=str(tmp_path))
    with open(desk._library_path(), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert desk.library() == {}
    assert _sites() == ["desk.library"]
    clear_failures()
    with open(desk._library_path(), "w", encoding="utf-8") as fh:
        json.dump({"a/b": {"verdict": "X"}}, fh)
    assert desk.library() == {"a/b": {"verdict": "X"}}
    assert _sites() == []


def test_desk_status_sports_fixtures(tmp_path, monkeypatch):
    from nomorals.agents.betdesk import sources as sources_mod
    from nomorals.agents.betdesk.desk import BettingDesk

    monkeypatch.setattr(sources_mod, "ensure_provisioning", lambda: [])
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)

    class FakeRegistry:
        def by_kind(self, kind):
            return []

    key = "nomorals.agents.sports_bet"
    saved = sys.modules.get(key)

    class RaisingBetStore:
        def __init__(self):
            raise RuntimeError("store down")

    stub = types.ModuleType(key)
    stub.BetStore = RaisingBetStore
    sys.modules[key] = stub
    try:
        desk = BettingDesk(desk_dir=str(tmp_path))
        desk._registry = FakeRegistry()
        st = desk.status()
    finally:
        if saved is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved
    assert st["sports_fixtures"] is None
    assert "desk.status.sports_fixtures" in _sites()

    class HealthyBetStore:
        def fixtures(self):
            return [1, 2]

    stub2 = types.ModuleType(key)
    stub2.BetStore = HealthyBetStore
    sys.modules[key] = stub2
    try:
        clear_failures()
        st = desk.status()
    finally:
        if saved is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved
    assert st["sports_fixtures"] == 2
    assert _sites() == []


def test_desk_provisioning_view_vault_failure(monkeypatch):
    from nomorals.agents.betdesk.desk import _provisioning_view

    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)

    class RaisingVault:
        def list_all(self, **k):
            raise RuntimeError("vault down")

    rows = _provisioning_view(RaisingVault())
    assert all(not r["key_present"] for r in rows)
    # two services, each listing the vault -> two recorded entries
    assert _sites() == ["desk._provisioning_view"] * 2
    clear_failures()

    class EmptyVault:
        def list_all(self, **k):
            return []

    _provisioning_view(EmptyVault())
    assert _sites() == []


def test_sources_persist_sites():
    from nomorals.agents.betdesk import sources as sources_mod

    class RaisingPersist:
        def __init__(self, stale):
            self._by_key = {}
            self._stale = stale

        def for_domain(self, domain):
            return self._stale

        def register(self, desc):
            return True

        def _persist(self):
            raise OSError("disk full")

    stale = SimpleNamespace(kinds=("other",), search_url_template="x",
                            auth_needed=False)
    out = sources_mod.ensure_betting_sources(RaisingPersist(stale))
    assert out and all(r.get("updated") for r in out)
    sites = _sites()
    assert "sources.ensure_betting_sources.persist_update" in sites
    assert "sources.ensure_betting_sources.persist_register" not in sites
    clear_failures()

    class RegisterOnly(RaisingPersist):
        def for_domain(self, domain):
            return None

    out = sources_mod.ensure_betting_sources(RegisterOnly(None))
    assert out and all(r.get("registered") for r in out)
    sites = _sites()
    assert "sources.ensure_betting_sources.persist_register" in sites
    assert "sources.ensure_betting_sources.persist_update" not in sites


def test_sources_registry_bind_failure():
    from nomorals.agents.betdesk import sources as sources_mod
    from nomorals.media.source_registry import DEFAULT_REGISTRY

    class BadDbContext:
        @property
        def db(self):
            raise RuntimeError("db gone")

    reg = sources_mod.betting_registry_for_context(BadDbContext())
    assert reg is DEFAULT_REGISTRY
    assert _sites() == ["sources.betting_registry_for_context"]
    clear_failures()
    assert sources_mod.betting_registry_for_context(None) is DEFAULT_REGISTRY
    assert _sites() == []


# ── subsystem remainder: exactly the two deliberate probes ─────────────

def _silent_handlers_without_record():
    """(file, function) for pass-only except handlers with no record call."""
    import pathlib

    base = pathlib.Path(__file__).resolve().parent.parent / \
        "nomorals" / "agents" / "betdesk"
    found = set()
    for fname in ("books.py", "collectors.py", "desk.py", "sources.py"):
        tree = ast.parse((base / fname).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    if not isinstance(sub, ast.ExceptHandler):
                        continue
                    body = [s for s in sub.body
                            if not (isinstance(s, ast.Expr)
                                    and isinstance(s.value, ast.Constant)
                                    and isinstance(s.value.value, str))]
                    if body and all(isinstance(s, ast.Pass) for s in body):
                        has_record = any(
                            isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name)
                            and n.func.id == "record_failure"
                            for n in ast.walk(sub))
                        if not has_record:
                            found.add((fname, node.name))
    return found


def test_subsystem_remainder_is_exactly_the_two_probes():
    assert _silent_handlers_without_record() == {
        ("collectors.py", "_coerce_ts"),        # float->datetime fall-through
        ("sources.py", "_extract_hex_api_key"),  # JS->text-scan fall-through
    }
