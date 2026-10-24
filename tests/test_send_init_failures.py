"""Pass #139: send ``__init__`` failures stop vanishing.

``nomorals/marketing/send.py`` is a never-raises engine, and its
construction degraded silently in three places:

- ``__init__`` tracker build: a present-but-broken cost-tracker
  constructor left the engine untracked with no trace — the same
  silently-disabled shape as the failed cost-tracker import
  converted in #137. The explicit-absent case (``_CostTracker is
  None``) is guarded, so the except fires only when a present
  constructor raises: ``None`` on that path is a stale envelope,
  not the honest absent case.
- ``__init__`` sweep column: a column ALTER that failed for any
  reason other than the routine duplicate-column answer was
  swallowed. This one is a SPLIT, not a pure conversion: the
  duplicate-column error is the migration's idempotency mechanism
  — it fires on every restart of an already-migrated database —
  so it stays quiet, while any other ALTER failure records. The
  split is pinned on both branches against real sqlite3 behaviour.
- ``__init__`` sweep commit: a commit that never landed left the
  engine running on a half-migrated queue table with no trace —
  the engine proceeds as if the migration succeeded.

All three now record into the shared marketing ledger
(``nomorals/marketing/failures.py``, reused from #100 — no new
ledger); returns and control flow are unchanged (the engine still
builds, the tracker is still ``None`` on failure, the sweep still
passes through in all cases).

Deliberately NOT in scope here (no new evidence): the ``__init__``
DB handler (``_log.debug(..., exc_info=True)``, proven by walking
Call keywords — pinned below by the no-record control), the 3
debug-honest sites elsewhere (``render_template``, ``process``),
and the 4 pinned exclusions from #137 (``parse_dsn``,
``_dns_txt`` socket/parse-continues, ``deliverability_check``
inner).
"""

import ast
import sqlite3
from pathlib import Path

import nomorals.marketing.send as send_mod
import nomorals.tools.error_scan as es
from nomorals.marketing.failures import clear_failures, recent_failures
from nomorals.marketing.send import SendEngine
from nomorals.tools.error_scan import scan


def setup_function():
    clear_failures()


def teardown_function():
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(1000)]


TRACKER_SITE = "send.SendEngine.__init__.tracker"
COLUMN_SITE = "send.SendEngine.__init__.sweep_column"
SWEEP_SITE = "send.SendEngine.__init__.sweep"


# ── tracker: a broken constructor is not the absent case ──

def test_tracker_constructor_failure_records_but_tracker_still_none(
        monkeypatch, tmp_path):
    class _BoomTracker:
        def __init__(self):
            raise RuntimeError("boom-tracker")

    monkeypatch.setattr(send_mod, "_CostTracker", _BoomTracker)
    eng = SendEngine(db_path=str(tmp_path / "eng.db"))
    assert eng._tracker is None
    assert _sites() == [TRACKER_SITE]


def test_tracker_benign_path_records_nothing(tmp_path):
    eng = SendEngine(db_path=str(tmp_path / "eng.db"))
    assert eng._db is not None
    assert _sites() == []


# ── sweep column: duplicate is the probe answer, anything else records ──

def test_sweep_duplicate_column_stays_quiet(tmp_path):
    db = str(tmp_path / "eng.db")
    SendEngine(db_path=db)  # first migration: columns added, no record
    clear_failures()
    eng2 = SendEngine(db_path=db)  # second run: both ALTERs duplicate
    assert eng2._db is not None
    assert _sites() == []


class _LockedDB:
    """DB handle whose ALTERs fail locked; everything else no-ops."""

    def executescript(self, *args, **kwargs):
        return None

    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    def commit(self, *args, **kwargs):
        return None


def test_sweep_nondup_alter_failure_records_per_column(monkeypatch):
    # Isolate the sweep: the real _CostTracker opens sqlite3 itself,
    # so a global connect patch would leak into its constructor (the
    # tracker handler has its own pins above).
    monkeypatch.setattr(send_mod, "_CostTracker", None)
    monkeypatch.setattr(send_mod.sqlite3, "connect",
                        lambda *args, **kwargs: _LockedDB())
    eng = SendEngine(db_path="whatever.db")
    assert eng._db is not None
    # One record per failed column, both at the column site; the
    # commit never ran, so the outer sweep handler never fires.
    assert _sites() == [COLUMN_SITE, COLUMN_SITE]


class _CommitBoomDB:
    """First commit succeeds (the __init__ DB block); the sweep commit fails."""

    def __init__(self):
        self.commits = 0

    def executescript(self, *args, **kwargs):
        return None

    def execute(self, *args, **kwargs):
        return None

    def commit(self, *args, **kwargs):
        self.commits += 1
        if self.commits >= 2:
            raise RuntimeError("boom-commit")


def test_sweep_commit_failure_records_but_engine_still_builds(monkeypatch):
    # Isolate the sweep (see the locked-DB pin above).
    monkeypatch.setattr(send_mod, "_CostTracker", None)
    stub = _CommitBoomDB()
    monkeypatch.setattr(send_mod.sqlite3, "connect",
                        lambda *args, **kwargs: stub)
    eng = SendEngine(db_path="whatever.db")
    assert eng._db is stub
    assert stub.commits == 2
    assert _sites() == [SWEEP_SITE]


# ── control: the debug-honest DB handler stays unrecorded ──

def test_init_db_failure_records_nothing(monkeypatch, tmp_path):
    # Isolate the DB handler: the real _CostTracker opens sqlite3
    # itself, so a global connect patch would leak into its
    # constructor (the tracker handler has its own pins above).
    monkeypatch.setattr(send_mod, "_CostTracker", None)

    def _boom_connect(*args, **kwargs):
        raise RuntimeError("boom-connect")

    monkeypatch.setattr(send_mod.sqlite3, "connect", _boom_connect)
    eng = SendEngine(db_path=str(tmp_path / "eng.db"))
    assert eng._db is None
    assert _sites() == []


# ── structural + scan pins ──

def _init_handlers():
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SendEngine":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    return [n for n in ast.walk(sub)
                            if isinstance(n, ast.ExceptHandler)]
    raise AssertionError("SendEngine.__init__ not found")


def _record_sites(handler):
    return [n.args[0].value for n in ast.walk(handler)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "record_failure"
            and n.args and isinstance(n.args[0], ast.Constant)]


def _has_debug_exc_info(handler):
    for n in ast.walk(handler):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg == "exc_info":
                    return True
    return False


def test_ast_init_sites():
    handlers = _init_handlers()
    assert len(handlers) == 4, "DB + tracker + column + sweep"
    # Identify by content, not walk order: the DB handler is the
    # unbound debug-log one; the rest carry their own record site.
    by_site = {}
    db_h = None
    for h in handlers:
        sites = _record_sites(h)
        if not sites:
            db_h = h
        else:
            assert len(sites) == 1
            by_site[sites[0]] = h
    assert by_site.keys() == {TRACKER_SITE, COLUMN_SITE, SWEEP_SITE}
    tracker_h, column_h, sweep_h = (by_site[TRACKER_SITE],
                                    by_site[COLUMN_SITE], by_site[SWEEP_SITE])
    # DB handler untouched: unbound, debug log with exc_info, no record.
    assert db_h is not None and db_h.name is None
    assert _has_debug_exc_info(db_h)
    # Tracker: bound, exactly one record at the tracker site.
    assert tracker_h.name == "e"
    # Column: bound, record guarded by the duplicate-column probe check.
    assert column_h.name == "e"
    probe_checks = [n for n in ast.walk(column_h)
                    if isinstance(n, ast.Compare)
                    and any(isinstance(c, ast.Constant)
                            and c.value == "duplicate column"
                            for c in ast.walk(n))]
    assert probe_checks, "column handler must test the duplicate answer"
    # Sweep: bound, exactly one record at the sweep site.
    assert sweep_h.name == "e"


def _enclosing_func(line):
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    best = "<module>"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                best = node.name
    return best


def test_scan_send_remainder_is_exactly_7(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 10 remainder after #138 - 3 __init__ converts in #139
    # = 7 remainder: 3 debug-honest (render_template, __init__ DB
    # handler, process) + 4 pinned exclusions (parse_dsn, _dns_txt
    # socket-continue, _dns_txt parse-continue, deliverability_check
    # inner) + 0 untouched sites: the send.py estate is now closed
    # except for pinned judgments.
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 7
    funcs = sorted(_enclosing_func(f.line) for f in rep.findings)
    assert funcs == sorted(["render_template", "parse_dsn", "_dns_txt",
                            "_dns_txt", "deliverability_check", "__init__",
                            "process"])


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
