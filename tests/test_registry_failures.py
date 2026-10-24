"""Pins for Pass #126: ToolRegistry bookkeeping failures land in the
nomorals.tools.registry_failures ring ledger.

Converted (8): the failure-ledger method's own fingerprint + insert
handlers, family attribution, idempotency table/load/decode/store,
and approval minting. Deliberate exclusions (2, pinned as behaviour):
the _args_digest fallbacks reproduce the canonical digest
byte-identically — same value by another method, probe class.
Debug-honest (stay, 2): sanitize_tool_description + _emit_tool_call
log _log.debug(..., exc_info=True).
"""
import ast
import sys
import types
from types import SimpleNamespace

import pytest

from nomorals.tools import registry as reg_mod
from nomorals.tools.registry import ToolRegistry
from nomorals.tools.registry_failures import (
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
    return [f.site for f in recent_failures(50)]


class FakeDb:
    """Scriptable stand-in for the registry's context db."""

    def __init__(self, execute=None, query_one=None):
        self._execute = execute
        self._query_one = query_one
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if self._execute is not None:
            return self._execute(sql, params)
        return None

    def query_one(self, sql, params=()):
        if self._query_one is not None:
            return self._query_one(sql, params)
        return None


def _reg(db=None, policy=None):
    ctx = SimpleNamespace(db=db, policy=policy)
    return ToolRegistry(context=ctx)


# ── ledger basics ───────────────────────────────────────────────────

def test_ledger_basics_and_format():
    assert format_failures() == "no recorded registry failures"
    record_failure("s1", ValueError("boom"))
    record_failure("s2", KeyError("k"))
    assert failure_count() == 2
    assert [f.site for f in recent_failures(2)] == ["s2", "s1"]
    assert "s2: KeyError: 'k'" in format_failures()
    clear_failures()
    assert failure_count() == 0
    assert recent_failures() == []


def test_ledger_bound_keeps_200():
    for _ in range(250):
        record_failure("s", RuntimeError("x"))
    assert failure_count() == 250
    assert len(recent_failures(500)) == 200


def test_ledger_unprintable_exc_safe():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("s", BadStr())
    assert "<unprintable RuntimeError>" in recent_failures()[0].message


def test_direct_import_surface_and_all_unchanged():
    import nomorals.tools.registry_failures as lf

    assert lf.format_failures() == "no recorded registry failures"
    assert reg_mod.__all__ == ["ToolRegistry", "ToolSpec", "ToolHealth",
                              "sanitize_tool_description", "annotations_for_kind"]


# ── converted sites ─────────────────────────────────────────────────

def test_record_failure_ledger_fingerprint_failure_records(monkeypatch):
    import nomorals.core.errors as core_errors

    db = FakeDb()
    reg = _reg(db=db)
    monkeypatch.setattr(core_errors, "failure_fingerprint",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fp broke")))
    reg._record_failure_ledger("mytool", "died", "traceback-text")
    assert _sites() == ["ToolRegistry._record_failure_ledger:fingerprint"]
    # The row still lands — but unfingerprinted, invisible to recurrence
    # checks keyed on fingerprint. That invisibility is the failure.
    assert db.executed[0][1][7] == ""
    assert reg._recording is False


def test_record_failure_ledger_fingerprint_ok_no_record():
    db = FakeDb()
    reg = _reg(db=db)
    reg._record_failure_ledger("mytool", "died", "traceback-text")
    assert _sites() == []
    assert db.executed[0][1][7] != ""


def test_record_failure_ledger_insert_failure_records():
    db = FakeDb(execute=lambda sql, params: (_ for _ in ()).throw(RuntimeError("db down")))
    reg = _reg(db=db)
    reg._record_failure_ledger("mytool", "died", "traceback-text")  # must not raise
    assert _sites() == ["ToolRegistry._record_failure_ledger:insert"]
    assert reg._recording is False  # finally resets even on insert failure


def test_failure_family_fallback_records(monkeypatch):
    import nomorals.core.errors as core_errors

    monkeypatch.setattr(core_errors, "categorize",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cat broke")))
    assert ToolRegistry._failure_family("some error") == "tool"
    assert _sites() == ["ToolRegistry._failure_family"]


def test_idempotency_table_failure_records():
    db = FakeDb(execute=lambda sql, params: (_ for _ in ()).throw(RuntimeError("create broke")))
    reg = _reg(db=db)
    assert reg._idempotency_table() is False
    assert _sites() == ["ToolRegistry._idempotency_table"]


def test_idempotency_table_absent_db_no_record():
    reg = _reg(db=None)
    assert reg._idempotency_table() is False
    assert _sites() == []


def test_idempotency_load_query_failure_records():
    db = FakeDb(query_one=lambda sql, params: (_ for _ in ()).throw(RuntimeError("query broke")))
    reg = _reg(db=db)
    reg._idempotency_db_ready = True  # skip CREATE; exercise the query path
    assert reg._idempotency_load("t", "k") is None
    assert _sites() == ["ToolRegistry._idempotency_load:query"]


def test_idempotency_load_miss_no_record():
    db = FakeDb(query_one=lambda sql, params: None)
    reg = _reg(db=db)
    reg._idempotency_db_ready = True
    assert reg._idempotency_load("t", "k") is None
    assert _sites() == []


def test_idempotency_load_decode_failure_records():
    row = {"actor": "a", "tool_version": "1", "ts": 0.0, "outcome_json": "not-json{{"}
    db = FakeDb(query_one=lambda sql, params: row)
    reg = _reg(db=db)
    reg._idempotency_db_ready = True
    assert reg._idempotency_load("t", "k") is None
    # Isolation: the query succeeded, so only the decode site records.
    assert _sites() == ["ToolRegistry._idempotency_load:decode"]


def test_idempotency_store_failure_records_memory_kept():
    from nomorals.core.result import Ok
    from nomorals.tools.registry import _IdempotencyRecord

    db = FakeDb(execute=lambda sql, params: (_ for _ in ()).throw(RuntimeError("insert broke")))
    reg = _reg(db=db)
    reg._idempotency_db_ready = True  # skip CREATE; exercise the INSERT path
    rec = _IdempotencyRecord(tool="t", key="k", actor="a", args_digest="d",
                             tool_version="1", outcome=Ok("v"), created_at=1.0)
    reg._idempotency_store(rec)
    assert reg._idempotency_records[("t", "k")] is rec  # memory still holds it
    assert _sites() == ["ToolRegistry._idempotency_store"]


def test_request_approval_failure_records(monkeypatch):
    import nomorals.core.policy as core_policy

    reg = _reg(db=None, policy=object())
    reg.register("gated", lambda: "ok", confirm="biometric")
    monkeypatch.setattr(core_policy, "approve_with_biometric",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bio broke")))
    assert reg.request_approval("gated") is None
    assert _sites() == ["ToolRegistry.request_approval"]


def test_request_approval_no_biometric_no_record():
    reg = ToolRegistry()
    reg.register("plain", lambda: "ok", confirm=False)
    assert reg.request_approval("plain") is None
    assert _sites() == []


# ── deliberate exclusions: _args_digest fallbacks are same-value ───

def test_args_digest_fallback_digest_identical(monkeypatch):
    from nomorals.core.oversight import args_digest as canonical

    kwargs = {"b": 2, "a": [1, {"z": "y"}]}
    # Normal kwargs: fallback digest == canonical digest.
    stub = types.ModuleType("nomorals.core.oversight")
    monkeypatch.setitem(sys.modules, "nomorals.core.oversight", stub)
    assert ToolRegistry._args_digest(kwargs) == canonical(kwargs)

    # Dumps-failing kwargs: both paths take the repr branch identically.
    class BadStr:
        def __str__(self):
            raise TypeError("boom")

    weird = {"x": BadStr()}
    assert ToolRegistry._args_digest(weird) == canonical(weird)
    # Neither fallback path records: the value produced is identical.
    assert _sites() == []


# ── scan pins ───────────────────────────────────────────────────────

def test_scan_remainder_is_exactly_debug_plus_exclusions():
    from nomorals.tools import error_scan as es

    path = reg_mod.__file__
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([path])
    finally:
        import importlib

        importlib.reload(es)
    tree = ast.parse(open(path).read())
    func_of = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for t in ast.walk(node):
                if isinstance(t, ast.Try):
                    for h in t.handlers:
                        func_of[h.lineno] = node.name
    found = sorted((f.rule, func_of.get(f.line)) for f in report.findings
                   if f.line in func_of)
    assert found == sorted([
        ("E104", "sanitize_tool_description"),
        ("E104", "_emit_tool_call"),
        ("E104", "_args_digest"),
        ("E104", "_args_digest"),
    ])
    # Each debug-honest body carries exc_info=True (Call-keyword walk,
    # not an ast.dump string grep — the #125 classifier lesson).
    for name in ("sanitize_tool_description", "_emit_tool_call"):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                for t in ast.walk(node):
                    if isinstance(t, ast.Try):
                        for h in t.handlers:
                            kws = [kw.arg for n in ast.walk(h)
                                   for kw in getattr(n, "keywords", [])]
                            assert "exc_info" in kws, name


def test_ledger_file_clean():
    from nomorals.tools import error_scan as es
    import nomorals.tools.registry_failures as lf

    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([lf.__file__])
    finally:
        import importlib

        importlib.reload(es)
    assert report.findings == []
