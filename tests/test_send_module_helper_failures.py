"""Pass #137: send module-helper failures stop vanishing.

``nomorals/marketing/send.py``'s module helpers returned quiet fallback
values with no trace. Eight sites now record into the shared marketing
ledger (``nomorals/marketing/failures.py``, reused from #100 — no new
ledger); returns and control flow are unchanged:

* ``send.import._resolve_profile`` / ``send.import._CostTracker`` — the
  two module-level import fallbacks. These are FIRST-PARTY modules, not
  optional third-party deps: an import failure is breakage, not a
  configuration, and the ``None`` fallback is not an equivalent
  implementation (the #124 discriminator's probe case does not apply).
  A broken profile import degrades every batch decision to "pc"; a
  broken cost-tracker import silently disables cost tracking.
* ``send._default_db`` — a failed home path swaps the persistent queue
  DB for ``:memory:`` (staged sends vanish on restart), the #131
  cache-relocation shape.
* ``send._profile_kind`` — the explicit-absent case (resolver ``None``)
  is guarded; the except fires only when a present resolver raises,
  so "pc" on that path is a stale envelope, not the default.
* ``send.spam_score`` — the fallback names the fact ("unknown") but
  the score 0 reads as clean to score-keyed callers.
* ``send.warmup_plan`` — an empty plan uncaps the ramp:
  ``warmup_status`` falls back to the full daily target when the plan
  is empty, the #134 ``_warmup_ok`` fail-open twin.
* ``send._dns_txt`` (outer) — per-resolver failures are handled by the
  inner fall-throughs; the outer fires only on a setup/internal
  failure, and its ``[]`` then reads as "no TXT records".
* ``send.deliverability_check`` (outer) — a partial result shaped like
  a complete grade (initial score 0 / grade "F") is not the true grade.

Deliberately NOT converted (pinned below, each with its reason):

* ``parse_dsn`` — unparseable input IS the documented "unknown"
  verdict; this is a pure-parse probe over untrusted bounce text
  (#124 precedent). ``process_bounce`` acts only on "hard", and its
  own handler already records.
* ``_dns_txt`` per-resolver handlers — "try the other resolver"
  fall-throughs (#116 precedent); the value is still produced when
  the other resolver answers.
* ``deliverability_check`` DNS inner — the resolution failure is
  itself reported as a check (``ok: False``), lowering the grade:
  the failure is visible in the result, not a success shape.
"""

import ast
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import nomorals.tools.error_scan as es
from nomorals.marketing import send as send_mod
from nomorals.marketing.failures import clear_failures, recent_failures
from nomorals.marketing.send import (
    _default_db,
    _dns_txt,
    _profile_kind,
    deliverability_check,
    parse_dsn,
    spam_score,
    warmup_plan,
)
from nomorals.tools.error_scan import scan

CONVERTED_SITES = {
    "send.import._resolve_profile",
    "send.import._CostTracker",
    "send._default_db",
    "send._profile_kind",
    "send.spam_score",
    "send.warmup_plan",
    "send._dns_txt",
    "send.deliverability_check",
}


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(200)]


def _exactly_once(site):
    sites = _sites()
    assert sites.count(site) == 1, sites
    return sites


# ── import fallbacks (subprocess: fresh interpreter, import blocked) ──

_IMPORT_PROBE = r'''
import importlib.abc
import importlib.machinery
import sys

BLOCKED = {"nomorals.core.profile", "nomorals.social.whatsapp_cost"}

class _BlockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise ImportError("blocked for test")

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in BLOCKED:
            return importlib.machinery.ModuleSpec(name, _BlockLoader())
        return None

sys.meta_path.insert(0, _Blocker())
import nomorals.marketing.send as s
from nomorals.marketing.failures import recent_failures
print("resolve:", s._resolve_profile)
print("tracker:", s._CostTracker)
print("sites:", sorted(f.site for f in recent_failures(200)))
'''


def test_import_fallbacks_record_and_keep_fallback():
    tree = Path("nomorals/marketing/send.py").resolve().parent.parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        cwd=str(tree), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "resolve: None" in proc.stdout, proc.stdout
    assert "tracker: None" in proc.stdout, proc.stdout
    assert "send.import._resolve_profile" in proc.stdout, proc.stdout
    assert "send.import._CostTracker" in proc.stdout, proc.stdout


# ── _default_db ──


def test_default_db_failure_records_and_returns_memory():
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        assert _default_db() == ":memory:"
    _exactly_once("send._default_db")


def test_default_db_benign_returns_persistent_path_no_record():
    with patch("os.path.expanduser", return_value="/tmp/nm137fake"), \
            patch("os.makedirs"):
        path = _default_db()
    assert path.endswith("send_queue.db")
    assert _sites() == []


# ── _profile_kind ──


def test_profile_kind_resolver_failure_records_and_returns_pc():
    with patch.object(send_mod, "_resolve_profile",
                      side_effect=RuntimeError("resolver broken")):
        assert _profile_kind("") == "pc"
    _exactly_once("send._profile_kind")


def test_profile_kind_benign_paths_no_record():
    with patch.object(send_mod, "_resolve_profile", None):
        assert _profile_kind("") == "pc"
    assert _sites() == []
    from types import SimpleNamespace
    with patch.object(send_mod, "_resolve_profile",
                      return_value=SimpleNamespace(kind="Termux")):
        assert _profile_kind("") == "termux"
    assert _sites() == []
    assert _profile_kind("Workstation") == "workstation"
    assert _sites() == []


# ── spam_score ──


class _BoomFormat:
    def __format__(self, spec):
        raise RuntimeError("cannot format")


def test_spam_score_failure_records_and_returns_unknown():
    out = spam_score(_BoomFormat(), "")
    assert out == {"score": 0, "verdict": "unknown", "issues": []}
    _exactly_once("send.spam_score")


def test_spam_score_benign_no_record():
    out = spam_score("FREE!!! WINNER!!! click http://a.example http://b.example "
                     "http://c.example http://d.example", "x" * 60)
    assert out["score"] >= 30
    assert _sites() == []
    out = spam_score("hello", "short note")
    assert out["verdict"] == "clean"
    assert _sites() == []


# ── warmup_plan ──


class _BoomInt:
    def __int__(self):
        raise RuntimeError("cannot int")


def test_warmup_plan_failure_records_and_returns_empty():
    assert warmup_plan(_BoomInt()) == []
    _exactly_once("send.warmup_plan")


def test_warmup_plan_benign_no_record():
    plan = warmup_plan(100)
    assert len(plan) == 21
    assert plan[-1]["cap"] == 100
    assert _sites() == []


# ── _dns_txt: outer records, inners fall through silently ──


def test_dns_txt_setup_failure_records_and_returns_empty():
    assert _dns_txt(None) == []
    _exactly_once("send._dns_txt")


def test_dns_txt_resolver_failures_do_not_record():
    with patch("socket.socket", side_effect=OSError("network down")):
        assert _dns_txt("example.com") == []
    assert _sites() == []


class _FakeSocket:
    """Answers with a truncated packet: qid matches, answer section cut
    short, so the parse branch raises and the loop falls through."""

    def __init__(self, *args, **kwargs):
        self.pkt = b""

    def settimeout(self, timeout):
        pass

    def sendto(self, pkt, addr):
        self.pkt = pkt

    def recvfrom(self, n):
        qid = struct.unpack(">H", self.pkt[:2])[0]
        header = struct.pack(">HHHHHH", qid, 0x8000, 1, 1, 0, 0)
        return header + b"\x00", ("8.8.8.8", 53)

    def close(self):
        pass


def test_dns_txt_parse_failure_falls_through_without_record():
    with patch("socket.socket", side_effect=lambda *a, **k: _FakeSocket()):
        assert _dns_txt("example.com") == []
    assert _sites() == []


# ── parse_dsn probe exclusion ──


def test_parse_dsn_unparseable_is_unknown_without_record():
    out = parse_dsn("hello world, no bounce here")
    assert out["kind"] == "unknown"
    assert out["addresses"] == []
    assert _sites() == []
    out = parse_dsn("Delivery failed: 5.1.1 user unknown <a@b.com>")
    assert out["kind"] == "hard"
    assert out["addresses"] == ["a@b.com"]
    assert _sites() == []


# ── deliverability_check: inner honest, outer records ──


def test_deliverability_inner_reports_failure_without_record():
    with patch.object(send_mod, "_dns_txt", return_value=[]), \
            patch("socket.gethostbyname", side_effect=OSError("no dns")):
        res = deliverability_check("example.com")
    checks = {c["name"]: c for c in res["checks"]}
    assert checks["domain resolves"]["ok"] is False
    assert checks["domain resolves"]["detail"] == "no A record"
    assert _sites() == []


def test_deliverability_outer_records_and_returns_partial():
    with patch.object(send_mod, "_dns_txt",
                      side_effect=RuntimeError("dns dead")):
        res = deliverability_check("example.com")
    assert res == {"domain": "example.com", "score": 0, "grade": "F",
                   "checks": []}
    _exactly_once("send.deliverability_check")


# ── structural + scan pins ──


def test_ast_all_converted_sites_record():
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    sites = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "record_failure":
                if node.args and isinstance(node.args[0], ast.Constant):
                    sites.add(node.args[0].value)
    assert CONVERTED_SITES <= sites, CONVERTED_SITES - sites


def _enclosing_func(line):
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    best = "<module>"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                best = node.name
    return best


def test_scan_send_remainder_is_exactly_10(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 13 remainder after #137 - 3 tail converts in #138
    # = 10 remainder: 3 debug-honest (render_template, __init__,
    # process) + 4 pinned exclusions (parse_dsn, _dns_txt
    # socket-continue, _dns_txt parse-continue, deliverability_check
    # inner) + 3 untouched __init__-silent sites (the #139 candidate).
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 10
    funcs = sorted(_enclosing_func(f.line) for f in rep.findings)
    assert funcs == sorted(["render_template", "parse_dsn", "_dns_txt",
                            "_dns_txt", "deliverability_check", "__init__",
                            "__init__", "__init__", "__init__", "process"])


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
