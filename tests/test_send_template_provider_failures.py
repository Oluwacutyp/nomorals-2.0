"""Pass #135: send templates/providers failures stop vanishing.

``nomorals/marketing/send.py`` is a never-raises engine, and its
templates/providers region returned the "empty config" shape on DB
failures with no trace: a failed template save returned ``False``
like an invalid name, a failed template/provider list returned ``[]``
like "nothing configured yet", a failed template lookup returned
``None`` like "no such template", and a failed provider registration
returned ``False`` while silently leaving the provider unregistered.
On a dead config store the engine looked unconfigured rather than
broken. Five sites in the region now record into the shared marketing
ledger (``nomorals/marketing/failures.py``, reused from #100 — no new
ledger); returns and control flow are unchanged.

Deliberately NOT in scope here: the module helpers (12), the
``__init__`` silent sites (3), ``spam_check_template``,
``status``, and ``get_engine`` — the remainder the scan pin below
counts (7 sites: 10 after #138 minus the 3 __init__ sites
converted in #139).

Each site's explicit-absence path is handled before the DB work
(``self._db is None`` / empty name returns the same fallback), so
the except fires only on a genuine DB failure returning that same
fallback — the #124 discriminator's residual-only case, not a probe.
"""

import ast
import tempfile

import pytest

import nomorals.tools.error_scan as es
from nomorals.marketing.failures import clear_failures, recent_failures
from nomorals.marketing.send import SendEngine
from nomorals.tools.error_scan import scan


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(1000)]


class _BoomDB:
    """DB handle whose execute/commit always explode."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("boom-db")

    def commit(self, *args, **kwargs):
        raise RuntimeError("boom-commit")


def _boom_engine():
    e = SendEngine.__new__(SendEngine)
    e._db = _BoomDB()
    e._tracker = None
    e._callables = {}
    return e


def _eng():
    return SendEngine(db_path=tempfile.mktemp(suffix=".db"))


EXPECTED_SITES = [
    "send.SendEngine.save_template",
    "send.SendEngine.list_templates",
    "send.SendEngine.get_template",
    "send.SendEngine.register_provider",
    "send.SendEngine.list_providers",
]


def test_all_five_sites_record():
    # Structural pin: every converted site has a record_failure call
    # with its exact site name as the first statement of its handler.
    src = open("nomorals/marketing/send.py").read()
    tree = ast.parse(src)
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "record_failure"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            found.add(node.args[0].value)
    for site in EXPECTED_SITES:
        assert site in found, f"missing record site {site}"


# ── save_template: a dead template write is not an invalid name ──────

def test_save_template_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.save_template("launch", "Hi!", "Launch") is False
    assert _sites() == ["send.SendEngine.save_template"]


def test_save_template_explicit_absence_records_nothing():
    # Empty name is the designed-absence path, not a failure.
    e = _boom_engine()
    assert e.save_template("", "Hi!") is False
    assert _sites() == []


# ── list_templates: [] on a dead DB is not "no templates yet" ─────────

def test_list_templates_db_failure_records_but_still_returns_empty():
    e = _boom_engine()
    assert e.list_templates() == []
    assert _sites() == ["send.SendEngine.list_templates"]


def test_list_templates_benign_path_records_nothing():
    e = _eng()
    assert e.list_templates() == []
    assert e.save_template("launch", "Hi {{ name }}!", "Launch day") is True
    names = [t["name"] for t in e.list_templates()]
    assert names == ["launch"]
    assert _sites() == []


# ── get_template: None on a dead DB is not "no such template" ─────────

def test_get_template_db_failure_records_but_still_returns_none():
    e = _boom_engine()
    assert e.get_template("launch") is None
    assert _sites() == ["send.SendEngine.get_template"]


def test_get_template_benign_path_records_nothing():
    e = _eng()
    assert e.get_template("missing") is None
    assert e.save_template("launch", "Hi {{ name }}!", "Launch day") is True
    got = e.get_template("launch")
    assert got["body"] == "Hi {{ name }}!" and got["subject"] == "Launch day"
    assert _sites() == []


# ── register_provider: a dead registration is not a declined one ──────

def test_register_provider_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.register_provider("mock", kind="mock", rate_per_minute=100) is False
    assert _sites() == ["send.SendEngine.register_provider"]


def test_register_provider_benign_path_records_nothing():
    e = _eng()
    assert e.register_provider("mock", kind="mock", rate_per_minute=100) is True
    names = [p["name"] for p in e.list_providers()]
    assert names == ["mock"]
    assert _sites() == []


# ── list_providers: [] on a dead DB is not "no providers yet" ─────────

def test_list_providers_db_failure_records_but_still_returns_empty():
    e = _boom_engine()
    assert e.list_providers() == []
    assert _sites() == ["send.SendEngine.list_providers"]


def test_list_providers_benign_path_records_nothing():
    e = _eng()
    assert e.list_providers() == []
    assert _sites() == []


def test_scan_send_remainder_is_exactly_7(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 10 remainder after #138 - 3 __init__ converts in #139
    # = 7 remainder: 3 debug-honest (render_template, __init__ DB
    # handler, process) + 4 pinned exclusions (parse_dsn, _dns_txt
    # socket-continue, _dns_txt parse-continue, deliverability_check
    # inner) + 0 untouched sites: send.py is closed except for
    # pinned judgments.
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 7


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
