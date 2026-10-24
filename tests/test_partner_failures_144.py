"""Pass #144: partner runtime failures, second wave.

Pass #102 built the agents/partner ledger and converted the first
sites; #144 converts the 40 remaining silent swallows in
nomorals/agents/partner/ (8 sites stay pinned exclusions — see
SURVEY.md in nm144-verify: already-recorded shutdown hook,
format-probe fall-through, expected-absent default, parse probe,
tour probes, try-the-other-verb, fail-closed gate).

Pins: durability of the REUSED partner ledger (reload round-trip,
200-cap on reload, clear-gone-on-reload, namespace isolation,
both save-failure shapes incl. the #141 broken-home lesson,
concurrent records, disk hits, subprocess-restart set) +
converted-site behavior pins proving fallback value unchanged AND
site recorded.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner import failures as partner_failures
from nomorals.agents.partner.failures import (
    clear_failures,
    failure_count,
    recent_failures,
    record_failure,
)
from nomorals.core.failure_store import Store

TREE = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


@pytest.fixture()
def _tmp_home(monkeypatch, tmp_path):
    """Isolated NM_HOME + freshly unloaded ledger state."""
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    partner_failures._ledger.clear()
    partner_failures._total = 0
    partner_failures._loaded = False
    yield tmp_path
    partner_failures._ledger.clear()
    partner_failures._total = 0
    partner_failures._loaded = False


def _reset_for_reload():
    """Force the next access to reload from disk."""
    partner_failures._ledger.clear()
    partner_failures._total = 0
    partner_failures._loaded = False


def _run_sub(code, home):
    env = dict(os.environ, NM_HOME=str(home), PYTHONPATH=str(TREE))
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          cwd=str(TREE), capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


_READ = (
    "from nomorals.agents.partner.failures import "
    "failure_count, recent_failures\n"
    "import json\n"
    "print(json.dumps({'count': failure_count(), "
    "'sites': [f.site for f in recent_failures(limit=1000)]}))\n"
)


# --- durability, in-process (isolated home) ---

def test_ledger_round_trip_via_reload(_tmp_home):
    record_failure("runtime_meta.test", ValueError("a"))
    record_failure("runtime_system.test", KeyError("b"))
    record_failure("runtime_search.test", RuntimeError("c"))
    _reset_for_reload()
    assert failure_count() == 3
    sites = [f.site for f in recent_failures()]
    assert sites == ["runtime_search.test", "runtime_system.test",
                     "runtime_meta.test"]  # newest-first


def test_ledger_cap_200_on_reload(_tmp_home):
    for _ in range(250):
        record_failure("runtime_meta.test", ValueError("x"))
    _reset_for_reload()
    assert failure_count() == 250
    assert len(recent_failures(limit=1000)) == 200


def test_ledger_clear_gone_on_reload(_tmp_home):
    record_failure("runtime_meta.test", ValueError("x"))
    record_failure("runtime_meta.test", ValueError("y"))
    clear_failures()
    _reset_for_reload()
    assert failure_count() == 0
    assert recent_failures() == []


def test_ledger_namespace_isolation(_tmp_home):
    record_failure("runtime_meta.test", ValueError("x"))
    assert Store("agents.arena").load() == ([], 0)
    entries, total = Store("agents.partner").load()
    assert total == 1 and entries[0]["site"] == "runtime_meta.test"


def test_store_save_fails_on_file_home(_tmp_home):
    blocker = _tmp_home / "blocker"
    blocker.write_text("x", encoding="utf-8")
    os.environ["NM_HOME"] = str(blocker)  # home is a file, not a dir
    try:
        assert Store("agents.partner").save(
            [{"ts": 1.0, "site": "s", "exc_type": "E", "message": "m"}],
            1) is False
        assert Store("agents.partner").load() == ([], 0)
    finally:
        os.environ["NM_HOME"] = str(_tmp_home)


def test_ledger_broken_home_records_in_memory(monkeypatch):
    """#141 lesson, ledger level: broken expanduser must degrade,
    never raise out of record_failure."""
    monkeypatch.delenv("NM_HOME", raising=False)
    real = Path.expanduser

    def _broken(self):
        if str(self).startswith("~"):
            raise RuntimeError("no home")
        return real(self)

    monkeypatch.setattr(Path, "expanduser", _broken)
    record_failure("runtime_meta.test", ValueError("boom"))  # must not raise
    assert failure_count() == 1
    assert recent_failures()[0].site == "runtime_meta.test"
    assert Store("agents.partner").save([], 0) is False


def test_ledger_concurrent_records(_tmp_home):
    def _work():
        for _ in range(25):
            record_failure("runtime_meta.test", ValueError("x"))

    threads = [threading.Thread(target=_work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failure_count() == 200
    assert len(recent_failures(limit=1000)) == 200
    doc = json.loads((_tmp_home / "failures" / "agents.partner.json")
                     .read_text(encoding="utf-8"))
    assert doc["total"] == 200 and len(doc["entries"]) == 200


def test_ledger_disk_hit(_tmp_home):
    record_failure("runtime_system.test", KeyError("missing"))
    doc = json.loads((_tmp_home / "failures" / "agents.partner.json")
                     .read_text(encoding="utf-8"))
    assert doc["total"] == 1
    assert doc["entries"][0]["site"] == "runtime_system.test"
    assert doc["entries"][0]["exc_type"] == "KeyError"


# --- durability, subprocess restarts (fresh process = fresh load) ---

def test_subprocess_round_trip(_tmp_home):
    _run_sub("from nomorals.agents.partner.failures import record_failure\n"
             "record_failure('runtime_meta.test', ValueError('a'))\n"
             "record_failure('runtime_system.test', KeyError('b'))\n",
             _tmp_home)
    out = json.loads(_run_sub(_READ, _tmp_home))
    assert out["count"] == 2
    assert out["sites"] == ["runtime_system.test", "runtime_meta.test"]


def test_subprocess_cap_200(_tmp_home):
    _run_sub("from nomorals.agents.partner.failures import record_failure\n"
             "for _ in range(250):\n"
             "    record_failure('runtime_meta.test', ValueError('x'))\n",
             _tmp_home)
    out = json.loads(_run_sub(_READ, _tmp_home))
    assert out["count"] == 250 and len(out["sites"]) == 200


def test_subprocess_clear_gone(_tmp_home):
    _run_sub("from nomorals.agents.partner.failures import "
             "record_failure, clear_failures\n"
             "record_failure('runtime_meta.test', ValueError('x'))\n"
             "clear_failures()\n", _tmp_home)
    out = json.loads(_run_sub(_READ, _tmp_home))
    assert out["count"] == 0 and out["sites"] == []


def test_subprocess_corrupt_recovery(_tmp_home):
    _run_sub("from nomorals.agents.partner.failures import record_failure\n"
             "record_failure('runtime_meta.test', ValueError('x'))\n",
             _tmp_home)
    (_tmp_home / "failures" / "agents.partner.json").write_text(
        "{garbage", encoding="utf-8")
    out = json.loads(_run_sub(_READ, _tmp_home))
    assert out["count"] == 0
    _run_sub("from nomorals.agents.partner.failures import record_failure\n"
             "record_failure('runtime_meta.test', ValueError('y'))\n",
             _tmp_home)
    out = json.loads(_run_sub(_READ, _tmp_home))
    assert out["count"] == 1


# --- converted-site behavior: fallback unchanged AND recorded ---

def _sites():
    return [f.site for f in recent_failures()]


def test_converted_gcal_line_keeps_raw_and_records():
    from nomorals.agents.partner.runtime_system import RuntimeSystemMixin

    out = RuntimeSystemMixin._gcal_line(
        {"summary": "Standup", "start": {"dateTime": "not-a-date"}}, "UTC")
    assert out == "not-a-date — Standup"
    assert "runtime_system._gcal_line" in _sites()
    good = RuntimeSystemMixin._gcal_line(
        {"summary": "Standup", "start": {"dateTime": "2026-10-24T14:00:00+00:00"}},
        "UTC")
    assert "14:00" in good and "Standup" in good


def test_converted_notion_title_falls_back_and_records():
    from nomorals.agents.partner.runtime_system import RuntimeSystemMixin

    assert RuntimeSystemMixin._notion_title_of(None) == "(untitled)"
    assert "runtime_system._notion_title_of" in _sites()
    page = {"properties": {"Name": {"type": "title", "title": [
        {"plain_text": "My Page"}]}}}
    assert RuntimeSystemMixin._notion_title_of(page) == "My Page"


def test_converted_model_status_line_na_and_records():
    from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

    def _boom():
        raise RuntimeError("router down")

    inst = object.__new__(RuntimeMetaMixin)
    inst.context = SimpleNamespace(router=SimpleNamespace(stats_snapshot=_boom))
    assert inst._model_status_line() == "n/a"
    assert "runtime_meta._model_status_line" in _sites()


def test_converted_wisdom_ask_status_fallback_and_records():
    from nomorals.agents.partner.runtime_wisdom import RuntimeWisdomMixin

    class _Keeper:
        def status(self):
            raise RuntimeError("corpus unreadable")

        def ask(self, query, top=5):
            return SimpleNamespace(synthesis="SYN", passages=[])

    inst = object.__new__(RuntimeWisdomMixin)
    inst._send_long_checked = lambda platform, chat, text: text
    out = inst._control_wisdom_ask(_Keeper(), "when was the earth created?",
                                   SimpleNamespace(platform="telegram"))
    assert "SYN" in out
    assert "corpus is empty" in out
    assert "runtime_wisdom._control_wisdom_ask" in _sites()


def test_converted_notify_row_excerpt_falls_back_and_records(monkeypatch):
    import nomorals.social.chat.present as present
    from nomorals.agents.partner.runtime_schedule import _notify_row_excerpt

    def _raise(text):
        raise RuntimeError("doc rebuild broke")

    monkeypatch.setattr(present, "doc_from_json", _raise)
    out = _notify_row_excerpt({"doc_json": "{}", "body": "hello\nbody"})
    assert out == "hello body"
    assert "runtime_schedule._notify_row_excerpt" in _sites()


def test_converted_provision_offer_denies_and_records():
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    def _boom(message):
        raise RuntimeError("operator check broke")

    inst = object.__new__(RuntimeProvisionMixin)
    inst._is_operator = _boom
    out = inst._provision_offer_for_missing(
        SimpleNamespace(service="x"), retry=("a", "b"),
        chat_key="telegram:1", message=object())
    assert out == ""
    assert "runtime_provision._provision_offer_for_missing" in _sites()
