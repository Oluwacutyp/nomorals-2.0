"""Pass #181 pins: the REUSED llm failure ledger + brain conversions.

``nomorals/llm/swallow_failures.py`` (namespace ``llm``, from #165)
is reused — brain.py is an llm-subsystem module, so no new ledger file.
The 17 durability tests below mirror the #141+ shape exactly (same as
#180's, against the same Store backend); the 2 site pins prove each
conversion records without changing flow (fallback value unchanged AND
a ledger entry under its site name); the 15 NOT-recording pins prove
each exclusion stays silent with the designed output. No pin hardcodes
scan totals or brain counts. Every test redirects stores with the app's
own ``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import dataclasses
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.llm import swallow_failures as lf
from nomorals.llm import brain as br
from nomorals.llm.base import LLMResponse
from nomorals.llm.broker import BrokerConstraints
from nomorals.llm.capabilities import Capability


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    lf.clear_failures()
    yield
    lf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in lf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("llm")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("llm").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("llm")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("llm").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("llm")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("llm").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("llm").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("llm").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("llm")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("llm").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("llm").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("llm").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("llm").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("llm")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("llm").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        lf.record_failure("no-home", ValueError("boom"))
        assert lf.failure_count() == 1
        assert [f.site for f in lf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []
    assert lf.format_failures() == "no llm failures recorded."
    lf.record_failure("llm.site-a", ValueError("boom"))
    lf.record_failure("llm.site-b", RuntimeError("bang"))
    assert lf.failure_count() == 2
    assert [f.site for f in lf.recent_failures()] == ["llm.site-b", "llm.site-a"]
    assert [f.site for f in lf.recent_failures(1)] == ["llm.site-b"]
    text = lf.format_failures()
    assert "2 total" in text and "llm.site-a" in text and "llm.site-b" in text
    lf.clear_failures()
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    lf.record_failure("llm.disk", ValueError("boom"))
    entries, total = fs.Store("llm").load()
    assert total == 1 and entries[0]["site"] == "llm.disk"
    lf.clear_failures()
    assert fs.Store("llm").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    lf.record_failure("llm.mem", ValueError("boom"))
    assert lf.failure_count() == 1
    assert [f.site for f in lf.recent_failures()] == ["llm.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                lf.record_failure(f"llm.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert lf.failure_count() == 400
    assert len(lf.recent_failures(1000)) == 200
    entries, total = lf._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.llm import swallow_failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.llm import swallow_failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.llm import swallow_failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("llm").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site pins (converts) -------------------------------------------------

def test_pin_with_quality():
    bc = BrokerConstraints()
    with patch.object(dataclasses, "replace", side_effect=RuntimeError("boom")):
        out = br._with_quality(bc, 0.8)
    assert out is bc and out.min_quality == 0.0
    assert "brain._with_quality" in _sites()


def test_pin_classify():
    resp = LLMResponse(text="", error="timeout after 30s")
    with patch.object(br, "classify_failure", side_effect=RuntimeError("boom")):
        out = br.Brain(router=object())._classify(resp)
    assert out is resp and not (out.failure_class or "")
    assert "brain._classify" in _sites()


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_pin_fix_hint_not_recorded():
    with patch.object(br, "classify_failure", side_effect=RuntimeError("boom")):
        hint = br._fix_hint("provider 429 quota exceeded")
    assert "quota" in hint
    assert _sites() == []


class _BadResp:
    @property
    def error(self):
        raise RuntimeError("boom")

    @property
    def fallback_note(self):
        raise RuntimeError("boom")

    @property
    def failed_providers(self):
        raise RuntimeError("boom")


def test_pin_explain_failure_not_recorded():
    assert br.explain_failure(_BadResp()) == "Brain unavailable — no provider served this call."
    assert _sites() == []


def test_pin_llm_ladder_not_recorded():
    brain = br.Brain(router=object())
    with patch("nomorals.core.error_system.get_error_system",
               side_effect=RuntimeError("boom")):
        assert brain._llm_ladder() is None
    assert _sites() == []


class _BadRung:
    @property
    def honesty(self):
        raise RuntimeError("boom")


class _BadLadder:
    @property
    def rungs(self):
        return [_BadRung()]


def test_pin_ladder_failure_text_not_recorded():
    assert br.Brain._ladder_failure_text(_BadLadder(), RuntimeError("provider 503")) == "provider 503"
    assert _sites() == []


class _BadUsage:
    @property
    def prompt_tokens(self):
        raise RuntimeError("boom")


def test_pin_coerce_usage_not_recorded():
    u = br.Brain._coerce_usage(_BadUsage())
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (0, 0, 0)
    assert _sites() == []


class _BadText:
    @property
    def text(self):
        raise RuntimeError("boom")


def test_pin_coerce_response_not_recorded():
    r = br.Brain(router=object())._coerce_response("chat", _BadText())
    assert r.text == "" and r.error == "brain.chat returned no response"
    assert _sites() == []


def test_pin_summarize_for_fit_not_recorded():
    with patch.object(br.Brain, "complete", side_effect=RuntimeError("boom")):
        assert br.Brain(router=object())._summarize_for_fit("x") == ""
    assert _sites() == []


class _BadProviders:
    def providers(self):
        raise RuntimeError("boom")


def test_pin_best_of_not_recorded():
    brain = br.Brain(router=_BadProviders())
    with patch.object(br.Brain, "complete",
                      return_value=LLMResponse(text="single", error="")):
        text, judgment = brain.best_of("prompt", n=3)
    assert text == "single" and judgment.rationale == "single provider"
    assert _sites() == []


class _BadBroker:
    def escalation_chain(self, *a, **k):
        raise RuntimeError("boom")


class _ChainRouter:
    broker = _BadBroker()


def test_pin_escalate_not_recorded():
    brain = br.Brain(router=_ChainRouter())
    with patch.object(br.Brain, "complete",
                      return_value=LLMResponse(text="esc", error="")) as comp:
        out = brain.escalate("prompt")
    assert out.text == "esc" and comp.call_count == 1
    assert _sites() == []


class _BadActive:
    @property
    def active(self):
        raise RuntimeError("boom")


def test_pin_active_model_not_recorded():
    assert br.Brain(router=_BadActive()).active_model() == ""
    assert _sites() == []


def test_pin_diagnose_not_recorded():
    with patch.object(br.Brain, "status", side_effect=RuntimeError("boom")):
        assert br.Brain(router=object()).diagnose() == "brain: status unavailable (diagnose failed)"
    assert _sites() == []


class _BadCards:
    def cards(self):
        raise RuntimeError("boom")


class _CardRouter:
    broker = _BadCards()


def test_pin_is_owner_card_not_recorded():
    assert br.Brain._is_owner_card(_CardRouter(), "x") is False
    assert _sites() == []


def _select(cap, op):
    if op == "chat":
        raise RuntimeError("boom")
    return SimpleNamespace(id="pick-" + op)


class _PickBroker:
    select = staticmethod(_select)


class _PickRouter:
    broker = _PickBroker()


def test_pin_broker_picks_inner_not_recorded():
    picks = br.Brain._broker_picks(_PickRouter())
    assert "chat" not in picks and picks.get("vision") == "pick-vision"
    assert _sites() == []


class _BadBrokerProp:
    @property
    def broker(self):
        raise RuntimeError("boom")


def test_pin_broker_picks_outer_not_recorded():
    assert br.Brain._broker_picks(_BadBrokerProp()) == {}
    assert _sites() == []


class _NoSetCtx:
    def __init__(self, router):
        object.__setattr__(self, "router", router)

    def __setattr__(self, name, value):
        raise RuntimeError("frozen context")


def test_pin_brain_for_cache_not_recorded():
    got = br.brain_for(_NoSetCtx(object()))
    assert isinstance(got, br.Brain)
    assert _sites() == []
