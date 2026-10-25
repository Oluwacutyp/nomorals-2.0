"""Durability + behavior pins for Pass #200 (accounts/signup_driver.py).

Durability pins mirror the #141+ shape: round-trip, 250-to-200 reload
with total preserved, clear-gone-on-reload, namespace isolation,
corrupt recovery, both save-failure shapes, concurrent saves, disk
hits, memory degradation, concurrent records, subprocess-restart,
broken-home at store AND ledger level. The site pin proves the one
converted terminal verify-click handler records; NOT-recording pins
prove the twelve exclusions stay silent.
"""
import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.accounts import failures as cf
from nomorals.accounts import signup_driver as sd
from nomorals.core import failure_store as fs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    cf.clear_failures()
    yield
    cf.clear_failures()


def _entries(n, prefix):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": "boom"}
            for i in range(n)]


# ---- Durability (#141+ shape) -------------------------------------------

def test_store_round_trip():
    cf.record_failure("accounts.rt", ValueError("boom"))
    entries, total = fs.Store("accounts").load()
    assert total == 1 and len(entries) == 1
    assert entries[0]["site"] == "accounts.rt"


def test_store_250_to_200_reload_total_preserved():
    for i in range(250):
        cf.record_failure(f"accounts.ring{i}", ValueError("boom"))
    assert cf.failure_count() == 250
    assert len(cf.recent_failures(1000)) == 200
    entries, total = fs.Store("accounts").load()
    assert len(entries) == 200 and total == 250


def test_store_clear_gone_on_reload():
    cf.record_failure("accounts.tmp", ValueError("boom"))
    cf.clear_failures()
    assert fs.Store("accounts").load() == ([], 0)
    assert cf.failure_count() == 0


def test_store_namespace_isolation():
    fs.Store("accounts").save(_entries(2, "acct-"), 2)
    fs.Store("agents").save(_entries(3, "agent-"), 3)
    acct, acct_total = fs.Store("accounts").load()
    agent, agent_total = fs.Store("agents").load()
    assert acct_total == 2 and len(acct) == 2
    assert agent_total == 3 and len(agent) == 3
    assert {e["site"] for e in acct} == {"acct-0", "acct-1"}
    assert {e["site"] for e in agent} == {"agent-0", "agent-1", "agent-2"}


def test_store_corrupt_recovers():
    # Store-level recovery: a corrupt file reads back empty and the
    # store accepts fresh writes afterwards. (Ledger-level recovery
    # across a restart is covered by test_restart_corrupt_store_recovers;
    # in-process ledger memory is authoritative within its process, so
    # corrupting the file mid-process is not the ledger's scenario.)
    fs.Store("accounts").save(_entries(2, "pre-"), 2)
    fs.Store("accounts").path().write_text("{not json", encoding="utf-8")
    assert fs.Store("accounts").load() == ([], 0)
    assert fs.Store("accounts").save(_entries(1, "post-"), 1) is True
    entries, total = fs.Store("accounts").load()
    assert total == 1 and entries[0]["site"] == "post-0"


def test_store_save_failure_home_is_file(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("accounts").save(_entries(1, "x-"), 1) is False
    assert fs.write_failures() > before


def test_store_save_failure_expanduser_raises(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: (_ for _ in ()).throw(OSError("no home")))
    before = fs.write_failures()
    assert fs.Store("accounts").save(_entries(1, "x-"), 1) is False
    assert fs.write_failures() > before


def test_store_concurrent_saves_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            fs.Store("accounts").save(_entries(50, f"w{k}-"), 50)
        except Exception as exc:  # noqa: BLE001 - pins collect, never swallow
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("accounts").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


def test_ledger_record_hits_disk():
    cf.record_failure("accounts.disk", ValueError("boom"))
    entries, total = fs.Store("accounts").load()
    assert total == 1 and entries[0]["site"] == "accounts.disk"


def test_ledger_blocked_home_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    cf.record_failure("accounts.mem", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["accounts.mem"]
    assert fs.write_failures() > before


def test_ledger_broken_home_resolution_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: (_ for _ in ()).throw(OSError("no home")))
    before = fs.write_failures()
    cf.record_failure("accounts.nohome", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["accounts.nohome"]
    assert fs.write_failures() > before


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                cf.record_failure(f"accounts.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 - pins collect, never swallow
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cf.failure_count() == 400
    assert len(cf.recent_failures(1000)) == 200
    entries, total = cf._store.load()
    assert len(entries) == 200
    assert total == 400


def test_ledger_ring_cap_drops_oldest_silently():
    for i in range(250):
        cf.record_failure(f"accounts.ring{i}", ValueError("boom"))
    assert cf.failure_count() == 250
    ring = cf.recent_failures(1000)
    assert len(ring) == 200
    assert ring[0].site == "accounts.ring249"
    assert ring[-1].site == "accounts.ring50"


def test_ledger_api_record_recent_count_clear_format():
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []
    assert cf.format_failures() == "no accounts failures recorded."
    cf.record_failure("accounts.site-a", ValueError("boom"))
    cf.record_failure("accounts.site-b", RuntimeError("bang"))
    assert cf.failure_count() == 2
    assert [f.site for f in cf.recent_failures()] == [
        "accounts.site-b", "accounts.site-a"]
    assert [f.site for f in cf.recent_failures(1)] == ["accounts.site-b"]
    text = cf.format_failures()
    assert "2 total" in text
    assert "accounts.site-a" in text and "accounts.site-b" in text
    cf.clear_failures()
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


def _run_subprocess(home, code):
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n, clear_first=True):
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.accounts import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.accounts import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.accounts import failures as f\n"
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


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    corrupt = (
        "from pathlib import Path\n"
        "from nomorals.core import failure_store as fs\n"
        "fs.Store('accounts').path().write_text('{not json', encoding='utf-8')\n"
        "print('corrupted')\n")
    proc = _run_subprocess(home, corrupt)
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0 and data["sites"] == []
    proc = _run_subprocess(home, _record_code(1))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1 and data["sites"] == ["site0"]



# ---- Site pins (converts record, fallback behavior unchanged) -------------

def _sites():
    return [f.site for f in cf.recent_failures(1000)]


def _fresh_driver(**attrs):
    d = object.__new__(sd.SignupDriver)
    d.verify_timeout_s = 60.0
    d.verify_poll_s = 1.0
    d.slider_solve = None
    d.notify = None
    d.creator = None
    d.attempts = None
    for k, v in attrs.items():
        setattr(d, k, v)
    return d


class _Tab:
    def __init__(self, *, fill_ok=True, submit_ok=True, click_ok=True,
                 text="", html="", captcha=None):
        self._fill_ok = fill_ok
        self._submit_ok = submit_ok
        self._click_ok = click_ok
        self._text = text
        self._html = html
        self._captcha = captcha if captcha is not None else {}

    def fill(self, label, value):
        if not self._fill_ok:
            raise RuntimeError("no such field")
        return True

    def submit(self, target=""):
        if not self._submit_ok:
            raise RuntimeError("no form to submit")
        return True

    def click(self, target):
        if not self._click_ok:
            raise RuntimeError("no such button")
        return True

    def text(self, max_chars=40000):
        return self._text

    def html(self):
        return self._html

    def check_captcha(self, fetch_bytes=False):
        return self._captcha

    def navigate(self, url):
        raise RuntimeError("navigate down")


class _RaisingTab(_Tab):
    def text(self, max_chars=40000):
        raise RuntimeError("text down")

    def html(self):
        raise RuntimeError("html down")

    def check_captcha(self, fetch_bytes=False):
        raise RuntimeError("captcha check down")


def _poll_driver(tab, inbox):
    d = _fresh_driver()
    d._read_inbox = lambda cred, provider: inbox
    return d, tab


def test_pin_poll_verification_verify_click():
    inbox = [{"subject": "verify", "text": "your code is 123456", "html": ""}]
    d, tab = _poll_driver(
        _Tab(fill_ok=True, submit_ok=False, click_ok=False), inbox)
    got = asyncio.run(d._poll_verification(
        SimpleNamespace(service="github"), tab,
        SimpleNamespace(credential=object(), service="email_tempmail"),
        "github"))
    assert got is True
    assert "signup_driver.poll_verification_verify_click" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_not_recording_page_text():
    assert sd._page_text(_RaisingTab()) == ""
    assert _sites() == []


def test_pin_not_recording_page_html():
    assert sd._page_html(_RaisingTab()) == ""
    assert _sites() == []


def test_pin_not_recording_captcha_detected():
    assert sd._captcha_detected(_RaisingTab()) is False
    assert _sites() == []


def test_pin_not_recording_fill_submit_click_fallback():
    d = _fresh_driver()
    persona = SimpleNamespace(username_variants=["u1"], name="N",
                              password="pw", dob="2000-01-01")
    got = asyncio.run(d._fill_and_submit(
        SimpleNamespace(service="github"),
        _Tab(fill_ok=True, submit_ok=False, click_ok=False,
             text="", html="", captcha={}),
        persona, "u1@example.com", {}))
    assert got == "u1"
    assert _sites() == []


def test_pin_not_recording_poll_code_fill_dispatch():
    inbox = [{"subject": "verify", "text": "your code is 123456", "html": ""}]
    d, tab = _poll_driver(_Tab(fill_ok=False, submit_ok=True), inbox)
    got = asyncio.run(d._poll_verification(
        SimpleNamespace(service="github"), tab,
        SimpleNamespace(credential=object(), service="email_tempmail"),
        "github"))
    assert got is True
    assert _sites() == []


def test_pin_not_recording_poll_submit_click_fallback():
    inbox = [{"subject": "verify", "text": "your code is 123456", "html": ""}]
    d, tab = _poll_driver(
        _Tab(fill_ok=True, submit_ok=False, click_ok=True), inbox)
    got = asyncio.run(d._poll_verification(
        SimpleNamespace(service="github"), tab,
        SimpleNamespace(credential=object(), service="email_tempmail"),
        "github"))
    assert got is True
    assert _sites() == []


def test_pin_not_recording_captcha_challenge_fetch():
    class _CaptchaDown(_Tab):
        def check_captcha(self, fetch_bytes=False):
            raise RuntimeError("captcha down")

    d = _fresh_driver(creator=SimpleNamespace(
        attempt_captcha_solve=lambda **kw: "tok"))
    d._notify = lambda *a, **k: None
    got = asyncio.run(d._handle_captcha_wall(
        SimpleNamespace(service="github", id="a1"), _CaptchaDown(),
        "ev", "http://x"))
    assert got is None
    assert _sites() == []


def test_pin_not_recording_sms_phone_submit():
    calls = {"n": 0}

    def cascade():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "ok", "number": "5550001",
                    "masked": "***001", "provider": "p1"}
        return {"status": "error", "notes": "none"}

    d = _fresh_driver(creator=SimpleNamespace(
        get_temp_number_cascade=cascade,
        poll_sms_code=lambda *a, **k: "",
        checkpoints=SimpleNamespace(
            create=lambda *a, **k: SimpleNamespace(id="cp1", title="t"))))
    d.attempts = SimpleNamespace(set_wall=lambda *a, **k: None,
                                 transition=lambda *a, **k: None)
    d._notify = lambda *a, **k: None
    with pytest.raises(sd.AccountCheckpointPending):
        asyncio.run(d._handle_sms_wall(
            SimpleNamespace(service="github", id="a1",
                            stage=sd.SignupStage.STARTED,
                            persona_name="p", persona_id="pid"),
            _Tab(fill_ok=True, submit_ok=False), "ev", "http://x",
            owner_contacts={}))
    assert _sites() == []


def test_pin_not_recording_sms_code_submit():
    class _SmsTab(_Tab):
        def __init__(self):
            super().__init__(fill_ok=True, submit_ok=True,
                             text="", html="", captcha={})
            self.submits = 0

        def submit(self, target=""):
            self.submits += 1
            if self.submits == 2:
                raise RuntimeError("code submit down")
            return True

    d = _fresh_driver(creator=SimpleNamespace(
        get_temp_number_cascade=lambda: {
            "status": "ok", "number": "5550001",
            "masked": "***001", "provider": "p1"},
        poll_sms_code=lambda *a, **k: "123456",
        checkpoints=SimpleNamespace(
            create=lambda *a, **k: SimpleNamespace(id="cp1", title="t"))))
    d.attempts = SimpleNamespace(set_wall=lambda *a, **k: None,
                                 transition=lambda *a, **k: None)
    d._notify = lambda *a, **k: None
    got = asyncio.run(d._handle_sms_wall(
        SimpleNamespace(service="github", id="a1",
                        stage=sd.SignupStage.STARTED,
                        persona_name="p", persona_id="pid"),
        _SmsTab(), "ev", "http://x", owner_contacts={}))
    assert got is None
    assert _sites() == []


def test_pin_not_recording_rate_limit_navigate():
    d = _fresh_driver()
    d._notify = lambda *a, **k: None
    got = asyncio.run(d._handle_rate_limit_wall(
        SimpleNamespace(service="github", id="a1"),
        _Tab(text="", html="", captcha={}), "ev", "http://x",
        waits=(0,)))
    assert got is None
    assert _sites() == []


def test_pin_not_recording_fill_first():
    assert sd.SignupDriver._fill_first(
        _Tab(fill_ok=False), ("a", "b"), "v") is False
    assert sd.SignupDriver._fill_first(
        _Tab(fill_ok=True), ("a", "b"), "v") is True
    assert _sites() == []
