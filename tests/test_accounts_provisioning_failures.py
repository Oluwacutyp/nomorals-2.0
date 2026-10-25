"""Durability + behavior pins for Pass #199 (accounts/provisioning.py).

Durability pins mirror the #141+ shape: round-trip, 250-to-200 reload
with total preserved, clear-gone-on-reload, namespace isolation,
corrupt recovery, both save-failure shapes, concurrent saves, disk
hits, memory degradation, concurrent records, subprocess-restart,
broken-home at store AND ledger level. Site pins prove the three
converted teardown handlers record; NOT-recording pins prove the ten
exclusions stay silent.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.accounts import failures as cf
from nomorals.accounts import provisioning as pv
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

def _fresh_driver(**attrs):
    d = object.__new__(pv.ProvisionDriver)
    d.audit_trail = []
    d.captcha_solver = None
    d.manager = object()
    d.sessions = object()
    for k, v in attrs.items():
        setattr(d, k, v)
    return d


class _FakeTab:
    def __init__(self, *, close_exc=None, click_fail_labels=(),
                 fill_exc=False, text=""):
        self._close_exc = close_exc
        self._click_fail = set(click_fail_labels)
        self._fill_exc = fill_exc
        self._text = text

    def close(self):
        if self._close_exc is not None:
            raise self._close_exc

    def click(self, label):
        if label in self._click_fail:
            raise RuntimeError("click broken")
        return None

    def navigate(self, url):
        return None

    def fill(self, candidate, value):
        if self._fill_exc:
            raise RuntimeError("fill broken")
        return None

    def text(self, max_chars=20000):
        return {"text": self._text}


def _svc(**kw):
    base = dict(id="svc", display="Svc", vault_service="svc",
                signup_flow="fake_flow", signup_url="",
                flow="fake_flow", scopes=["read"], login_url="",
                token_url="https://huggingface.co/settings/tokens")
    base.update(kw)
    return SimpleNamespace(**base)


def _sites():
    return [f.site for f in cf.recent_failures(1000)]


def test_pin_attempt_signup_tab_close(monkeypatch):
    tab = _FakeTab(close_exc=RuntimeError("tab close down"))
    d = _fresh_driver(open_tab=lambda vs: tab)

    def _flow(*a, **k):
        raise pv.FlowStepError("stepx", "page moved")

    monkeypatch.setitem(pv._SIGNUP_FLOWS, "fake_flow", _flow)
    res = d._attempt_signup(_svc(), object(), name="n", email="e@x.y",
                            username="u", password="p", note="",
                            identity_kind="generated")
    assert res is not None
    assert res.outcome == pv.ProvisionOutcome.PAGE_CHANGED
    assert "provisioning.attempt_signup_tab_close" in _sites()


def test_pin_drive_tab_close(monkeypatch):
    tab = _FakeTab(close_exc=RuntimeError("tab close down"))
    d = _fresh_driver(open_tab=lambda vs: tab)
    d._login_cred = lambda s: SimpleNamespace(username="u")
    monkeypatch.setattr(pv, "ensure_login",
                        lambda *a, **k: {"username": "u"})

    def _flow(*a, **k):
        raise pv.FlowStepError("stepy", "page moved")

    monkeypatch.setitem(pv._PAGE_FLOWS, "fake_flow", _flow)
    res = d._drive_with_login(_svc(), account_created=True)
    assert res is not None
    assert res.outcome == pv.ProvisionOutcome.PAGE_CHANGED
    assert "provisioning.drive_tab_close" in _sites()


def test_pin_license_tab_close():
    tab = _FakeTab(close_exc=RuntimeError("tab close down"))
    d = _fresh_driver(open_tab=lambda vs: tab)
    step = SimpleNamespace(subject="model-x", url="http://x/",
                           agree_texts=("I agree",))
    notes = d._run_license_steps(_svc(licenses=[step]))
    assert len(notes) == 1
    assert "provisioning.license_tab_close" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_not_recording_json_name():
    res = pv._validate_hf_whoami("tok", lambda url, h: (200, "{not json"))
    assert res.status == "ok"
    assert res.detail == "token accepted for account"
    assert _sites() == []


def test_pin_not_recording_tab_text():
    class _BrokenText:
        def text(self, max_chars=20000):
            raise RuntimeError("tab text down")

    assert pv._tab_text(_BrokenText()) == ""
    assert _sites() == []


def test_pin_not_recording_click_shape():
    class _ClickDown:
        def click(self, label):
            raise RuntimeError("click down")

        def evaluate(self, js):
            return True

    assert pv._click_text(_ClickDown(), "Label") == "Label"
    assert _sites() == []


def test_pin_not_recording_click_label():
    class _AllDown:
        def click(self, label):
            raise RuntimeError("click down")

        def evaluate(self, js):
            raise RuntimeError("evaluate down")

    with pytest.raises(pv.FlowStepError):
        pv._click_text(_AllDown(), "Nope")
    assert _sites() == []


def test_pin_not_recording_captcha_detect():
    class _CheckerDown:
        def check_captcha(self):
            raise RuntimeError("checker down")

    assert pv._detect_captchas(_CheckerDown()) == []
    assert _sites() == []


def test_pin_not_recording_token_extract():
    class _ExtractDown:
        def evaluate(self, js):
            raise RuntimeError("evaluate down")

        def text(self, max_chars=20000):
            raise RuntimeError("text down")

    assert pv._extract_hf_token(_ExtractDown()) == ""
    assert _sites() == []


def test_pin_not_recording_name_fill():
    tab = _FakeTab(fill_exc=True)
    with pytest.raises(pv.FlowStepError) as err:
        pv.hf_token_flow(tab, _svc(), token_name="t")
    assert err.value.step == "name_token"
    assert _sites() == []


def test_pin_not_recording_role_click():
    tab = _FakeTab(click_fail_labels=("Read", "read"),
                   text="your token hf_abcdefghij1234567890 here")
    token = pv.hf_token_flow(tab, _svc(), token_name="t")
    assert token == "hf_abcdefghij1234567890"
    assert _sites() == []


def test_pin_not_recording_fill_first():
    tab = _FakeTab(fill_exc=True)
    with pytest.raises(pv.FlowStepError):
        pv._fill_first(tab, ("a", "b"), "v", step="s", required=True)
    assert pv._fill_first(tab, ("a", "b"), "v", step="s",
                          required=False) == ""
    assert _sites() == []


def test_pin_not_recording_coro_ferry():
    async def _boom():
        raise ValueError("inner boom")

    with pytest.raises(ValueError):
        pv._run_coro(_boom())

    async def _sys_exit():
        raise SystemExit(3)

    with pytest.raises(SystemExit) as err:
        pv._run_coro(_sys_exit())
    assert err.value.code == 3
    assert _sites() == []
